-- Agent Registry Table
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_name VARCHAR(50) PRIMARY KEY CHECK (agent_name ~ '^[a-zA-Z0-9_]+$'),
    table_prefix VARCHAR(100) NOT NULL,
    ingestion_folder TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT unique_table_prefix UNIQUE (table_prefix)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_agent_registry_created_at ON agent_registry (created_at DESC);

-- Function to create agent-specific tables
CREATE OR REPLACE FUNCTION create_agent_tables(agent_name TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    documents_table TEXT;
    chunks_table TEXT;
    table_exists BOOLEAN;
BEGIN
    -- Validate agent name
    IF NOT agent_name ~ '^[a-zA-Z0-9_]+$' THEN
        RAISE EXCEPTION 'Invalid agent name: %. Must contain only alphanumeric characters and underscores.', agent_name;
    END IF;
    
    -- Generate table names
    documents_table := 'agent_' || agent_name || '_documents';
    chunks_table := 'agent_' || agent_name || '_chunks';
    
    -- Check if tables already exist
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = documents_table
    ) INTO table_exists;
    
    IF table_exists THEN
        RAISE NOTICE 'Tables for agent % already exist', agent_name;
        RETURN FALSE;
    END IF;
    
    -- Create documents table
    EXECUTE format('
        CREATE TABLE %I (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT ''{}''::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )', documents_table);
    
    -- Create chunks table
    EXECUTE format('
        CREATE TABLE %I (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id UUID NOT NULL REFERENCES %I(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector(1536),
            chunk_index INTEGER NOT NULL,
            metadata JSONB DEFAULT ''{}''::jsonb,
            token_count INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )', chunks_table, documents_table);
    
    -- Create indexes for documents table
    EXECUTE format('CREATE INDEX %I ON %I USING GIN (metadata)', 
                   'idx_' || documents_table || '_metadata', documents_table);
    EXECUTE format('CREATE INDEX %I ON %I (created_at DESC)', 
                   'idx_' || documents_table || '_created_at', documents_table);
    
    -- Create indexes for chunks table
    EXECUTE format('CREATE INDEX %I ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1)', 
                   'idx_' || chunks_table || '_embedding', chunks_table);
    EXECUTE format('CREATE INDEX %I ON %I (document_id)', 
                   'idx_' || chunks_table || '_document_id', chunks_table);
    EXECUTE format('CREATE INDEX %I ON %I (document_id, chunk_index)', 
                   'idx_' || chunks_table || '_chunk_index', chunks_table);
    EXECUTE format('CREATE INDEX %I ON %I USING GIN (content gin_trgm_ops)', 
                   'idx_' || chunks_table || '_content_trgm', chunks_table);
    
    -- Create trigger for updating updated_at column
    EXECUTE format('
        CREATE TRIGGER %I BEFORE UPDATE ON %I
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()', 
        'update_' || documents_table || '_updated_at', documents_table);
    
    RAISE NOTICE 'Created tables for agent: %', agent_name;
    RETURN TRUE;
END;
$$;

-- Function to create agent-specific search functions
CREATE OR REPLACE FUNCTION create_agent_search_functions(agent_name TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    documents_table TEXT;
    chunks_table TEXT;
    match_function TEXT;
    hybrid_function TEXT;
    get_chunks_function TEXT;
BEGIN
    -- Generate table and function names
    documents_table := 'agent_' || agent_name || '_documents';
    chunks_table := 'agent_' || agent_name || '_chunks';
    match_function := 'match_chunks_' || agent_name;
    hybrid_function := 'hybrid_search_' || agent_name;
    get_chunks_function := 'get_document_chunks_' || agent_name;
    
    -- Create match_chunks function for this agent
    EXECUTE format('
        CREATE OR REPLACE FUNCTION %I(
            query_embedding vector(1536),
            match_count INT DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id UUID,
            document_id UUID,
            content TEXT,
            similarity FLOAT,
            metadata JSONB,
            document_title TEXT,
            document_source TEXT
        )
        LANGUAGE plpgsql
        AS $func$
        BEGIN
            RETURN QUERY
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                c.content,
                1 - (c.embedding <=> query_embedding) AS similarity,
                c.metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM %I c
            JOIN %I d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $func$', match_function, chunks_table, documents_table);
    
    -- Create hybrid_search function for this agent
    EXECUTE format('
        CREATE OR REPLACE FUNCTION %I(
            query_embedding vector(1536),
            query_text TEXT,
            match_count INT DEFAULT 10,
            text_weight FLOAT DEFAULT 0.3
        )
        RETURNS TABLE (
            chunk_id UUID,
            document_id UUID,
            content TEXT,
            combined_score FLOAT,
            vector_similarity FLOAT,
            text_similarity FLOAT,
            metadata JSONB,
            document_title TEXT,
            document_source TEXT
        )
        LANGUAGE plpgsql
        AS $func$
        BEGIN
            RETURN QUERY
            WITH vector_results AS (
                SELECT 
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    1 - (c.embedding <=> query_embedding) AS vector_sim,
                    c.metadata,
                    d.title AS doc_title,
                    d.source AS doc_source
                FROM %I c
                JOIN %I d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            ),
            text_results AS (
                SELECT 
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    ts_rank_cd(to_tsvector(''english'', c.content), plainto_tsquery(''english'', query_text)) AS text_sim,
                    c.metadata,
                    d.title AS doc_title,
                    d.source AS doc_source
                FROM %I c
                JOIN %I d ON c.document_id = d.id
                WHERE to_tsvector(''english'', c.content) @@ plainto_tsquery(''english'', query_text)
            )
            SELECT 
                COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
                COALESCE(v.document_id, t.document_id) AS document_id,
                COALESCE(v.content, t.content) AS content,
                (COALESCE(v.vector_sim, 0) * (1 - text_weight) + COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
                COALESCE(v.vector_sim, 0) AS vector_similarity,
                COALESCE(t.text_sim, 0) AS text_similarity,
                COALESCE(v.metadata, t.metadata) AS metadata,
                COALESCE(v.doc_title, t.doc_title) AS document_title,
                COALESCE(v.doc_source, t.doc_source) AS document_source
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $func$', hybrid_function, chunks_table, documents_table, chunks_table, documents_table);
    
    -- Create get_document_chunks function for this agent
    EXECUTE format('
        CREATE OR REPLACE FUNCTION %I(doc_id UUID)
        RETURNS TABLE (
            chunk_id UUID,
            content TEXT,
            chunk_index INTEGER,
            metadata JSONB
        )
        LANGUAGE plpgsql
        AS $func$
        BEGIN
            RETURN QUERY
            SELECT 
                id AS chunk_id,
                %I.content,
                %I.chunk_index,
                %I.metadata
            FROM %I
            WHERE document_id = doc_id
            ORDER BY chunk_index;
        END;
        $func$', get_chunks_function, chunks_table, chunks_table, chunks_table, chunks_table);
    
    RAISE NOTICE 'Created search functions for agent: %', agent_name;
    RETURN TRUE;
END;
$$;

-- Function to register a new agent
CREATE OR REPLACE FUNCTION register_agent(
    p_agent_name TEXT,
    p_ingestion_folder TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    table_prefix TEXT;
BEGIN
    -- Validate agent name
    IF NOT p_agent_name ~ '^[a-zA-Z0-9_]+$' THEN
        RAISE EXCEPTION 'Invalid agent name: %. Must contain only alphanumeric characters and underscores.', p_agent_name;
    END IF;
    
    -- Generate table prefix
    table_prefix := 'agent_' || p_agent_name || '_';
    
    -- Insert into agent registry
    INSERT INTO agent_registry (agent_name, table_prefix, ingestion_folder, metadata)
    VALUES (p_agent_name, table_prefix, p_ingestion_folder, p_metadata)
    ON CONFLICT (agent_name) DO UPDATE SET
        last_updated = CURRENT_TIMESTAMP,
        ingestion_folder = COALESCE(EXCLUDED.ingestion_folder, agent_registry.ingestion_folder),
        metadata = EXCLUDED.metadata;
    
    -- Create agent tables
    PERFORM create_agent_tables(p_agent_name);
    
    -- Create agent search functions
    PERFORM create_agent_search_functions(p_agent_name);
    
    RETURN TRUE;
END;
$$;

-- Function to drop agent tables and functions
CREATE OR REPLACE FUNCTION drop_agent(agent_name TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    documents_table TEXT;
    chunks_table TEXT;
    match_function TEXT;
    hybrid_function TEXT;
    get_chunks_function TEXT;
BEGIN
    -- Generate names
    documents_table := 'agent_' || agent_name || '_documents';
    chunks_table := 'agent_' || agent_name || '_chunks';
    match_function := 'match_chunks_' || agent_name;
    hybrid_function := 'hybrid_search_' || agent_name;
    get_chunks_function := 'get_document_chunks_' || agent_name;
    
    -- Drop functions
    EXECUTE format('DROP FUNCTION IF EXISTS %I(vector, INT)', match_function);
    EXECUTE format('DROP FUNCTION IF EXISTS %I(vector, TEXT, INT, FLOAT)', hybrid_function);
    EXECUTE format('DROP FUNCTION IF EXISTS %I(UUID)', get_chunks_function);
    
    -- Drop tables (chunks first due to foreign key)
    EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', chunks_table);
    EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', documents_table);
    
    -- Remove from registry
    DELETE FROM agent_registry WHERE agent_name = drop_agent.agent_name;
    
    RAISE NOTICE 'Dropped agent: %', agent_name;
    RETURN TRUE;
END;
$$;

-- Function to list all agents
CREATE OR REPLACE FUNCTION list_agents()
RETURNS TABLE (
    agent_name TEXT,
    table_prefix TEXT,
    ingestion_folder TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE,
    document_count BIGINT,
    chunk_count BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ar.agent_name,
        ar.table_prefix,
        ar.ingestion_folder,
        ar.created_at,
        ar.last_updated,
        COALESCE(doc_counts.doc_count, 0) as document_count,
        COALESCE(chunk_counts.chunk_count, 0) as chunk_count
    FROM agent_registry ar
    LEFT JOIN LATERAL (
        SELECT COUNT(*) as doc_count
        FROM information_schema.tables t
        WHERE t.table_name = 'agent_' || ar.agent_name || '_documents'
        AND EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = t.table_name AND column_name = 'id'
        )
    ) doc_counts ON true
    LEFT JOIN LATERAL (
        SELECT COUNT(*) as chunk_count
        FROM information_schema.tables t
        WHERE t.table_name = 'agent_' || ar.agent_name || '_chunks'
        AND EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = t.table_name AND column_name = 'id'
        )
    ) chunk_counts ON true
    ORDER BY ar.created_at DESC;
END;
$$;