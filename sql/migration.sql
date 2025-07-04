-- Migration script to convert existing single-agent system to multi-agent system
-- This script will:
-- 1. Create agent registry table and functions
-- 2. Register the default 'main_agent' 
-- 3. Create agent-specific tables for main_agent
-- 4. Migrate existing data to main_agent tables
-- 5. Optionally drop old tables (commented out for safety)

BEGIN;

-- Step 1: Run the agent schema setup
\i agent_schema.sql

-- Step 2: Register the main agent
SELECT register_agent('main_agent', 'documents/', '{"description": "Default main agent migrated from single-agent system"}'::jsonb);

-- Step 3: Migrate existing data to main_agent tables
-- First, check if old tables exist and have data
DO $$
DECLARE
    old_doc_count INTEGER;
    old_chunk_count INTEGER;
    new_doc_count INTEGER;
    new_chunk_count INTEGER;
BEGIN
    -- Check if old tables exist
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents') THEN
        SELECT COUNT(*) INTO old_doc_count FROM documents;
        RAISE NOTICE 'Found % documents in old documents table', old_doc_count;
        
        IF old_doc_count > 0 THEN
            -- Migrate documents
            INSERT INTO agent_main_agent_documents (id, title, source, content, metadata, created_at, updated_at)
            SELECT id, title, source, content, metadata, created_at, updated_at
            FROM documents
            ON CONFLICT (id) DO NOTHING;
            
            GET DIAGNOSTICS new_doc_count = ROW_COUNT;
            RAISE NOTICE 'Migrated % documents to agent_main_agent_documents', new_doc_count;
        END IF;
    ELSE
        RAISE NOTICE 'No old documents table found - clean installation';
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chunks') THEN
        SELECT COUNT(*) INTO old_chunk_count FROM chunks;
        RAISE NOTICE 'Found % chunks in old chunks table', old_chunk_count;
        
        IF old_chunk_count > 0 THEN
            -- Migrate chunks
            INSERT INTO agent_main_agent_chunks (id, document_id, content, embedding, chunk_index, metadata, token_count, created_at)
            SELECT id, document_id, content, embedding, chunk_index, metadata, token_count, created_at
            FROM chunks
            ON CONFLICT (id) DO NOTHING;
            
            GET DIAGNOSTICS new_chunk_count = ROW_COUNT;
            RAISE NOTICE 'Migrated % chunks to agent_main_agent_chunks', new_chunk_count;
        END IF;
    ELSE
        RAISE NOTICE 'No old chunks table found - clean installation';
    END IF;
END $$;

-- Step 4: Create a view for backward compatibility (optional)
CREATE OR REPLACE VIEW documents AS
SELECT id, title, source, content, metadata, created_at, updated_at
FROM agent_main_agent_documents;

CREATE OR REPLACE VIEW chunks AS
SELECT id, document_id, content, embedding, chunk_index, metadata, token_count, created_at
FROM agent_main_agent_chunks;

-- Step 5: Update the original search functions to use main_agent by default
CREATE OR REPLACE FUNCTION match_chunks(
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
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM match_chunks_main_agent(query_embedding, match_count);
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
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
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM hybrid_search_main_agent(query_embedding, query_text, match_count, text_weight);
END;
$$;

CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM get_document_chunks_main_agent(doc_id);
END;
$$;

-- Step 6: Create the document_summaries view for the main agent
CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    AVG(c.token_count) AS avg_tokens_per_chunk,
    SUM(c.token_count) AS total_tokens
FROM agent_main_agent_documents d
LEFT JOIN agent_main_agent_chunks c ON d.id = c.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata;

COMMIT;

-- Instructions for completing the migration:
-- 
-- IMPORTANT: The original tables are preserved for safety!
-- 
-- After verifying the migration worked correctly, you can optionally drop the old tables:
-- 
-- BEGIN;
-- DROP VIEW IF EXISTS documents CASCADE;
-- DROP VIEW IF EXISTS chunks CASCADE;
-- DROP TABLE IF EXISTS chunks CASCADE;  -- Drop first due to foreign key
-- DROP TABLE IF EXISTS documents CASCADE;
-- DROP TABLE IF EXISTS messages CASCADE;
-- DROP TABLE IF EXISTS sessions CASCADE;
-- COMMIT;
--
-- Note: sessions and messages tables are not part of the agent system,
-- so they can remain as global tables or be migrated separately if needed.

-- Verification queries:
-- 
-- -- List all agents
-- SELECT * FROM list_agents();
-- 
-- -- Check main agent data
-- SELECT COUNT(*) as doc_count FROM agent_main_agent_documents;
-- SELECT COUNT(*) as chunk_count FROM agent_main_agent_chunks;
-- 
-- -- Test search functions
-- SELECT * FROM match_chunks_main_agent('[0,0,0,...]'::vector, 5);
-- SELECT * FROM hybrid_search_main_agent('[0,0,0,...]'::vector, 'test query', 5, 0.3);