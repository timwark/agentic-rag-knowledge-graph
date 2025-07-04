"""
Database utilities for PostgreSQL connection and operations.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from uuid import UUID
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """Create connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection


# Global database pool instance
db_pool = DatabasePool()


async def initialize_database():
    """Initialize database connection pool."""
    await db_pool.initialize()


async def close_database():
    """Close database connection pool."""
    await db_pool.close()


# Session Management Functions
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
) -> str:
    """
    Create a new session.
    
    Args:
        user_id: Optional user identifier
        metadata: Optional session metadata
        timeout_minutes: Session timeout in minutes
    
    Returns:
        Session ID
    """
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(metadata or {}),
            expires_at
        )
        
        return result["id"]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get session by ID.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Session data or None if not found/expired
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id,
                metadata,
                created_at,
                updated_at,
                expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        
        return None


async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Update session metadata.
    
    Args:
        session_id: Session UUID
        metadata: New metadata to merge
    
    Returns:
        True if updated, False if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = metadata || $2::jsonb
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
            json.dumps(metadata)
        )
        
        return result.split()[-1] != "0"


# Message Management Functions
async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a message to a session.
    
    Args:
        session_id: Session UUID
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Optional message metadata
    
    Returns:
        Message ID
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get messages for a session.
    
    Args:
        session_id: Session UUID
        limit: Maximum number of messages to return
    
    Returns:
        List of messages ordered by creation time
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                id::text,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Document Management Functions
async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a document by ID.
    
    Args:
        document_id: Document UUID
    
    Returns:
        Document data or None if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    List documents with optional filtering.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        metadata_filter: Optional metadata filter
    
    Returns:
        List of documents
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        conditions = []
        
        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]


# Vector Search Functions
async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search.
    
    Args:
        embedding: Query embedding vector
        limit: Maximum number of results
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str,
            limit
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search (vector + keyword).
    
    Args:
        embedding: Query embedding vector
        query_text: Query text for keyword search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
    
    Returns:
        List of matching chunks ordered by combined score (best first)
    """
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


# Chunk Management Functions
async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document.
    
    Args:
        document_id: Document UUID
    
    Returns:
        List of chunks ordered by chunk index
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"])
            }
            for row in results
        ]


# Utility Functions
async def execute_query(query: str, *params) -> List[Dict[str, Any]]:
    """
    Execute a custom query.
    
    Args:
        query: SQL query
        *params: Query parameters
    
    Returns:
        Query results
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]


async def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# Multi-Agent Support Functions
async def ensure_agent_schema() -> None:
    """Ensure agent registry and schema functions exist."""
    try:
        from .schema_manager import schema_manager
        await schema_manager.create_agent_registry_table()
    except Exception as e:
        logger.error(f"Failed to ensure agent schema: {e}")
        raise


async def vector_search_agent(
    agent_name: str,
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform vector search for a specific agent.
    
    Args:
        agent_name: Agent name
        embedding: Query embedding vector
        limit: Maximum number of results
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    from .schema_manager import schema_manager
    
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    function_names = schema_manager.get_function_names(agent_name)
    
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            f"SELECT * FROM {function_names['match_chunks']}($1::vector, $2)",
            embedding_str,
            limit
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "score": row["similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search_agent(
    agent_name: str,
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search for a specific agent.
    
    Args:
        agent_name: Agent name
        embedding: Query embedding vector
        query_text: Query text for keyword search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
    
    Returns:
        List of matching chunks ordered by combined score (best first)
    """
    from .schema_manager import schema_manager
    
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    function_names = schema_manager.get_function_names(agent_name)
    
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            f"SELECT * FROM {function_names['hybrid_search']}($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def get_document_chunks_agent(
    agent_name: str,
    document_id: str
) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document for a specific agent.
    
    Args:
        agent_name: Agent name
        document_id: Document UUID
    
    Returns:
        List of chunks ordered by chunk index
    """
    from .schema_manager import schema_manager
    
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    function_names = schema_manager.get_function_names(agent_name)
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            f"SELECT * FROM {function_names['get_document_chunks']}($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"])
            }
            for row in results
        ]


async def save_document_agent(
    agent_name: str,
    title: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a document for a specific agent.
    
    Args:
        agent_name: Agent name
        title: Document title
        source: Document source path
        content: Document content
        metadata: Optional metadata
    
    Returns:
        Document ID
    """
    from .schema_manager import schema_manager
    
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    table_names = schema_manager.get_table_names(agent_name)
    
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            f"""
            INSERT INTO {table_names['documents']} (title, source, content, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id::text
            """,
            title,
            source,
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]


async def save_chunk_agent(
    agent_name: str,
    document_id: str,
    content: str,
    embedding: Optional[List[float]] = None,
    chunk_index: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    token_count: Optional[int] = None
) -> str:
    """
    Save a chunk for a specific agent.
    
    Args:
        agent_name: Agent name
        document_id: Document ID
        content: Chunk content
        embedding: Optional embedding vector
        chunk_index: Chunk index in document
        metadata: Optional metadata
        token_count: Optional token count
    
    Returns:
        Chunk ID
    """
    from .schema_manager import schema_manager
    
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    table_names = schema_manager.get_table_names(agent_name)
    
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector format if provided
        embedding_str = None
        if embedding:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        result = await conn.fetchrow(
            f"""
            INSERT INTO {table_names['chunks']} 
            (document_id, content, embedding, chunk_index, metadata, token_count)
            VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
            RETURNING id::text
            """,
            document_id,
            content,
            embedding_str,
            chunk_index,
            json.dumps(metadata or {}),
            token_count
        )
        
        return result["id"]


async def clean_agent_data(agent_name: str) -> bool:
    """
    Clean all data for a specific agent.
    
    Args:
        agent_name: Agent name
    
    Returns:
        True if successful
    """
    from .schema_manager import schema_manager
    return await schema_manager.clean_agent_data(agent_name)


async def get_agent_stats(agent_name: str) -> Dict[str, int]:
    """
    Get statistics for a specific agent.
    
    Args:
        agent_name: Agent name
    
    Returns:
        Dictionary with statistics
    """
    from .schema_manager import schema_manager
    return await schema_manager.get_agent_table_counts(agent_name)


async def list_documents_agent(
    agent_name: str,
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    List documents for a specific agent with optional filtering.
    
    Args:
        agent_name: Agent name
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        metadata_filter: Optional metadata filter
    
    Returns:
        List of documents
    """
    from .schema_manager import schema_manager
    
    # Validate agent exists
    if not await schema_manager.agent_exists(agent_name):
        logger.warning(f"Agent {agent_name} does not exist")
        return []
    
    # Get agent-specific table names
    table_names = schema_manager.get_table_names(agent_name)
    
    async with db_pool.acquire() as conn:
        query = f"""
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM {table_names['documents']} d
            LEFT JOIN {table_names['chunks']} c ON d.id = c.document_id
        """
        
        params = []
        conditions = []
        
        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f"""
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """
        
        params.extend([limit, offset])
        
        try:
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "source": row["source"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "chunk_count": row["chunk_count"]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error listing documents for agent {agent_name}: {e}")
            return []