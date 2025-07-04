"""
Dynamic schema management for multi-agent vector store support.

This module provides functions to manage agent-specific database tables
and search functions dynamically.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import asyncpg
from dotenv import load_dotenv

from .db_utils import db_pool

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AgentSchemaManager:
    """Manages database schemas for multiple agents."""
    
    def __init__(self):
        """Initialize the schema manager."""
        self.table_prefix = os.getenv('AGENT_TABLE_PREFIX', 'agent_')
        
    def validate_agent_name(self, agent_name: str) -> bool:
        """
        Validate agent name format.
        
        Args:
            agent_name: Agent name to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not agent_name:
            return False
        
        # Must be alphanumeric and underscores only, 1-50 characters
        pattern = r'^[a-zA-Z0-9_]{1,50}$'
        return bool(re.match(pattern, agent_name))
    
    def get_table_names(self, agent_name: str) -> Dict[str, str]:
        """
        Get table names for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dictionary with table names
        """
        if not self.validate_agent_name(agent_name):
            raise ValueError(f"Invalid agent name: {agent_name}")
        
        prefix = f"{self.table_prefix}{agent_name}_"
        return {
            'documents': f"{prefix}documents",
            'chunks': f"{prefix}chunks",
        }
    
    def get_function_names(self, agent_name: str) -> Dict[str, str]:
        """
        Get function names for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dictionary with function names
        """
        if not self.validate_agent_name(agent_name):
            raise ValueError(f"Invalid agent name: {agent_name}")
        
        return {
            'match_chunks': f"match_chunks_{agent_name}",
            'hybrid_search': f"hybrid_search_{agent_name}",
            'get_document_chunks': f"get_document_chunks_{agent_name}",
        }
    
    async def agent_exists(self, agent_name: str) -> bool:
        """
        Check if an agent is registered.
        
        Args:
            agent_name: Agent name to check
            
        Returns:
            True if agent exists, False otherwise
        """
        if not self.validate_agent_name(agent_name):
            return False
        
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM agent_registry WHERE agent_name = $1)",
                agent_name
            )
            return result
    
    async def create_agent_registry_table(self) -> None:
        """Create the agent registry table if it doesn't exist."""
        async with db_pool.acquire() as conn:
            # Read and execute the agent schema SQL
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'sql', 'agent_schema.sql')
            
            try:
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                # Execute the schema creation
                await conn.execute(schema_sql)
                logger.info("Agent registry and functions created successfully")
                
            except FileNotFoundError:
                # Fallback to inline creation
                await self._create_registry_inline(conn)
    
    async def _create_registry_inline(self, conn: asyncpg.Connection) -> None:
        """Create agent registry table inline (fallback)."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_name VARCHAR(50) PRIMARY KEY CHECK (agent_name ~ '^[a-zA-Z0-9_]+$'),
                table_prefix VARCHAR(100) NOT NULL,
                ingestion_folder TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}',
                CONSTRAINT unique_table_prefix UNIQUE (table_prefix)
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_registry_created_at 
            ON agent_registry (created_at DESC)
        """)
        
        logger.info("Agent registry table created (inline)")
    
    async def register_agent(
        self, 
        agent_name: str, 
        ingestion_folder: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register a new agent.
        
        Args:
            agent_name: Agent name
            ingestion_folder: Default ingestion folder for this agent
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_agent_name(agent_name):
            raise ValueError(f"Invalid agent name: {agent_name}")
        
        async with db_pool.acquire() as conn:
            try:
                # Use the database function to register the agent
                import json
                metadata_json = json.dumps(metadata or {})
                result = await conn.fetchval(
                    "SELECT register_agent($1, $2, $3::jsonb)",
                    agent_name,
                    ingestion_folder,
                    metadata_json
                )
                
                if result:
                    logger.info(f"Successfully registered agent: {agent_name}")
                    return True
                else:
                    logger.warning(f"Failed to register agent: {agent_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error registering agent {agent_name}: {e}")
                return False
    
    async def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister an agent and drop its tables.
        
        Args:
            agent_name: Agent name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.validate_agent_name(agent_name):
            raise ValueError(f"Invalid agent name: {agent_name}")
        
        async with db_pool.acquire() as conn:
            try:
                # Use the database function to drop the agent
                result = await conn.fetchval("SELECT drop_agent($1)", agent_name)
                
                if result:
                    logger.info(f"Successfully unregistered agent: {agent_name}")
                    return True
                else:
                    logger.warning(f"Failed to unregister agent: {agent_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error unregistering agent {agent_name}: {e}")
                return False
    
    async def list_agents(self) -> List[Dict]:
        """
        List all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        async with db_pool.acquire() as conn:
            try:
                rows = await conn.fetch("SELECT * FROM list_agents()")
                
                agents = []
                for row in rows:
                    agents.append({
                        'agent_name': row['agent_name'],
                        'table_prefix': row['table_prefix'],
                        'ingestion_folder': row['ingestion_folder'],
                        'created_at': row['created_at'],
                        'last_updated': row['last_updated'],
                        'document_count': row['document_count'],
                        'chunk_count': row['chunk_count']
                    })
                
                return agents
                
            except Exception as e:
                logger.error(f"Error listing agents: {e}")
                return []
    
    async def get_agent_info(self, agent_name: str) -> Optional[Dict]:
        """
        Get information about a specific agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Agent information dictionary or None if not found
        """
        if not self.validate_agent_name(agent_name):
            return None
        
        async with db_pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT agent_name, table_prefix, ingestion_folder, 
                           created_at, last_updated, metadata
                    FROM agent_registry 
                    WHERE agent_name = $1
                    """,
                    agent_name
                )
                
                if row:
                    return {
                        'agent_name': row['agent_name'],
                        'table_prefix': row['table_prefix'],
                        'ingestion_folder': row['ingestion_folder'],
                        'created_at': row['created_at'],
                        'last_updated': row['last_updated'],
                        'metadata': row['metadata']
                    }
                
                return None
                
            except Exception as e:
                logger.error(f"Error getting agent info for {agent_name}: {e}")
                return None
    
    async def ensure_agent_exists(
        self, 
        agent_name: str, 
        ingestion_folder: Optional[str] = None
    ) -> bool:
        """
        Ensure an agent exists, creating it if necessary.
        
        Args:
            agent_name: Agent name
            ingestion_folder: Default ingestion folder
            
        Returns:
            True if agent exists or was created successfully
        """
        if await self.agent_exists(agent_name):
            logger.debug(f"Agent {agent_name} already exists")
            return True
        
        logger.info(f"Creating new agent: {agent_name}")
        return await self.register_agent(agent_name, ingestion_folder)
    
    async def get_agent_table_counts(self, agent_name: str) -> Dict[str, int]:
        """
        Get row counts for an agent's tables.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dictionary with table names and counts
        """
        if not await self.agent_exists(agent_name):
            return {'documents': 0, 'chunks': 0}
        
        table_names = self.get_table_names(agent_name)
        counts = {}
        
        async with db_pool.acquire() as conn:
            try:
                for table_type, table_name in table_names.items():
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    counts[table_type] = count
                
                return counts
                
            except Exception as e:
                logger.error(f"Error getting table counts for {agent_name}: {e}")
                return {'documents': 0, 'chunks': 0}
    
    async def clean_agent_data(self, agent_name: str) -> bool:
        """
        Clean all data for an agent (but keep the agent registered).
        
        Args:
            agent_name: Agent name
            
        Returns:
            True if successful
        """
        if not await self.agent_exists(agent_name):
            logger.warning(f"Agent {agent_name} does not exist")
            return False
        
        table_names = self.get_table_names(agent_name)
        
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Delete chunks first (foreign key constraint)
                    await conn.execute(f"DELETE FROM {table_names['chunks']}")
                    await conn.execute(f"DELETE FROM {table_names['documents']}")
                    
                    logger.info(f"Cleaned data for agent: {agent_name}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error cleaning data for {agent_name}: {e}")
                    return False


# Global instance
schema_manager = AgentSchemaManager()


# Utility functions for common operations
async def ensure_agent_exists(agent_name: str, ingestion_folder: Optional[str] = None) -> bool:
    """Utility function to ensure an agent exists."""
    return await schema_manager.ensure_agent_exists(agent_name, ingestion_folder)


async def get_agent_table_names(agent_name: str) -> Dict[str, str]:
    """Utility function to get table names for an agent."""
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    return schema_manager.get_table_names(agent_name)


async def get_agent_function_names(agent_name: str) -> Dict[str, str]:
    """Utility function to get function names for an agent."""
    if not await schema_manager.agent_exists(agent_name):
        raise ValueError(f"Agent {agent_name} does not exist")
    
    return schema_manager.get_function_names(agent_name)