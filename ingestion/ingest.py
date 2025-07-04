"""
Document ingestion pipeline using LlamaIndex OSS patterns.

This module provides a complete rewrite of the document ingestion system
following LlamaIndex best practices for handling PDF, DOCX, PPTX, and MD files.
"""

import os
import asyncio
import logging
import json
import argparse
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime

# Core LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, BaseNode, MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    PptxReader
)

# Database imports
from dotenv import load_dotenv

# Local imports
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult, IngestionMode
except ImportError:
    # For direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult, IngestionMode

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LlamaIndexIngestionPipeline:
    """
    LlamaIndex-based document ingestion pipeline.
    
    Follows OSS patterns from LlamaIndex documentation for:
    - Document loading with specialized readers
    - Text splitting with semantic chunking
    - Embedding generation with batching
    - Vector store integration
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        local_folder: str,
        file_types: Optional[List[str]] = None,
        clean_before_ingest: bool = False
    ):
        """
        Initialize the LlamaIndex ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            local_folder: Path to folder containing documents
            file_types: List of file extensions to process
            clean_before_ingest: Whether to clean existing data
        """
        self.config = config
        self.local_folder = Path(local_folder)
        self.file_types = file_types or ['.pdf', '.docx', '.pptx', '.md']
        self.clean_before_ingest = clean_before_ingest
        
        # Initialize LlamaIndex components
        self.text_splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n\n"
        )
        
        # Initialize embedding model using environment variables
        embedding_api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY must be set in environment")
        
        self.embed_model = OpenAIEmbedding(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            api_key=embedding_api_key,
            embed_batch_size=100  # Batch embeddings for efficiency
        )
        
        # Initialize readers
        self.readers = {
            '.pdf': PyMuPDFReader(),
            '.docx': DocxReader(),
            '.pptx': PptxReader(),
            '.md': None  # Will use simple text reading
        }
        
        self._initialized = False
        logger.info(f"LlamaIndex pipeline initialized for: {self.local_folder}")
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing LlamaIndex ingestion pipeline...")
        
        # Initialize database connections
        await initialize_database()
        
        # Initialize graph if needed
        if self.config.ingestion_mode in [IngestionMode.GRAPH_ONLY, IngestionMode.BOTH]:
            await initialize_graph()
        
        # Clean databases if requested
        if self.clean_before_ingest:
            await self._clean_databases()
        
        self._initialized = True
        logger.info("LlamaIndex pipeline initialized successfully")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await close_graph()
            await close_database()
            self._initialized = False
    
    async def ingest_documents(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the local folder.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()
        
        # Discover documents
        document_paths = self._discover_documents()
        
        if not document_paths:
            logger.warning(f"No documents found in {self.local_folder}")
            return []
        
        logger.info(f"Found {len(document_paths)} documents to process")
        
        results = []
        for i, doc_path in enumerate(document_paths):
            try:
                logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path.name}")
                
                result = await self._ingest_single_document(doc_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(document_paths))
                
                # Add small delay between documents
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process {doc_path.name}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=doc_path.name,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))
        
        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors")
        
        return results
    
    def _discover_documents(self) -> List[Path]:
        """Discover supported documents in the local folder."""
        if not self.local_folder.exists():
            logger.error(f"Local folder not found: {self.local_folder}")
            return []
        
        documents = []
        for file_type in self.file_types:
            pattern = f"**/*{file_type}"
            documents.extend(self.local_folder.glob(pattern))
        
        # Filter out hidden files and sort
        documents = [d for d in documents if not d.name.startswith('.')]
        return sorted(documents)
    
    async def _ingest_single_document(self, doc_path: Path) -> IngestionResult:
        """
        Ingest a single document using LlamaIndex patterns.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Load document using appropriate reader
            documents = await self._load_document(doc_path)
            
            if not documents:
                raise Exception("No content extracted from document")
            
            # Step 2: Split documents into chunks
            nodes = await self._split_documents(documents)
            
            if not nodes:
                raise Exception("No chunks created from document")
            
            # Step 3: Generate embeddings for nodes
            if self.config.ingestion_mode in [IngestionMode.VECTOR_ONLY, IngestionMode.BOTH]:
                await self._generate_embeddings(nodes)
            
            # Step 4: Save to PostgreSQL
            document_id = ""
            if self.config.ingestion_mode in [IngestionMode.VECTOR_ONLY, IngestionMode.BOTH]:
                document_id = await self._save_to_postgres(doc_path, documents[0], nodes)
            
            # Step 5: Build knowledge graph (if enabled)
            relationships_created = 0
            if self.config.ingestion_mode in [IngestionMode.GRAPH_ONLY, IngestionMode.BOTH]:
                relationships_created = await self._build_knowledge_graph(doc_path, nodes)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                document_id=document_id,
                title=doc_path.name,
                chunks_created=len(nodes),
                entities_extracted=0,  # TODO: Implement entity extraction
                relationships_created=relationships_created,
                processing_time_ms=processing_time,
                errors=[]
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error processing {doc_path.name}: {e}")
            
            return IngestionResult(
                document_id="",
                title=doc_path.name,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=processing_time,
                errors=[str(e)]
            )
    
    async def _load_document(self, doc_path: Path) -> List[Document]:
        """
        Load document using appropriate LlamaIndex reader.
        
        Args:
            doc_path: Path to document
            
        Returns:
            List of LlamaIndex Document objects
        """
        file_type = doc_path.suffix.lower()
        
        try:
            if file_type == '.pdf':
                reader = self.readers['.pdf']
                documents = reader.load_data(file_path=str(doc_path))
            elif file_type == '.docx':
                reader = self.readers['.docx']
                documents = reader.load_data(file_path=str(doc_path))
            elif file_type == '.pptx':
                reader = self.readers['.pptx']
                documents = reader.load_data(file_path=str(doc_path))
            elif file_type == '.md':
                # Simple text reading for markdown
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(
                    text=content,
                    metadata={
                        'file_path': str(doc_path),
                        'file_name': doc_path.name,
                        'file_type': file_type
                    }
                )]
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    'file_path': str(doc_path),
                    'file_name': doc_path.name,
                    'file_type': file_type,
                    'file_size': doc_path.stat().st_size,
                    'ingestion_date': datetime.now().isoformat()
                })
            
            logger.info(f"Loaded {len(documents)} document(s) from {doc_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {doc_path.name}: {e}")
            raise
    
    async def _split_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Split documents into chunks using LlamaIndex SentenceSplitter.
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of TextNode objects
        """
        nodes = []
        
        for doc_idx, document in enumerate(documents):
            try:
                # Split document text into chunks
                text_chunks = self.text_splitter.split_text(document.text)
                
                # Create TextNode for each chunk
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    if not chunk_text.strip():
                        continue
                    
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            'chunk_index': chunk_idx,
                            'doc_index': doc_idx,
                            'chunk_id': f"{doc_idx}_{chunk_idx}"
                        }
                    )
                    nodes.append(node)
                
                logger.info(f"Created {len(text_chunks)} chunks from document {doc_idx}")
                
            except Exception as e:
                logger.error(f"Error splitting document {doc_idx}: {e}")
                raise
        
        return nodes
    
    async def _generate_embeddings(self, nodes: List[BaseNode]) -> None:
        """
        Generate embeddings for nodes using batch processing.
        
        Args:
            nodes: List of TextNode objects to embed
        """
        logger.info(f"Generating embeddings for {len(nodes)} nodes")
        
        try:
            # Batch process embeddings for efficiency
            batch_size = 50
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Generate embeddings for batch
                texts = [node.get_content(metadata_mode=MetadataMode.ALL) for node in batch]
                embeddings = await asyncio.to_thread(
                    self.embed_model.get_text_embedding_batch, texts
                )
                
                # Assign embeddings to nodes
                for node, embedding in zip(batch, embeddings):
                    node.embedding = embedding
                
                # Rate limiting
                if i + batch_size < len(nodes):
                    await asyncio.sleep(0.1)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(nodes) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def _save_to_postgres(
        self, 
        doc_path: Path, 
        document: Document, 
        nodes: List[BaseNode]
    ) -> str:
        """
        Save document and chunks to PostgreSQL.
        
        Args:
            doc_path: Path to original document
            document: LlamaIndex Document object
            nodes: List of TextNode objects with embeddings
            
        Returns:
            Document ID
        """
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    doc_path.name,
                    str(doc_path),
                    document.text,
                    json.dumps(document.metadata)
                )
                
                document_id = document_result["id"]
                
                # Insert chunks
                for node in nodes:
                    # Convert embedding to PostgreSQL vector format
                    embedding_data = None
                    if hasattr(node, 'embedding') and node.embedding:
                        embedding_data = '[' + ','.join(map(str, node.embedding)) + ']'
                    
                    # Get text content from node
                    node_text = node.get_content() if hasattr(node, 'get_content') else str(node)
                    
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        node_text,
                        embedding_data,
                        node.metadata.get('chunk_index', 0),
                        json.dumps(node.metadata),
                        len(node_text.split())
                    )
                
                logger.info(f"Saved document {doc_path.name} with {len(nodes)} chunks to PostgreSQL")
                return document_id
    
    async def _build_knowledge_graph(self, doc_path: Path, nodes: List[BaseNode]) -> int:
        """
        Build knowledge graph from document nodes.
        
        Args:
            doc_path: Path to original document
            nodes: List of TextNode objects
            
        Returns:
            Number of relationships created
        """
        # TODO: Implement knowledge graph building
        # This would integrate with the existing graph_builder
        # When implemented, this will process the provided nodes for entity extraction
        logger.info(f"Knowledge graph building not yet implemented for {doc_path.name} ({len(nodes)} nodes)")
        return 0
    
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")


async def main():
    """Main function for running LlamaIndex-based ingestion."""
    parser = argparse.ArgumentParser(description="LlamaIndex-based document ingestion")
    
    # Required arguments
    parser.add_argument("--local-folder", required=True,
                       help="Path to folder containing documents")
    
    # Optional arguments
    parser.add_argument("--file-types", default="pdf,docx,pptx,md",
                       help="Comma-separated list of file extensions (default: pdf,docx,pptx,md)")
    parser.add_argument("--clean", "-c", action="store_true",
                       help="Clean existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size for text splitting (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Chunk overlap size (default: 200)")
    parser.add_argument("--ingestion-mode", choices=["vector_only", "graph_only", "both"],
                       default="both", help="Ingestion mode (default: both)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create ingestion configuration
    ingestion_mode_map = {
        "vector_only": IngestionMode.VECTOR_ONLY,
        "graph_only": IngestionMode.GRAPH_ONLY,
        "both": IngestionMode.BOTH
    }
    
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=True,
        extract_entities=True,
        ingestion_mode=ingestion_mode_map[args.ingestion_mode]
    )
    
    # Parse file types
    file_types = [f".{ext.strip()}" for ext in args.file_types.split(',')]
    
    # Create and initialize pipeline
    pipeline = LlamaIndexIngestionPipeline(
        config=config,
        local_folder=args.local_folder,
        file_types=file_types,
        clean_before_ingest=args.clean
    )
    
    await pipeline.initialize()
    
    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")
    
    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "="*60)
        print("LLAMAINDEX INGESTION SUMMARY")
        print("="*60)
        print(f"Local folder: {args.local_folder}")
        print(f"File types: {args.file_types}")
        print(f"Ingestion mode: {args.ingestion_mode}")
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Processing time: {total_time:.2f} seconds")
        print()
        
        # Print individual results
        for result in results:
            status = "✓" if not result.errors else "✗"
            print(f"{status} {result.title}: {result.chunks_created} chunks")
            
            for error in result.errors:
                print(f"  Error: {error}")
        
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())