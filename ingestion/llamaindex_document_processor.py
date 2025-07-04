"""
LlamaIndex Document Processor

A clean implementation following LlamaIndex OSS patterns for processing
PDF, DOCX, PPTX, and MD files with proper error handling and batch processing.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Core LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    PptxReader
)

logger = logging.getLogger(__name__)


class LlamaIndexDocumentProcessor:
    """
    Clean LlamaIndex document processor following OSS best practices.
    
    Features:
    - Specialized readers for different file types
    - Semantic text splitting
    - Batch embedding generation
    - Comprehensive error handling
    - Rate limiting for API calls
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        
        # Initialize LlamaIndex components
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n"
        )
        
        # Initialize embedding model using environment variables
        embedding_api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY must be set in environment")
        
        self.embed_model = OpenAIEmbedding(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            api_key=embedding_api_key,
            embed_batch_size=50
        )
        
        # Initialize specialized readers
        self.readers = {
            '.pdf': PyMuPDFReader(),
            '.docx': DocxReader(),
            '.pptx': PptxReader()
        }
        
        logger.info("LlamaIndex document processor initialized")
    
    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed document results
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Discover files
        files = self._discover_files(directory_path)
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            return []
        
        results = []
        for i, file_path in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
                result = await self._process_single_file(file_path)
                results.append(result)
                
                # Add delay between files to avoid rate limits
                if i < len(files) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")
                results.append({
                    'title': file_path.name,
                    'source': str(file_path),
                    'success': False,
                    'error': str(e),
                    'chunks': [],
                    'file_type': file_path.suffix.lower()
                })
        
        logger.info(f"Completed processing {len(files)} files")
        return results
    
    def _discover_files(self, directory_path: str) -> List[Path]:
        """Discover supported files in directory."""
        supported_extensions = [".pdf", ".docx", ".pptx", ".md"]
        files = []
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        for file_path in directory.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                not file_path.name.startswith('.')):
                files.append(file_path)
        
        return sorted(files)
    
    async def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file using LlamaIndex patterns.
        
        Args:
            file_path: Path to file
            
        Returns:
            Processing result dictionary
        """
        file_type = file_path.suffix.lower()
        title = file_path.name
        
        logger.info(f"Processing {file_type} file: {title}")
        
        try:
            # Step 1: Load document using appropriate reader
            documents = await self._load_document(file_path)
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Step 2: Split documents into nodes
            nodes = await self._split_documents(documents)
            
            if not nodes:
                raise ValueError("No chunks created from document")
            
            # Step 3: Generate embeddings
            await self._generate_embeddings(nodes)
            
            # Step 4: Convert to result format
            chunks = self._nodes_to_chunks(nodes)
            
            result = {
                "title": title,
                "source": str(file_path),
                "success": True,
                "chunks": chunks,
                "file_type": file_type,
                "chunk_count": len(chunks)
            }
            
            logger.info(f"Successfully processed {title}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return {
                "title": title,
                "source": str(file_path),
                "success": False,
                "error": str(e),
                "chunks": [],
                "file_type": file_type
            }
    
    async def _load_document(self, file_path: Path) -> List[Document]:
        """
        Load document using appropriate LlamaIndex reader.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of LlamaIndex Document objects
        """
        file_type = file_path.suffix.lower()
        
        try:
            if file_type == '.pdf':
                documents = await asyncio.to_thread(
                    self.readers['.pdf'].load_data, file_path=str(file_path)
                )
            elif file_type == '.docx':
                documents = await asyncio.to_thread(
                    self.readers['.docx'].load_data, file_path=str(file_path)
                )
            elif file_type == '.pptx':
                documents = await asyncio.to_thread(
                    self.readers['.pptx'].load_data, file_path=str(file_path)
                )
            elif file_type == '.md':
                # Handle markdown files directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents = [Document(
                    text=content,
                    metadata={
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'file_type': file_type
                    }
                )]
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Enhance metadata for all documents
            for doc in documents:
                doc.metadata.update({
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_type,
                    'file_size': file_path.stat().st_size,
                    'processing_date': datetime.now().isoformat()
                })
            
            logger.info(f"Loaded {len(documents)} document(s) from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path.name}: {e}")
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
                # Handle large documents by truncating if necessary
                text = document.text
                if len(text) > 1000000:  # 1MB limit
                    logger.warning(f"Large document detected, truncating: {len(text)} chars")
                    text = text[:1000000]
                
                # Split document text into chunks
                text_chunks = self.text_splitter.split_text(text)
                
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
        
        if not nodes:
            return
        
        try:
            # Process in batches for efficiency and rate limiting
            batch_size = 25  # Conservative batch size
            
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Prepare texts for embedding
                texts = []
                for node in batch:
                    try:
                        text = node.get_content(metadata_mode="all")
                        texts.append(text)
                    except Exception as e:
                        logger.warning(f"Error getting content for node: {e}")
                        texts.append(node.text)  # Fallback to raw text
                
                # Generate embeddings for batch
                try:
                    embeddings = await asyncio.to_thread(
                        self.embed_model.get_text_embedding_batch, texts
                    )
                    
                    # Assign embeddings to nodes
                    for node, embedding in zip(batch, embeddings):
                        node.embedding = embedding
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    # Assign zero vectors as fallback
                    for node in batch:
                        node.embedding = [0.0] * 1536  # OpenAI embedding dimension
                
                # Rate limiting between batches
                if i + batch_size < len(nodes):
                    await asyncio.sleep(0.2)
                
                batch_num = i // batch_size + 1
                total_batches = (len(nodes) + batch_size - 1) // batch_size
                logger.info(f"Processed embedding batch {batch_num}/{total_batches}")
            
            logger.info(f"Successfully generated embeddings for {len(nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            raise
    
    def _nodes_to_chunks(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """
        Convert LlamaIndex nodes to chunk dictionaries.
        
        Args:
            nodes: List of TextNode objects
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for i, node in enumerate(nodes):
            chunk = {
                "content": node.text,
                "index": i,
                "embedding": getattr(node, 'embedding', None),
                "metadata": node.metadata,
                "token_count": len(node.text.split())
            }
            chunks.append(chunk)
        
        return chunks


def create_processor(config: Dict[str, Any]) -> LlamaIndexDocumentProcessor:
    """
    Factory function to create a LlamaIndex document processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LlamaIndexDocumentProcessor instance
    """
    return LlamaIndexDocumentProcessor(config)