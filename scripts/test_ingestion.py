#!/usr/bin/env python3
"""
Test script for the new LlamaIndex-based ingestion system.

This script tests the rewritten ingestion pipeline with sample documents.
"""

import os
import sys
import asyncio
import logging
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the new ingestion system
from ingestion.ingest import LlamaIndexIngestionPipeline
from agent.models import IngestionConfig, IngestionMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_test_documents(test_dir: Path) -> None:
    """Create sample test documents for testing."""
    
    # Create a sample markdown file
    md_content = """# Test Document

This is a test markdown document for verifying the LlamaIndex ingestion pipeline.

## Section 1

This section contains some sample content about artificial intelligence and machine learning.
The content is designed to test the chunking and embedding generation process.

## Section 2

This is another section with different content about data science and analytics.
It should be split into separate chunks during processing.

### Subsection 2.1

More detailed content here with technical information about:
- Vector databases
- Embedding models
- Semantic search
- Knowledge graphs

## Conclusion

This document serves as a test case for the ingestion pipeline.
"""
    
    md_file = test_dir / "test_document.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Created test markdown file: {md_file}")


async def test_ingestion_pipeline():
    """Test the LlamaIndex ingestion pipeline."""
    
    # Create temporary directory for test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        logger.info(f"Created temporary test directory: {test_dir}")
        
        # Create test documents
        create_test_documents(test_dir)
        
        # Create ingestion configuration
        config = IngestionConfig(
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=100,
            use_semantic_chunking=True,
            extract_entities=True,
            ingestion_mode=IngestionMode.VECTOR_ONLY  # Test vector only for now
        )
        
        # Create pipeline
        pipeline = LlamaIndexIngestionPipeline(
            config=config,
            local_folder=str(test_dir),
            file_types=['.md'],  # Test with markdown only
            clean_before_ingest=False
        )
        
        try:
            logger.info("Initializing ingestion pipeline...")
            await pipeline.initialize()
            
            logger.info("Starting document ingestion...")
            
            def progress_callback(current: int, total: int):
                logger.info(f"Progress: {current}/{total} documents processed")
            
            results = await pipeline.ingest_documents(progress_callback)
            
            # Print results
            logger.info("Ingestion completed!")
            logger.info(f"Processed {len(results)} documents")
            
            for result in results:
                if result.errors:
                    logger.error(f"Document {result.title} had errors: {result.errors}")
                else:
                    logger.info(f"✓ {result.title}: {result.chunks_created} chunks created")
                    logger.info(f"  Processing time: {result.processing_time_ms:.1f}ms")
            
            return len(results) > 0 and all(not r.errors for r in results)
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False
        finally:
            await pipeline.close()


async def main():
    """Main test function."""
    logger.info("Starting LlamaIndex ingestion pipeline test...")
    
    try:
        # Check environment variables
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable not set")
            logger.info("Please set your OpenAI API key to run this test")
            return
        
        if not os.getenv('DATABASE_URL'):
            logger.warning("DATABASE_URL not set - database operations will fail")
            logger.info("This test will still validate document processing logic")
        
        # Run the test
        success = await test_ingestion_pipeline()
        
        if success:
            logger.info("✓ Test completed successfully!")
        else:
            logger.error("✗ Test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())