#!/usr/bin/env python3
"""
Convert Markdown files to PDF.

This script converts all .md files in the big_tech_docs directory to PDF format
and saves them in the tests/docs directory for testing purposes.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List

# Try to import required libraries
try:
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install required packages:")
    print("pip install markdown weasyprint")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarkdownToPDFConverter:
    """Convert Markdown files to PDF using markdown and weasyprint."""
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            source_dir: Directory containing .md files
            output_dir: Directory to save .pdf files
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure markdown extensions
        self.markdown_extensions = [
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code'
        ]
        
        logger.info(f"Converter initialized: {source_dir} -> {output_dir}")
    
    def get_css_styles(self) -> str:
        """Get CSS styles for PDF formatting."""
        return """
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 24pt;
            margin-top: 0;
            margin-bottom: 20pt;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10pt;
        }
        
        h2 {
            color: #34495e;
            font-size: 18pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5pt;
        }
        
        h3 {
            color: #34495e;
            font-size: 14pt;
            margin-top: 18pt;
            margin-bottom: 10pt;
        }
        
        h4, h5, h6 {
            color: #34495e;
            font-size: 12pt;
            margin-top: 15pt;
            margin-bottom: 8pt;
        }
        
        p {
            margin-bottom: 12pt;
            text-align: justify;
        }
        
        ul, ol {
            margin-bottom: 12pt;
            padding-left: 20pt;
        }
        
        li {
            margin-bottom: 6pt;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2pt 4pt;
            border-radius: 3pt;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 12pt;
            border-radius: 6pt;
            border-left: 4px solid #3498db;
            overflow-x: auto;
            margin-bottom: 16pt;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
        }
        
        blockquote {
            margin: 16pt 0;
            padding-left: 16pt;
            border-left: 4px solid #3498db;
            color: #555;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8pt 12pt;
            text-align: left;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        strong {
            color: #2c3e50;
        }
        
        em {
            color: #34495e;
        }
        
        .page-break {
            page-break-before: always;
        }
        """
    
    def convert_markdown_to_html(self, md_content: str) -> str:
        """
        Convert markdown content to HTML.
        
        Args:
            md_content: Markdown content as string
            
        Returns:
            HTML content as string
        """
        # Initialize markdown converter
        md = markdown.Markdown(extensions=self.markdown_extensions)
        
        # Convert to HTML
        html_body = md.convert(md_content)
        
        # Create complete HTML document
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Document</title>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        
        return html_template
    
    def convert_file(self, md_file: Path) -> bool:
        """
        Convert a single markdown file to PDF.
        
        Args:
            md_file: Path to markdown file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read markdown content
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            html_content = self.convert_markdown_to_html(md_content)
            
            # Create PDF filename
            pdf_filename = md_file.stem + '.pdf'
            pdf_path = self.output_dir / pdf_filename
            
            # Convert HTML to PDF
            html_doc = HTML(string=html_content)
            css = CSS(string=self.get_css_styles())
            
            html_doc.write_pdf(
                target=str(pdf_path),
                stylesheets=[css],
                font_config=FontConfiguration()
            )
            
            logger.info(f"✓ Converted: {md_file.name} -> {pdf_filename}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to convert {md_file.name}: {e}")
            return False
    
    def convert_all(self) -> dict:
        """
        Convert all markdown files in the source directory.
        
        Returns:
            Dictionary with conversion statistics
        """
        # Find all markdown files
        md_files = list(self.source_dir.glob("*.md"))
        
        if not md_files:
            logger.warning(f"No .md files found in {self.source_dir}")
            return {"total": 0, "successful": 0, "failed": 0}
        
        logger.info(f"Found {len(md_files)} markdown files to convert")
        
        # Convert each file
        successful = 0
        failed = 0
        
        for md_file in sorted(md_files):
            if self.convert_file(md_file):
                successful += 1
            else:
                failed += 1
        
        # Summary
        stats = {
            "total": len(md_files),
            "successful": successful,
            "failed": failed
        }
        
        logger.info(f"Conversion complete: {successful}/{len(md_files)} successful")
        
        return stats


def main():
    """Main function."""
    # Define directories
    source_dir = "/Users/timwark/Projects/Github Local/GraphRAG/agentic-rag-knowledge-graph/big_tech_docs"
    output_dir = "/Users/timwark/Projects/Github Local/GraphRAG/agentic-rag-knowledge-graph/tests/docs"
    
    # Check if source directory exists
    if not Path(source_dir).exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    # Create converter and convert files
    converter = MarkdownToPDFConverter(source_dir, output_dir)
    stats = converter.convert_all()
    
    # Print summary
    print("\n" + "="*60)
    print("MARKDOWN TO PDF CONVERSION SUMMARY")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} files failed to convert. Check logs for details.")
        sys.exit(1)
    else:
        print(f"\n✅ All {stats['successful']} files converted successfully!")


if __name__ == "__main__":
    main()