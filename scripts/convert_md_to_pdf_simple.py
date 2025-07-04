#!/usr/bin/env python3
"""
Simple Markdown to PDF converter using reportlab.

This script converts all .md files in the big_tech_docs directory to PDF format
using a simpler approach that doesn't require system dependencies.
"""

import os
import sys
import re
from pathlib import Path
import logging
from typing import List, Tuple

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue, gray
except ImportError:
    print("reportlab not found. Installing...")
    os.system("pip install reportlab")
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.units import inch
        from reportlab.lib.colors import black, blue, gray
    except ImportError as e:
        print(f"Failed to install/import reportlab: {e}")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleMarkdownToPDF:
    """Simple Markdown to PDF converter using reportlab."""
    
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
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        logger.info(f"Simple converter initialized: {source_dir} -> {output_dir}")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_LEFT,
            textColor=blue
        ))
        
        # Heading 1 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            spaceBefore=20,
            textColor=blue
        ))
        
        # Heading 2 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=blue
        ))
        
        # Heading 3 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=blue
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            spaceBefore=12,
            backColor=gray
        ))
        
        # List style
        self.styles.add(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            leftIndent=20,
            spaceAfter=6,
            bulletIndent=10
        ))
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters for reportlab."""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text
    
    def _process_markdown_formatting(self, text: str) -> str:
        """
        Process markdown formatting in text.
        
        Args:
            text: Text with markdown formatting
            
        Returns:
            Text with HTML formatting for reportlab
        """
        # First escape HTML characters, but preserve what we'll add
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Bold text (do this before italic to avoid conflicts)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Italic text (be careful not to match bold)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<i>\1</i>', text)
        
        # Code inline
        text = re.sub(r'`([^`]+?)`', r'<font name="Courier">\1</font>', text)
        
        # Links (simple handling)
        text = re.sub(r'\[([^\]]+?)\]\([^)]+?\)', r'<u>\1</u>', text)
        
        return text

    def _process_markdown_line(self, line: str) -> Tuple[str, str]:
        """
        Process a single markdown line and return (content, style).
        
        Args:
            line: Markdown line
            
        Returns:
            Tuple of (processed_content, style_name)
        """
        line = line.strip()
        
        if not line:
            return "", "Normal"
        
        # Headers (process before applying formatting)
        if line.startswith('# '):
            content = self._process_markdown_formatting(line[2:])
            return f"<b>{content}</b>", "CustomTitle"
        elif line.startswith('## '):
            content = self._process_markdown_formatting(line[3:])
            return f"<b>{content}</b>", "CustomHeading1"
        elif line.startswith('### '):
            content = self._process_markdown_formatting(line[4:])
            return f"<b>{content}</b>", "CustomHeading2"
        elif line.startswith('#### '):
            content = self._process_markdown_formatting(line[5:])
            return f"<b>{content}</b>", "CustomHeading3"
        
        # Lists (handle before general formatting)
        if line.startswith('- ') or line.startswith('* '):
            content = self._process_markdown_formatting(line[2:])
            return f"• {content}", "ListItem"
        elif re.match(r'^\d+\. ', line):
            content_without_number = re.sub(r'^\d+\. ', '', line)
            content = self._process_markdown_formatting(content_without_number)
            number = re.match(r'^(\d+)\.', line).group(1)
            return f"{number}. {content}", "ListItem"
        
        # Code blocks (simplified)
        if line.startswith('```'):
            return "", "Normal"  # Skip code block markers
        
        # Regular paragraph with formatting
        content = self._process_markdown_formatting(line)
        return content, "Normal"
    
    def _convert_markdown_to_story(self, md_content: str) -> List:
        """
        Convert markdown content to reportlab story elements.
        
        Args:
            md_content: Markdown content as string
            
        Returns:
            List of reportlab flowables
        """
        story = []
        lines = md_content.split('\n')
        
        in_code_block = False
        code_block_content = []
        
        for line in lines:
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End code block
                    if code_block_content:
                        code_text = self._escape_html('\n'.join(code_block_content))
                        story.append(Paragraph(code_text, self.styles['CustomCode']))
                        story.append(Spacer(1, 6))
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                continue
            
            if in_code_block:
                code_block_content.append(line)
                continue
            
            # Process regular lines
            content, style_name = self._process_markdown_line(line)
            
            if content:
                story.append(Paragraph(content, self.styles[style_name]))
                
                # Add spacing after paragraphs
                if style_name == "Normal":
                    story.append(Spacer(1, 6))
                elif style_name.startswith("Custom"):
                    story.append(Spacer(1, 8))
        
        return story
    
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
            
            # Create PDF filename
            pdf_filename = md_file.stem + '.pdf'
            pdf_path = self.output_dir / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Convert markdown to story
            story = self._convert_markdown_to_story(md_content)
            
            # Build PDF
            doc.build(story)
            
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
    converter = SimpleMarkdownToPDF(source_dir, output_dir)
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