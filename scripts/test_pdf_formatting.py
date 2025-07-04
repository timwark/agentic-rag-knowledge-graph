#!/usr/bin/env python3
"""
Test script to verify that bold formatting is working in PDFs.

This creates a small test document with various markdown formatting
to validate the conversion is working properly.
"""

import os
from pathlib import Path
from convert_md_to_pdf_simple import SimpleMarkdownToPDF

def create_test_markdown():
    """Create a test markdown file with various formatting."""
    test_content = """# Test Document

This is a test document to verify **bold formatting** is working correctly.

## Section with Bold Elements

Here are some examples:
- **Implementation areas:** This should be bold
- **Strategic rationale:** Also bold  
- Normal text with **embedded bold text** in the middle
- Text with __double underscore bold__ formatting
- Mixed formatting: **bold** and *italic* and `code`

### Subsection

**Key points:**
1. **First point:** Details about the first point
2. **Second point:** More information here
3. Regular point without bold

Some regular text followed by **multiple** **bold** **words** in sequence.

**Note:** This entire line should start with bold.

End of test document.
"""
    
    test_file = Path("test_bold_formatting.md")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    return test_file

def main():
    """Test the bold formatting conversion."""
    print("Creating test markdown file...")
    test_file = create_test_markdown()
    
    try:
        # Create converter
        converter = SimpleMarkdownToPDF(".", "tests/docs")
        
        # Convert the test file
        print("Converting test file to PDF...")
        success = converter.convert_file(test_file)
        
        if success:
            print("✅ Test conversion successful!")
            print("Check tests/docs/test_bold_formatting.pdf to verify bold formatting")
        else:
            print("❌ Test conversion failed!")
            
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
            print("Cleaned up test markdown file")

if __name__ == "__main__":
    main()