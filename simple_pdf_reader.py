#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple PDF Reader - A very basic script to test PDF text extraction using pypdf.

This script processes PDF files and extracts text content using pypdf.
"""

import sys
from pypdf import PdfReader
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pypdf."""
    try:
        # Open the PDF file
        reader = PdfReader(pdf_path)
        
        # Get the number of pages
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")
        
        # Extract text from each page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # Print a preview
            print(f"\nPage {i+1} preview:")
            print("-" * 40)
            preview = text[:300] + "..." if len(text) > 300 else text
            print(preview)
            print("-" * 40)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python simple_pdf_reader.py <pdf_file>")
        return
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: Not a PDF file: {pdf_path}")
        return
    
    print(f"Processing: {pdf_path}")
    extract_text_from_pdf(pdf_path)

if __name__ == "__main__":
    main() 