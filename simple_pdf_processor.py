#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple PDF Processor - A standalone script to test PDF text extraction without Supabase.

This script processes PDF files and extracts text content using pypdfium2.
"""

import os
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimplePDFProcessor")

# Import the PDF processor
from pdf_processor import PDFProcessor, PDFDocument

def process_file(file_path):
    """Process a single PDF file and print the extracted text."""
    processor = PDFProcessor()
    
    logger.info(f"Processing PDF file: {file_path}")
    
    # Process the PDF file
    doc = processor.process_file(file_path)
    
    print(f"\nDocument: {doc.book_name}")
    print(f"Pages: {len(doc.pages)}")
    
    # Print page previews
    for i, page in enumerate(doc.pages):
        print(f"\nPage {i+1} preview:")
        print("-" * 40)
        preview = page.text[:300] + "..." if len(page.text) > 300 else page.text
        print(preview)
        print("-" * 40)

def process_directory(directory_path):
    """Process all PDF files in a directory."""
    processor = PDFProcessor()
    
    logger.info(f"Processing all PDFs in directory: {directory_path}")
    
    # Process all PDFs in the directory
    documents = processor.process_directory(directory_path)
    
    print(f"\nProcessed {len(documents)} documents")
    
    # Print a summary of each document
    for doc in documents:
        print(f"\nDocument: {doc.book_name}")
        print(f"Pages: {len(doc.pages)}")
        if doc.pages:
            # Print a preview of the first page
            print(f"\nFirst page preview:")
            print("-" * 40)
            preview = doc.pages[0].text[:300] + "..." if len(doc.pages[0].text) > 300 else doc.pages[0].text
            print(preview)
            print("-" * 40)

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python simple_pdf_processor.py <pdf_file_or_directory>")
        return
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        process_directory(path)
    elif os.path.isfile(path) and path.lower().endswith('.pdf'):
        process_file(path)
    else:
        print(f"Error: {path} is not a valid PDF file or directory.")

if __name__ == "__main__":
    main() 