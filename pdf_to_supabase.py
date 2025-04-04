#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF to Supabase - A script to process PDFs and store them in Supabase Vector Store.

This script:
1. Processes PDF files from the data/books directory
2. Chunks them by page
3. Generates embeddings using Google Gemini
4. Stores the embeddings in Supabase with metadata
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDF2Supabase")

# Import our modules
from pdf_processor import PDFProcessor
from vector_store import SupabaseVectorStore

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Default folder for PDFs
DEFAULT_PDF_FOLDER = "data/books"

def process_file(file_path):
    """Process a single PDF file and store it in Supabase."""
    logger.info(f"Processing file: {file_path}")
    
    # Create processor and vector store
    pdf_processor = PDFProcessor()
    vector_store = SupabaseVectorStore()
    
    # Process the PDF
    document = pdf_processor.process_file(file_path)
    
    if not document.pages:
        logger.warning(f"No pages found in {file_path}")
        return 0
    
    # Store in Supabase
    inserted = vector_store.insert_document(document)
    
    logger.info(f"Processed {file_path}: {inserted} pages stored in Supabase")
    return inserted

def process_directory(directory_path=DEFAULT_PDF_FOLDER):
    """Process all PDF files in a directory and store them in Supabase."""
    logger.info(f"Processing directory: {directory_path}")
    
    # Create directory if it doesn't exist
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    # Create processor and vector store
    pdf_processor = PDFProcessor()
    vector_store = SupabaseVectorStore()
    
    # Process all PDFs
    documents = pdf_processor.process_directory(directory_path)
    
    if not documents:
        logger.warning(f"No PDF documents found in {directory_path}")
        return 0
    
    # Store all documents in Supabase
    total_inserted = 0
    for doc in documents:
        inserted = vector_store.insert_document(doc)
        total_inserted += inserted
        logger.info(f"Document {doc.book_name}: {inserted} pages stored in Supabase")
    
    logger.info(f"Total processed: {len(documents)} documents, {total_inserted} pages stored in Supabase")
    return total_inserted

def main():
    """Main function."""
    # Make sure the default directory exists
    Path(DEFAULT_PDF_FOLDER).mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) < 2:
        # Default behavior: process all PDFs in the default folder
        print(f"Processing all PDFs in default folder: {DEFAULT_PDF_FOLDER}")
        process_directory()
        return
    
    command = sys.argv[1]
    
    if command == "file":
        if len(sys.argv) < 3:
            print(f"Error: File path is required")
            print(f"Available PDFs in {DEFAULT_PDF_FOLDER}:")
            pdf_files = list(Path(DEFAULT_PDF_FOLDER).glob("*.pdf"))
            if pdf_files:
                for pdf_file in pdf_files:
                    print(f"  - {pdf_file.name}")
                print(f"\nUsage: python pdf_to_supabase.py file <filename>")
                print(f"Example: python pdf_to_supabase.py file {pdf_files[0].name}")
            else:
                print(f"  No PDF files found. Please add PDF files to {DEFAULT_PDF_FOLDER}")
            return
        
        # Check if the file path is just a filename or a full path
        file_path = sys.argv[2]
        if not os.path.dirname(file_path):
            # Just a filename, prepend the default folder
            file_path = os.path.join(DEFAULT_PDF_FOLDER, file_path)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        process_file(file_path)
    
    elif command == "dir":
        if len(sys.argv) < 3:
            # Use default directory
            process_directory()
        else:
            directory_path = sys.argv[2]
            process_directory(directory_path)
    
    elif command == "stats":
        # Get stats from Supabase
        vector_store = SupabaseVectorStore()
        stats = vector_store.get_collection_stats()
        
        print("\nSupabase Vector Store Statistics:")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total chunks: {stats['total_chunks']}")
        
        if stats.get("books"):
            print("\nBooks:")
            for book in stats["books"]:
                print(f"  {book['book_name']}: {book['page_count']} pages (pages {book['min_page']} to {book['max_page']})")
        else:
            print("\nNo books found in the collection.")
    
    elif command == "list":
        # List all PDFs in the default folder
        print(f"\nPDF files in {DEFAULT_PDF_FOLDER}:")
        pdf_files = list(Path(DEFAULT_PDF_FOLDER).glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
        else:
            print(f"  No PDF files found. Please add PDF files to {DEFAULT_PDF_FOLDER}")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands:")
        print(f"  python pdf_to_supabase.py              # Process all PDFs in {DEFAULT_PDF_FOLDER}")
        print(f"  python pdf_to_supabase.py file <name>  # Process a single PDF file")
        print(f"  python pdf_to_supabase.py dir [path]   # Process all PDFs in a directory (default: {DEFAULT_PDF_FOLDER})")
        print(f"  python pdf_to_supabase.py list         # List all PDFs in {DEFAULT_PDF_FOLDER}")
        print(f"  python pdf_to_supabase.py stats        # Show statistics about stored PDFs")

if __name__ == "__main__":
    main() 