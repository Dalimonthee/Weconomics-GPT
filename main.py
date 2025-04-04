#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF RAG System - Main Module

This script ties together PDF processing and vector storage.
It ingests PDFs, converts them to text using pdfium, chunks them by page,
and stores them in Supabase with Gemini embeddings.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our modules
from pdf_processor import PDFProcessor, PDFDocument
from vector_store import SupabaseVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PDFRagSystem")


class PDFRagSystem:
    """Main class for the PDF RAG system."""
    
    def __init__(self, collection_name: str = None):
        """
        Initialize the PDF RAG system.
        
        Args:
            collection_name: Name of the collection to use in Supabase
        """
        self.pdf_processor = PDFProcessor()
        self.vector_store = SupabaseVectorStore(collection_name=collection_name)
        
        logger.info("Initialized PDF RAG system")
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Process all PDFs in a directory and store them in the vector store.
        
        Args:
            directory_path: Path to the directory containing PDF files
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Process all PDFs in the directory
        documents = self.pdf_processor.process_directory(directory_path)
        
        if not documents:
            logger.warning(f"No PDF documents found in {directory_path}")
            return {"status": "error", "message": "No PDF documents found"}
        
        # Get collection stats before insertion
        before_stats = self.vector_store.get_collection_stats()
        before_count = before_stats.get('total_chunks', 0)
        
        # Insert all documents into the vector store
        total_pages = sum(len(doc.pages) for doc in documents)
        total_inserted = 0
        
        for doc in documents:
            logger.info(f"Inserting document: {doc.book_name} with {len(doc.pages)} pages")
            inserted = self.vector_store.insert_document(doc)
            total_inserted += inserted
            
            logger.info(f"Inserted {inserted} out of {len(doc.pages)} pages for {doc.book_name}")
        
        # Get collection stats after insertion
        after_stats = self.vector_store.get_collection_stats()
        after_count = after_stats.get('total_chunks', 0)
        
        # Prepare result
        result = {
            "status": "success",
            "documents_processed": len(documents),
            "total_pages_found": total_pages,
            "total_pages_inserted": total_inserted,
            "before_chunk_count": before_count,
            "after_chunk_count": after_count,
            "books": []
        }
        
        # Add book statistics
        for book in after_stats.get('books', []):
            result["books"].append({
                "book_name": book.get('book_name'),
                "page_count": book.get('page_count'),
                "min_page": book.get('min_page'),
                "max_page": book.get('max_page')
            })
        
        return result
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file and store it in the vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing file: {file_path}")
        
        # Process the PDF file
        document = self.pdf_processor.process_file(file_path)
        
        if not document.pages:
            logger.warning(f"No pages found in {file_path}")
            return {"status": "error", "message": "No pages found in PDF"}
        
        # Insert the document into the vector store
        inserted = self.vector_store.insert_document(document)
        
        # Prepare result
        result = {
            "status": "success",
            "document": document.book_name,
            "total_pages_found": len(document.pages),
            "total_pages_inserted": inserted
        }
        
        return result
    
    def query(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing content and metadata
        """
        logger.info(f"Querying with: '{query_text}'")
        
        # Query the vector store
        results = self.vector_store.query(query_text, limit=limit)
        
        if not results:
            logger.warning(f"No results found for query: '{query_text}'")
        else:
            logger.info(f"Found {len(results)} results for query")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the system.
        
        Returns:
            Dictionary with system statistics
        """
        logger.info("Getting system statistics")
        
        # Get collection stats
        collection_stats = self.vector_store.get_collection_stats()
        
        return collection_stats


def main():
    """Main function to process command line arguments and run the system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PDF RAG System")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process directory command
    process_dir_parser = subparsers.add_parser("process-dir", help="Process all PDFs in a directory")
    process_dir_parser.add_argument("directory", help="Directory containing PDF files")
    process_dir_parser.add_argument("--collection", help="Collection name in Supabase")
    
    # Process file command
    process_file_parser = subparsers.add_parser("process-file", help="Process a single PDF file")
    process_file_parser.add_argument("file", help="PDF file to process")
    process_file_parser.add_argument("--collection", help="Collection name in Supabase")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    query_parser.add_argument("--collection", help="Collection name in Supabase")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics about the system")
    stats_parser.add_argument("--collection", help="Collection name in Supabase")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the system
    system = PDFRagSystem(collection_name=args.collection if hasattr(args, "collection") else None)
    
    # Run the appropriate command
    if args.command == "process-dir":
        result = system.process_directory(args.directory)
        
        # Print the result
        print("\nProcessing Results:")
        print(f"Documents processed: {result['documents_processed']}")
        print(f"Total pages found: {result['total_pages_found']}")
        print(f"Total pages inserted: {result['total_pages_inserted']}")
        print(f"Before chunk count: {result['before_chunk_count']}")
        print(f"After chunk count: {result['after_chunk_count']}")
        
        if result.get("books"):
            print("\nBooks:")
            for book in result["books"]:
                print(f"  {book['book_name']}: {book['page_count']} pages")
        
    elif args.command == "process-file":
        result = system.process_file(args.file)
        
        # Print the result
        print("\nProcessing Results:")
        print(f"Document: {result['document']}")
        print(f"Total pages found: {result['total_pages_found']}")
        print(f"Total pages inserted: {result['total_pages_inserted']}")
        
    elif args.command == "query":
        results = system.query(args.query, limit=args.limit)
        
        # Print the results
        print(f"\nFound {len(results)} results for query: '{args.query}'")
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Book: {result['metadata']['book_name']}")
            print(f"Page: {result['metadata']['page_number']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print("Content preview:")
            print("-" * 40)
            preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
            print(preview)
            print("-" * 40)
        
    elif args.command == "stats":
        stats = system.get_stats()
        
        # Print the stats
        print("\nSystem Statistics:")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total chunks: {stats['total_chunks']}")
        
        if stats.get("books"):
            print("\nBooks:")
            for book in stats["books"]:
                print(f"  {book['book_name']}: {book['page_count']} pages (pages {book['min_page']} to {book['max_page']})")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 