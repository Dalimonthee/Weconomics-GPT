#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Processor - Converts PDFs to text using pypdf with page-level chunking.

This module processes PDF files and extracts text content on a per-page basis,
preserving page boundaries and attaching metadata.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Use pypdf for PDF processing
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDFProcessor")

class PDFDocument:
    """Represents a processed PDF document with extracted text and metadata."""
    
    def __init__(self, file_path: str):
        """
        Initialize a PDF document from a file path.
        
        Args:
            file_path: Path to the PDF file
        """
        self.file_path = Path(file_path)
        self.book_name = self.file_path.stem  # Use filename without extension as book name
        self.pages = []
        self.metadata = {
            "title": self.book_name,
            "source": str(self.file_path),
            "file_size": os.path.getsize(file_path),
        }
    
    def __repr__(self) -> str:
        return f"PDFDocument(book_name={self.book_name}, pages={len(self.pages)})"


class PageChunk:
    """Represents a single page chunk extracted from a PDF document."""
    
    def __init__(self, text: str, page_number: int, book_name: str, doc_metadata: Dict = None):
        """
        Initialize a page chunk with text and metadata.
        
        Args:
            text: The text content of the page
            page_number: The page number (1-indexed)
            book_name: The name of the book this page belongs to
            doc_metadata: Optional additional document metadata
        """
        self.text = text
        self.metadata = {
            "page_number": page_number,
            "book_name": book_name,
        }
        
        # Add any additional document metadata if provided
        if doc_metadata:
            # Only add select metadata to avoid bloating the chunk
            for key in ["title", "author", "subject", "creator"]:
                if key in doc_metadata:
                    self.metadata[key] = doc_metadata[key]
    
    def __repr__(self) -> str:
        return f"PageChunk(book={self.metadata['book_name']}, page={self.metadata['page_number']}, chars={len(self.text)})"
    
    @property
    def id(self) -> str:
        """Generate a unique ID for this chunk based on book name and page."""
        return f"{self.metadata['book_name']}_{self.metadata['page_number']}"


class PDFProcessor:
    """Processes PDF files and extracts text content with metadata."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        logger.info("Initializing PDF processor with pypdf")
    
    def process_file(self, file_path: str) -> PDFDocument:
        """
        Process a single PDF file and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDFDocument object containing the processed pages and metadata
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            # Create a new document object
            doc = PDFDocument(file_path)
            
            # Open the PDF with pypdf
            reader = PdfReader(file_path)
            
            # Extract document metadata if available
            if reader.metadata:
                for key, value in reader.metadata.items():
                    if value and key.startswith('/'):
                        # Strip the leading slash and convert to lowercase
                        clean_key = key[1:].lower()
                        doc.metadata[clean_key] = value
            
            # Extract text from each page
            for i, page in enumerate(reader.pages):
                # Use extract_text from pypdf
                text = page.extract_text()
                
                # Clean up the text (remove excessive whitespace)
                text = self._clean_text(text)
                
                # Create a chunk for this page (adding 1 to make it 1-indexed)
                if text.strip():  # Only add non-empty pages
                    chunk = PageChunk(
                        text=text,
                        page_number=i + 1,
                        book_name=doc.book_name,
                        doc_metadata=doc.metadata
                    )
                    doc.pages.append(chunk)
            
            logger.info(f"Successfully processed {file_path}: extracted {len(doc.pages)} pages")
            return doc
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            # Return an empty document if processing fails
            return PDFDocument(file_path)
    
    def process_directory(self, directory_path: str) -> List[PDFDocument]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files
            
        Returns:
            List of PDFDocument objects
        """
        logger.info(f"Processing all PDFs in directory: {directory_path}")
        
        # Make sure the directory exists
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        documents = []
        
        # Process all PDF files in the directory
        for file_path in directory.glob("*.pdf"):
            try:
                doc = self.process_file(str(file_path))
                if doc and doc.pages:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(documents)} PDF documents from {directory_path}")
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text to improve quality.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single one
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()


def main():
    """Test the PDF processor by processing a sample PDF."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file_or_directory>")
        return
    
    path = sys.argv[1]
    processor = PDFProcessor()
    
    if os.path.isdir(path):
        docs = processor.process_directory(path)
        print(f"Processed {len(docs)} documents")
        
        # Print a summary of each document
        for doc in docs:
            print(f"\nDocument: {doc.book_name}")
            print(f"Pages: {len(doc.pages)}")
            if doc.pages:
                # Print a preview of the first page
                print(f"First page preview: {doc.pages[0].text[:100]}...")
    else:
        doc = processor.process_file(path)
        print(f"Document: {doc.book_name}")
        print(f"Pages: {len(doc.pages)}")
        
        # Print the first page if available
        if doc.pages:
            print(f"\nFirst page preview:")
            print(doc.pages[0].text[:500])


if __name__ == "__main__":
    main() 