#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple RAG Demo - A standalone script to demonstrate basic RAG functions.

This script processes PDF files and uses Google Gemini to answer questions
based on the content of those PDFs, without requiring Supabase.
"""

import os
import sys
import logging
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleRAGDemo")

# Import our PDF processor
from pdf_processor import PDFProcessor, PDFDocument, PageChunk

# Import Google Gemini for embeddings and LLM
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required in .env file")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)


class SimpleVectorStore:
    """A simple in-memory vector store for demonstration purposes."""
    
    def __init__(self):
        """Initialize the simple vector store."""
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.documents = []  # List of (text, metadata, embedding) tuples
        logger.info("Initialized simple in-memory vector store")
    
    def add_documents(self, chunks: List[PageChunk]):
        """Add documents to the vector store."""
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk.text)
            self.documents.append((chunk.text, chunk.metadata, embedding))
        
        logger.info(f"Added {len(chunks)} documents to the vector store")
        return len(chunks)
    
    def add_document(self, doc: PDFDocument):
        """Add a document to the vector store."""
        return self.add_documents(doc.pages)
    
    def similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, query: str, limit: int = 3):
        """Search for documents similar to the query."""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarity scores
        results = []
        for text, metadata, embedding in self.documents:
            similarity = self.similarity(query_embedding, embedding)
            results.append((text, metadata, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Return the top results
        top_results = results[:limit]
        
        formatted_results = []
        for text, metadata, similarity in top_results:
            formatted_results.append({
                'content': text,
                'metadata': metadata,
                'similarity': similarity
            })
        
        return formatted_results


class SimpleRAG:
    """A simple RAG system for demonstration purposes."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.pdf_processor = PDFProcessor()
        self.vector_store = SimpleVectorStore()
        
        # Initialize Gemini model for generation
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info("Initialized Simple RAG system")
    
    def process_directory(self, directory_path: str):
        """Process all PDFs in a directory."""
        # Process all PDFs in the directory
        documents = self.pdf_processor.process_directory(directory_path)
        
        if not documents:
            logger.warning(f"No PDF documents found in {directory_path}")
            return 0
        
        # Add all documents to the vector store
        total_added = 0
        for doc in documents:
            added = self.vector_store.add_document(doc)
            total_added += added
            
            logger.info(f"Added {added} pages from {doc.book_name}")
        
        logger.info(f"Processed {len(documents)} documents, added {total_added} pages to vector store")
        return total_added
    
    def process_file(self, file_path: str):
        """Process a single PDF file."""
        # Process the PDF file
        document = self.pdf_processor.process_file(file_path)
        
        if not document.pages:
            logger.warning(f"No pages found in {file_path}")
            return 0
        
        # Add the document to the vector store
        added = self.vector_store.add_document(document)
        
        logger.info(f"Processed {file_path}, added {added} pages to vector store")
        return added
    
    def query(self, query_text: str, limit: int = 3):
        """Query the vector store and generate a response."""
        # Search for relevant documents
        results = self.vector_store.search(query_text, limit=limit)
        
        if not results:
            return "I don't have any information about that. Please try another question."
        
        # Format results for the LLM
        context = ""
        for i, result in enumerate(results):
            context += f"\nDocument {i+1} (from {result['metadata']['book_name']}, page {result['metadata']['page_number']}):\n"
            context += result['content'] + "\n"
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided documents.
Please provide a comprehensive answer to the question based ONLY on the information in the documents.
If you don't know the answer based on the documents, say so.

DOCUMENTS:
{context}

QUESTION: {query_text}

ANSWER:"""
        
        # Generate response
        response = self.model.generate_content(prompt)
        
        # Return the response text
        return response.text


def main():
    """Main function to process command line arguments and run the RAG system."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_rag_demo.py process-dir <directory>")
        print("  python simple_rag_demo.py process-file <file.pdf>")
        print("  python simple_rag_demo.py query <query_text>")
        print("  python simple_rag_demo.py interactive")
        return
    
    # Initialize the RAG system
    rag = SimpleRAG()
    
    command = sys.argv[1]
    
    if command == "process-dir":
        if len(sys.argv) < 3:
            print("Error: Directory path is required")
            return
        
        directory = sys.argv[2]
        rag.process_directory(directory)
    
    elif command == "process-file":
        if len(sys.argv) < 3:
            print("Error: File path is required")
            return
        
        file_path = sys.argv[2]
        rag.process_file(file_path)
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: Query text is required")
            return
        
        # First process the default directory if no documents in vector store
        if not rag.vector_store.documents:
            default_dir = "data/books"
            print(f"Processing PDFs in default directory: {default_dir}")
            rag.process_directory(default_dir)
            
            if not rag.vector_store.documents:
                print("No documents processed. Please add PDFs to data/books directory or specify a directory.")
                return
        
        query_text = sys.argv[2]
        print(f"\nQuery: {query_text}")
        print("\nGenerating response...")
        response = rag.query(query_text)
        print("\nResponse:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    
    elif command == "interactive":
        # First process the default directory if no documents in vector store
        if not rag.vector_store.documents:
            default_dir = "data/books"
            print(f"Processing PDFs in default directory: {default_dir}")
            rag.process_directory(default_dir)
            
            if not rag.vector_store.documents:
                print("No documents processed. Please add PDFs to data/books directory or specify a directory.")
                return
        
        print("\nInteractive mode. Type 'exit' to quit.")
        while True:
            query_text = input("\nEnter your question: ")
            
            if query_text.lower() in ["exit", "quit", "q"]:
                break
            
            if not query_text.strip():
                continue
            
            print("\nGenerating response...")
            response = rag.query(query_text)
            print("\nResponse:")
            print("-" * 40)
            print(response)
            print("-" * 40)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: process-dir, process-file, query, interactive")


if __name__ == "__main__":
    main() 