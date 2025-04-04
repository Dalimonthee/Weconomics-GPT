#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector Store Module - Handles embedding generation and storage in Supabase.

This module interfaces with Google's Gemini embedding model for generating
embeddings and Supabase Vector Store for storage and retrieval.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback

# Data handling
import numpy as np
from tqdm import tqdm

# Environment variables
from dotenv import load_dotenv

# Google Gemini for embeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Supabase for storage
from supabase import create_client, Client

# Local imports
# Import the PageChunk and PDFDocument classes
from pdf_processor import PageChunk, PDFDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VectorStore")

# Load environment variables
load_dotenv()

# Get configuration from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "book_pages")


class EmbeddingGenerator:
    """Generates embeddings using Google's Gemini embedding model."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            model_name: Name of the embedding model to use (defaults to environment variable)
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model_name or EMBEDDING_MODEL
        
        if not self.api_key:
            raise ValueError("Google API key is required for embedding generation")
            
        # Configure Google Generative AI
        genai.configure(api_key=self.api_key)
        
        # Initialize the embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.model_name)
        
        logger.info(f"Initialized embedding generator with model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector in case of error
            return [0.0] * EMBEDDING_DIMENSION
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of embeddings to generate at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches with a progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            
            # Generate embeddings for the batch
            batch_embeddings = []
            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error generating embedding for batch: {str(e)}")
                    # Use a zero vector for failed embeddings
                    batch_embeddings.append([0.0] * EMBEDDING_DIMENSION)
            
            embeddings.extend(batch_embeddings)
            
            # Add a small delay to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return embeddings


class SupabaseVectorStore:
    """Handles storage and retrieval of vectors in Supabase."""
    
    def __init__(
        self, 
        url: str = None, 
        key: str = None, 
        collection_name: str = None,
        embedding_dim: int = None
    ):
        """
        Initialize the Supabase vector store.
        
        Args:
            url: Supabase URL (defaults to environment variable)
            key: Supabase API key (defaults to environment variable)
            collection_name: Name of the collection to use (defaults to environment variable)
            embedding_dim: Dimension of embeddings (defaults to environment variable)
        """
        self.url = url or SUPABASE_URL
        self.key = key or SUPABASE_KEY
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_dim = embedding_dim or EMBEDDING_DIMENSION
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key are required")
        
        # Initialize Supabase client
        self.supabase = create_client(self.url, self.key)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info(f"Initialized Supabase vector store with collection: {self.collection_name}")
        
        # Ensure the collection exists
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create the vector collection if it doesn't exist."""
        try:
            # Check if the collection exists using REST API instead of SQL
            res = self.supabase.table('vector_collections').select('*').eq('name', self.collection_name).execute()
            
            if not res.data:
                # Create the collection if it doesn't exist
                logger.info(f"Creating vector collection: {self.collection_name}")
                
                # Use a different approach for creating the table - this is now done via migrations
                # or the Supabase dashboard rather than direct SQL execution
                logger.info(f"Please create the {self.collection_name} table in Supabase dashboard with pgvector support")
                
                # Register the collection in the vector_collections table
                self.supabase.table('vector_collections').insert({
                    'name': self.collection_name,
                    'dimension': self.embedding_dim,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }).execute()
                
                logger.info(f"Successfully registered vector collection: {self.collection_name}")
            else:
                logger.info(f"Vector collection already exists: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            logger.error(traceback.format_exc())
    
    def insert_chunk(self, chunk: PageChunk) -> bool:
        """
        Insert a single chunk into the vector store.
        
        Args:
            chunk: PageChunk object containing text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for the chunk
            embedding = self.embedding_generator.generate_embedding(chunk.text)
            
            # Insert into Supabase
            self.supabase.table(self.collection_name).insert({
                'id': chunk.id,
                'content': chunk.text,
                'metadata': json.dumps(chunk.metadata),
                'embedding': embedding
            }).execute()
            
            return True
        
        except Exception as e:
            logger.error(f"Error inserting chunk {chunk.id}: {str(e)}")
            return False
    
    def insert_chunks(self, chunks: List[PageChunk], batch_size: int = 20) -> int:
        """
        Insert multiple chunks into the vector store.
        
        Args:
            chunks: List of PageChunk objects
            batch_size: Number of chunks to insert at once
            
        Returns:
            Number of successfully inserted chunks
        """
        if not chunks:
            logger.warning("No chunks to insert")
            return 0
        
        # Extract text for batch embedding generation
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        
        # Insert chunks with embeddings
        successful_inserts = 0
        
        # Process in batches with a progress bar
        for i in tqdm(range(0, len(chunks), batch_size), desc="Inserting chunks"):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            batch_data = []
            for j, chunk in enumerate(batch_chunks):
                try:
                    batch_data.append({
                        'id': chunk.id,
                        'content': chunk.text,
                        'metadata': json.dumps(chunk.metadata),
                        'embedding': batch_embeddings[j]
                    })
                except Exception as e:
                    logger.error(f"Error preparing chunk {chunk.id} for insertion: {str(e)}")
            
            # Insert the batch
            if batch_data:
                try:
                    self.supabase.table(self.collection_name).insert(batch_data).execute()
                    successful_inserts += len(batch_data)
                except Exception as e:
                    logger.error(f"Error inserting batch: {str(e)}")
                    
                    # Try inserting one by one to identify problem records
                    for data in batch_data:
                        try:
                            self.supabase.table(self.collection_name).insert(data).execute()
                            successful_inserts += 1
                        except Exception as e2:
                            logger.error(f"Error inserting chunk {data['id']}: {str(e2)}")
        
        logger.info(f"Successfully inserted {successful_inserts} out of {len(chunks)} chunks")
        return successful_inserts
    
    def insert_document(self, document: PDFDocument) -> int:
        """
        Insert all pages from a document into the vector store.
        
        Args:
            document: PDFDocument object containing pages
            
        Returns:
            Number of successfully inserted pages
        """
        if not document.pages:
            logger.warning(f"Document {document.book_name} has no pages to insert")
            return 0
        
        logger.info(f"Inserting {len(document.pages)} pages from document {document.book_name}")
        return self.insert_chunks(document.pages)
    
    def query(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing content and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query_text)
            
            # Query using REST API instead of SQL
            # This is a workaround since the sql method is not available
            # We'll use a function call instead
            result = self.supabase.rpc(
                'match_documents', 
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.5,
                    'match_count': limit,
                    'collection_name': self.collection_name
                }
            ).execute()
            
            # Process results
            chunks = []
            for item in result.data:
                # Parse metadata from JSON string
                metadata = json.loads(item['metadata']) if isinstance(item['metadata'], str) else item['metadata']
                
                chunks.append({
                    'id': item['id'],
                    'content': item['content'],
                    'metadata': metadata,
                    'similarity': item['similarity']
                })
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary containing statistics about the collection
        """
        try:
            # Query the collection table for the count using REST API
            count_result = self.supabase.table(self.collection_name).select('id', count='exact').execute()
            count = len(count_result.data) if count_result.data else 0
            
            # Get book statistics using REST API
            books_result = self.supabase.table(self.collection_name).select('metadata').execute()
            
            # Process book stats manually
            book_stats = {}
            for item in books_result.data:
                metadata = json.loads(item['metadata']) if isinstance(item['metadata'], str) else item['metadata']
                book_name = metadata.get('book_name', 'Unknown')
                page_number = int(metadata.get('page_number', 0))
                
                if book_name not in book_stats:
                    book_stats[book_name] = {
                        'page_count': 1,
                        'min_page': page_number,
                        'max_page': page_number
                    }
                else:
                    book_stats[book_name]['page_count'] += 1
                    book_stats[book_name]['min_page'] = min(book_stats[book_name]['min_page'], page_number)
                    book_stats[book_name]['max_page'] = max(book_stats[book_name]['max_page'], page_number)
            
            # Format book stats for return
            books = []
            for book_name, stats in book_stats.items():
                books.append({
                    'book_name': book_name,
                    'page_count': stats['page_count'],
                    'min_page': stats['min_page'],
                    'max_page': stats['max_page']
                })
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'books': books
            }
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }


def main():
    """Test the vector store by processing a sample directory of PDFs."""
    import sys
    from pdf_processor import PDFProcessor
    
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <pdf_directory>")
        return
    
    pdf_dir = sys.argv[1]
    
    # Process PDFs
    pdf_processor = PDFProcessor()
    documents = pdf_processor.process_directory(pdf_dir)
    
    if not documents:
        print(f"No documents found in {pdf_dir}")
        return
    
    # Initialize vector store
    try:
        vector_store = SupabaseVectorStore()
        
        # Print collection stats before insertion
        print("\nCollection stats before insertion:")
        stats = vector_store.get_collection_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        
        # Insert documents
        for doc in documents:
            print(f"\nInserting document: {doc.book_name}")
            inserted_count = vector_store.insert_document(doc)
            print(f"Inserted {inserted_count} out of {len(doc.pages)} pages")
        
        # Print collection stats after insertion
        print("\nCollection stats after insertion:")
        stats = vector_store.get_collection_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        
        for book in stats.get('books', []):
            print(f"Book: {book['book_name']}, Pages: {book['page_count']}")
        
        # Test a query
        if documents:
            # Extract a sample text from the first document
            sample_text = documents[0].pages[0].text[:100] if documents[0].pages else "example query"
            
            print("\nTesting a query with sample text:")
            results = vector_store.query(sample_text, limit=2)
            
            if results:
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"Book: {result['metadata']['book_name']}")
                    print(f"Page: {result['metadata']['page_number']}")
                    print(f"Similarity: {result['similarity']:.4f}")
                    print(f"Preview: {result['content'][:200]}...")
            else:
                print("No results found")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 