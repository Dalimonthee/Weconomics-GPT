"""Vector store module for Supabase integration."""

import logging
import os
import time
from typing import List, Optional, Dict, Any, Union, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from supabase import create_client

from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SupabaseVectorStoreManager:
    """Manager class for Supabase vector store operations."""

    def __init__(self, min_content_length: int = 200, force_refresh: bool = False):
        """Initialize the Supabase vector store manager.
        
        Args:
            min_content_length: Minimum content length to consider a result valid
            force_refresh: Force refresh the Supabase connection
        """
        logger.info(
            f"Initializing Supabase Vector Store Manager (min_length={min_content_length}, refresh={force_refresh})"
        )
        self.supabase_url = config.supabase.url
        self.supabase_key = config.supabase.key
        self.table_name = config.supabase.table_name
        self.query_name = getattr(config.supabase, 'query_name', None)  # Make query_name optional
        self.embedding_dimension = config.embedding_dimension
        self.min_content_length = min_content_length
        
        # Initialize the vector store
        self._initialize_vector_store(force_refresh)
        
        # Log table statistics if forcing refresh
        if force_refresh:
            self.log_table_statistics()
    
    def _initialize_vector_store(self, force_refresh: bool = False):
        """Initialize the vector store with the Supabase client.
        
        Args:
            force_refresh: Force refresh the Supabase connection
        """
        try:
            # Create Supabase client
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            
            # Create embedding function
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.gemini.api_key,
                dimensions=self.embedding_dimension
            )
            
            # Create vector store with optional query_name
            vector_store_args = {
                "client": self.supabase_client,
                "embedding": embedding_function,
                "table_name": self.table_name,
            }
            
            # Add query_name only if it's specified
            if self.query_name:
                vector_store_args["query_name"] = self.query_name
            
            # Create the vector store
            self.vector_store = SupabaseVectorStore(**vector_store_args)
            
            # Log initialization success
            logger.info(f"Supabase Vector Store initialized with table: {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Supabase Vector Store: {str(e)}")
            raise
    
    def refresh_connection(self):
        """Refresh the Supabase connection and vector store."""
        logger.info("Refreshing Supabase connection")
        self._initialize_vector_store(force_refresh=True)
        logger.info("Supabase connection refreshed")
    
    def search(self, query: str, k: int = 8) -> List[Document]:
        """Search for similar documents in the vector store.
        
        Args:
            query: The query to search for
            k: Number of results to return (default: 8)
            
        Returns:
            List of documents
        """
        logger.info(f"Searching for query: '{query}' (top {k} results)")
        try:
            start_time = time.time()
            
            # Perform the search with similarity_search
            docs = self.vector_store.similarity_search(
                query=query,
                k=k*2  # Fetch more initially to filter for quality
            )
            
            # Filter for content length
            filtered_docs = [
                doc for doc in docs
                if len(doc.page_content) >= self.min_content_length
            ]
            
            # If filtering removed too many results, try again with the original query
            # but a much larger k to find more substantial content
            if len(filtered_docs) < k and len(docs) < k*2:
                logger.info(f"Initial search returned only {len(filtered_docs)} substantial results, fetching more...")
                larger_k = k * 4
                docs_extended = self.vector_store.similarity_search(
                    query=query,
                    k=larger_k
                )
                
                # Filter and add new unique documents
                existing_content = {doc.page_content for doc in filtered_docs}
                for doc in docs_extended:
                    if doc.page_content not in existing_content and len(doc.page_content) >= self.min_content_length:
                        filtered_docs.append(doc)
                        existing_content.add(doc.page_content)
                        
                        # Break if we have enough documents
                        if len(filtered_docs) >= k:
                            break
            
            # Return at most k documents
            result_docs = filtered_docs[:k]
            
            end_time = time.time()
            logger.info(f"Search completed in {end_time - start_time:.2f}s. Found {len(result_docs)} documents.")
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Error searching Supabase: {str(e)}")
            return []
    
    # Keep the similarity_search method for backward compatibility
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Search for similar documents in the vector store (alias for search).
        
        Args:
            query: The query to search for
            k: Number of results to return (default: 8)
            
        Returns:
            List of documents
        """
        return self.search(query, k)
    
    def log_table_statistics(self) -> Dict[str, Any]:
        """Fetch and log statistics about the Supabase table.
        
        Returns:
            Dictionary with table statistics
        """
        logger.info(f"Fetching statistics for table: {self.table_name}")
        try:
            # Fetch record count
            count_response = self.supabase_client.table(self.table_name).select("count", count="exact").execute()
            total_records = count_response.count if hasattr(count_response, "count") else "Unknown"
            
            # Sample a few records to get average content length
            sample_response = self.supabase_client.table(self.table_name).select("content").limit(100).execute()
            
            if hasattr(sample_response, "data") and sample_response.data:
                content_lengths = [len(record.get("content", "")) for record in sample_response.data]
                avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                min_length = min(content_lengths) if content_lengths else 0
                max_length = max(content_lengths) if content_lengths else 0
                
                # Count documents by size ranges
                size_ranges = {
                    "tiny (< 200 chars)": sum(1 for l in content_lengths if l < 200),
                    "small (200-500 chars)": sum(1 for l in content_lengths if 200 <= l < 500),
                    "medium (500-1000 chars)": sum(1 for l in content_lengths if 500 <= l < 1000),
                    "large (1000-5000 chars)": sum(1 for l in content_lengths if 1000 <= l < 5000),
                    "very large (≥ 5000 chars)": sum(1 for l in content_lengths if l >= 5000)
                }
                
                # Percentage of documents in each size range
                size_percentages = {
                    range_name: count / len(content_lengths) * 100 if content_lengths else 0
                    for range_name, count in size_ranges.items()
                }
                
                # Log the statistics
                logger.info(f"Table statistics for {self.table_name}:")
                logger.info(f"Total records: {total_records}")
                logger.info(f"Average content length: {avg_content_length:.2f} characters")
                logger.info(f"Content length range: {min_length} to {max_length} characters")
                
                for range_name, percentage in size_percentages.items():
                    logger.info(f"{range_name}: {percentage:.1f}% of documents")
                
                # Print statistics to console for user visibility
                print("\n" + "=" * 40)
                print(f"SUPABASE TABLE STATISTICS: {self.table_name}")
                print(f"Total records: {total_records}")
                print(f"Average content length: {avg_content_length:.1f} characters")
                print(f"Content length range: {min_length} to {max_length} characters")
                print("\nDocument size distribution (from sample of {len(content_lengths)} documents):")
                
                for range_name, percentage in size_percentages.items():
                    print(f"  {range_name}: {percentage:.1f}% ({size_ranges[range_name]} docs)")
                
                # Print info about minimum content length setting
                print(f"\nCurrent minimum content length filter: {self.min_content_length} characters")
                print(f"Documents that will be filtered out: {sum(1 for l in content_lengths if l < self.min_content_length)} ({sum(1 for l in content_lengths if l < self.min_content_length) / len(content_lengths) * 100:.1f}%)")
                print("=" * 40)
                
                # Return the statistics
                return {
                    "total_records": total_records,
                    "avg_content_length": avg_content_length,
                    "min_length": min_length,
                    "max_length": max_length,
                    "size_ranges": size_ranges,
                    "size_percentages": size_percentages
                }
            
            else:
                logger.warning("No sample data available to calculate statistics")
                print("\n" + "=" * 40)
                print(f"SUPABASE TABLE STATISTICS: {self.table_name}")
                print(f"Total records: {total_records}")
                print("No sample data available to calculate detailed statistics")
                print("=" * 40)
                
                return {"total_records": total_records}
            
        except Exception as e:
            logger.error(f"Error fetching table statistics: {str(e)}")
            print(f"\nError fetching table statistics: {str(e)}")
            return {"error": str(e)} 