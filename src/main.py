"""Main module for the agentic RAG system."""

import asyncio
import json
import logging
import argparse
import re
from typing import Dict, Any, List, Set

from src.agents import AgentManager, UserQuery, SearchPlan
from src.config import config
from src.vector_store import SupabaseVectorStoreManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_keywords(text: str) -> Set[str]:
    """Extract potential keywords from text for highlighting."""
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split by whitespace
    words = text.split()
    # Filter out common words and short words
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                'from', 'of', 'that', 'this', 'these', 'those', 'it', 'they', 'them'}
    keywords = {word for word in words if word not in stopwords and len(word) > 3}
    return keywords


def display_search_plan(search_plan: Dict[str, Any]):
    """Display the search plan in a formatted way."""
    print("\n" + "=" * 40)
    print("SEARCH PLAN:")
    print(f"Reformulated query: {search_plan.get('reformulated_query', 'N/A')}")
    
    print("\nFocus Areas:")
    for area in search_plan.get('focus_areas', []):
        print(f"- {area}")
    
    print("\nKey Concepts:")
    for concept in search_plan.get('key_concepts', []):
        print(f"- {concept}")
    
    print("\nGuidance for Agent:")
    guidance = search_plan.get('guidance', '')
    # Format guidance for better readability
    for line in guidance.split('\n'):
        print(f"> {line}")
    
    print("-" * 40)


def display_search_quality(search_quality: Dict[str, Any]):
    """Display search quality metrics in a readable format."""
    if not search_quality:
        return
    
    print("\n" + "=" * 40)
    print("SEARCH QUALITY METRICS:")
    print(f"Documents found: {search_quality.get('found_documents', 0)}")
    print(f"Average document length: {int(search_quality.get('avg_document_length', 0))} characters")
    
    coverage = search_quality.get('key_concept_coverage', 0)
    print(f"Key concept coverage: {coverage}%")
    
    assessment = search_quality.get('assessment', 'Unknown')
    print(f"Quality assessment: {assessment}")
    
    if 'concept_coverage' in search_quality:
        print("\nConcept Coverage Details:")
        for concept, count in search_quality['concept_coverage'].items():
            print(f"- '{concept}': found in {count} documents")
    
    print("-" * 40)


async def process_single_query(query: str, min_content_length: int = 200, force_refresh: bool = False) -> Dict[str, Any]:
    """Process a single query through the agent manager."""
    logger.info(f"Processing query: {query} with min_content_length={min_content_length}, force_refresh={force_refresh}")
    # Create a custom vector store manager with the specified parameters
    vector_store = SupabaseVectorStoreManager(min_content_length=min_content_length, force_refresh=force_refresh)
    # Create manager with custom vector store
    manager = AgentManager(vector_store=vector_store)
    response = await manager.process_query(query)
    return response


async def interactive_session(min_content_length: int = 200, force_refresh: bool = False, verbose: bool = False):
    """Run an interactive session with the agentic RAG system."""
    # Create vector store with custom parameters
    vector_store = SupabaseVectorStoreManager(min_content_length=min_content_length, force_refresh=force_refresh)
    manager = AgentManager(vector_store=vector_store)
    
    print("Welcome to the Agentic RAG System")
    print(f"Using minimum content length: {min_content_length} characters")
    print(f"Force refresh: {force_refresh}")
    print(f"Verbose mode: {verbose}")
    print("\nAvailable commands:")
    print("- 'exit', 'quit', 'q': Exit the session")
    print("- 'refresh': Refresh the Supabase connection")
    print("- 'stats': View table statistics")
    print("-" * 40)
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting session. Goodbye!")
            break
            
        if query.lower() == "refresh":
            print("Refreshing Supabase connection...")
            vector_store.refresh_connection()
            print("Connection refreshed!")
            continue
            
        if query.lower() == "stats":
            print("Fetching table statistics...")
            vector_store.log_table_statistics()
            continue
        
        print("\nProcessing query...")
        try:
            # Extract keywords for context
            keywords = extract_keywords(query)
            logger.info(f"Extracted keywords: {keywords}")
            
            # Display full debug information in verbose mode
            logger.info(f"Processing user query: {query}")
            
            # Process the query
            response = await manager.process_query(query)
            
            # Display the search plan if verbose mode is enabled
            if verbose and "search_plan" in response:
                display_search_plan(response["search_plan"])
            
            # Display search quality metrics
            if "search_quality" in response:
                display_search_quality(response["search_quality"])
            
            # Display query execution trace if verbose
            if verbose and "trace" in response:
                trace = response["trace"]
                print("\n" + "=" * 40)
                print("QUERY EXECUTION TRACE:")
                print(f"Original query: {trace.get('original_query', 'N/A')}")
                if "search_plan" in trace:
                    plan = trace["search_plan"]
                    print(f"Reformulated query: {plan.get('reformulated_query', 'N/A')}")
                print(f"Sources retrieved: {trace.get('num_sources_retrieved', 0)}")
                print(f"Avg. source length: {int(trace.get('avg_source_length', 0))} characters")
                print("-" * 40)
            
            # Display the final response
            print("\n" + "=" * 40)
            print("FINAL RESPONSE:")
            print(response["final_response"])
            print("\n" + "-" * 40)
            
            # Display sources if available
            if response["sources"]:
                print("SOURCES:")
                for i, source in enumerate(response["sources"]):
                    content = source['content']
                    content_preview = content[:150] + "..." if len(content) > 150 else content
                    
                    # Highlight key concepts in preview
                    if "search_plan" in response:
                        key_concepts = response["search_plan"].get("key_concepts", [])
                        for concept in key_concepts:
                            if concept.lower() in content_preview.lower():
                                pattern = re.compile(re.escape(concept), re.IGNORECASE)
                                content_preview = pattern.sub(f"**{concept.upper()}**", content_preview)
                    
                    # Also highlight keywords in preview
                    for keyword in keywords:
                        if keyword in content_preview.lower() and f"**{keyword.upper()}**" not in content_preview.upper():
                            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                            content_preview = pattern.sub(f"**{keyword.upper()}**", content_preview)
                    
                    print(f"Source {i+1}:")
                    print(f"Content length: {len(content)} characters")
                    print(f"Content: {content_preview}")
                    
                    # Show more detailed metadata
                    if source['metadata']:
                        file_id = source['metadata'].get('file_id', 'Unknown')
                        line_range = "Unknown"
                        if 'loc' in source['metadata'] and 'lines' in source['metadata']['loc']:
                            lines = source['metadata']['loc']['lines']
                            line_range = f"{lines.get('from', '?')}-{lines.get('to', '?')}"
                        
                        print(f"File: {file_id}")
                        print(f"Lines: {line_range}")
                    print()
            else:
                print("No relevant sources found.")
                
            print("=" * 40)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"Error processing query: {str(e)}")


async def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--min-length", type=int, default=200,
                        help="Minimum content length in characters to consider a result valid")
    parser.add_argument("--refresh", action="store_true", 
                        help="Force refresh the Supabase connection")
    parser.add_argument("--query", type=str, 
                        help="Run a single query and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if configuration is valid
    if not config.supabase.url or not config.supabase.key or not config.gemini.api_key:
        logger.error("Missing required environment variables.")
        print("ERROR: Missing required environment variables.")
        print("Please check your .env file or environment variables.")
        print("Required variables: SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY")
        return
    
    # Log configuration (without sensitive data)
    logger.info(f"Embedding dimension: {config.embedding_dimension}")
    logger.info(f"Supabase table: {config.supabase.table_name}")
    logger.info(f"Using Gemini model: {config.gemini.model_name}")
    logger.info(f"Minimum content length: {args.min_length}")
    logger.info(f"Force refresh: {args.refresh}")
    logger.info(f"Verbose mode: {args.verbose}")
    
    # Run a single query if provided
    if args.query:
        response = await process_single_query(
            args.query, 
            min_content_length=args.min_length, 
            force_refresh=args.refresh
        )
        
        # Display the search plan if verbose
        if args.verbose and "search_plan" in response:
            display_search_plan(response["search_plan"])
            
        # Display search quality metrics
        if "search_quality" in response:
            display_search_quality(response["search_quality"])
            
        # Display the result in a formatted way
        print("\n" + "=" * 40)
        print("FINAL RESPONSE:")
        print(response["final_response"])
        print("\n" + "-" * 40)
        
        # Display a summary of sources
        if "sources" in response and response["sources"]:
            sources = response["sources"]
            print(f"Found {len(sources)} sources with average length of "
                 f"{sum(len(s['content']) for s in sources) / len(sources):.0f} characters.")
        else:
            print("No sources found.")
            
        print("=" * 40)
        return
    
    # Run interactive session
    await interactive_session(
        min_content_length=args.min_length,
        force_refresh=args.refresh,
        verbose=args.verbose
    )


if __name__ == "__main__":
    asyncio.run(main()) 