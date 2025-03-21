"""Streamlit interface for the Blockchain RAG System."""

import os
import sys
import asyncio
import streamlit as st
from typing import Dict, Any, List, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables from Streamlit secrets
if hasattr(st, "secrets") and "env" in st.secrets:
    for key, value in st.secrets.env.items():
        os.environ[key] = str(value)

# Import our RAG components
from src.agents import AgentManager
from src.vector_store import SupabaseVectorStoreManager
from src.config import config

# Configure the page - use simpler configuration to avoid JS errors
st.set_page_config(
    page_title="Blockchain Knowledge Assistant",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/blockchain-rag',
        'Report a bug': 'https://github.com/yourusername/blockchain-rag/issues',
        'About': 'Blockchain Knowledge Assistant powered by RAG technology.'
    }
)

# Simpler CSS styling without complex selectors
st.markdown("""
<style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        font-size: 16px;
    }
    h1 {
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.title("🔗 Blockchain Knowledge Assistant")
st.markdown("""
This intelligent assistant uses an advanced RAG (Retrieval-Augmented Generation) system to answer your questions about blockchain technology, cryptocurrencies, and related topics.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Min content length slider
    min_length = st.slider(
        "Minimum content length",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Minimum character length for retrieved content chunks"
    )
    
    # Refresh option
    force_refresh = st.checkbox(
        "Force refresh connection",
        value=False,
        help="Force refresh the Supabase connection"
    )
    
    # Verbose option
    verbose = st.checkbox(
        "Verbose mode",
        value=True,
        help="Show detailed information about the search process"
    )
    
    # Environment variable check
    with st.expander("Environment Status"):
        supabase_url = "✅ Set" if config.supabase.url else "❌ Missing"
        supabase_key = "✅ Set" if config.supabase.key else "❌ Missing"
        google_api_key = "✅ Set" if config.gemini.api_key else "❌ Missing"
        embedding_dim = config.embedding_dimension
        
        st.markdown("### Environment Variables")
        st.markdown(f"- SUPABASE_URL: {supabase_url}")
        st.markdown(f"- SUPABASE_KEY: {supabase_key}")
        st.markdown(f"- GOOGLE_API_KEY: {google_api_key}")
        st.markdown(f"- EMBEDDING_DIMENSION: {embedding_dim}")
        
        if not (config.supabase.url and config.supabase.key and config.gemini.api_key):
            st.error("Some required environment variables are missing. Please check your configuration.")
    
    # Show Supabase stats - use st.button with key to avoid conflicts
    if st.button("Show Database Statistics", key="db_stats"):
        try:
            with st.spinner("Fetching database statistics..."):
                vector_store = SupabaseVectorStoreManager(
                    min_content_length=min_length,
                    force_refresh=True
                )
                stats = vector_store.log_table_statistics()
                
                if "total_records" in stats:
                    st.success(f"Total records: {stats['total_records']}")
                
                if "avg_content_length" in stats:
                    st.info(f"Average content length: {stats['avg_content_length']:.1f} characters")
                    
                if "size_percentages" in stats:
                    st.subheader("Document Size Distribution")
                    for range_name, percentage in stats["size_percentages"].items():
                        st.write(f"{range_name}: {percentage:.1f}%")
        except Exception as e:
            st.error(f"Error fetching stats: {str(e)}")
    
    st.markdown("---")
    st.caption("Powered by Google Gemini + Supabase Vector Store")


# Process query function with asyncio - with error handling
async def process_query(query: str, min_length: int, force_refresh: bool) -> Dict[str, Any]:
    """Process a query through the RAG system."""
    try:
        # Check environment variables first
        if not config.supabase.url or not config.supabase.key or not config.gemini.api_key:
            missing_vars = []
            if not config.supabase.url:
                missing_vars.append("SUPABASE_URL")
            if not config.supabase.key:
                missing_vars.append("SUPABASE_KEY")
            if not config.gemini.api_key:
                missing_vars.append("GOOGLE_API_KEY")
                
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            return {
                "final_response": f"Error: {error_msg}. Please configure these in your Streamlit cloud settings.",
                "sources": [],
                "search_plan": {},
                "search_quality": {}
            }
        
        vector_store = SupabaseVectorStoreManager(min_content_length=min_length, force_refresh=force_refresh)
        manager = AgentManager(vector_store=vector_store)
        result = await manager.process_query(query)
        return result
    except Exception as e:
        st.error(f"Error in query processing: {str(e)}")
        return {
            "final_response": f"Error processing query: {str(e)}",
            "sources": [],
            "search_plan": {},
            "search_quality": {}
        }


# Function to run async code in Streamlit with proper error handling
def run_async_query(query: str, min_length: int, force_refresh: bool) -> Dict[str, Any]:
    """Run the async query function and return the result."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_query(query, min_length, force_refresh))
        loop.close()
        return result
    except TypeError as e:
        if "AIMessage" in str(e):
            st.error("Error with LLM response format. This is likely due to a change in the LangChain or Gemini API.")
            return {
                "final_response": "Error: The LLM returned a response in an unexpected format. The system has been updated to handle this, please try again.",
                "sources": [],
                "search_plan": {},
                "search_quality": {}
            }
        else:
            st.error(f"Type error: {str(e)}")
            return {
                "final_response": f"Error: {str(e)}",
                "sources": [],
                "search_plan": {},
                "search_quality": {}
            }
    except Exception as e:
        st.error(f"Error running query: {str(e)}")
        return {
            "final_response": f"Error running query: {str(e)}",
            "sources": [],
            "search_plan": {},
            "search_quality": {}
        }


# Function to display sources in a nice format
def display_sources(sources: List[Dict[str, Any]], keywords: List[str]):
    """Display sources in a formatted way."""
    if not sources:
        st.info("No relevant sources found.")
        return
    
    st.subheader(f"📚 Sources ({len(sources)})")
    
    for i, source in enumerate(sources):
        content = source.get('content', '')
        metadata = source.get('metadata', {})
        
        with st.expander(f"Source {i+1} ({len(content)} chars)"):
            st.markdown(f"**Content:**\n\n{content}")
            
            if metadata:
                st.markdown("**Metadata:**")
                for key, value in metadata.items():
                    st.markdown(f"- **{key}**: {value}")


# Function to display search plan
def display_search_plan(search_plan: Dict[str, Any]):
    """Display the search plan information."""
    if not search_plan:
        return
    
    st.subheader("🔍 Search Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reformulated Query:**")
        st.info(search_plan.get('reformulated_query', 'N/A'))
        
        st.markdown("**Key Concepts:**")
        concepts = search_plan.get('key_concepts', [])
        if concepts:
            for concept in concepts:
                st.markdown(f"- {concept}")
        else:
            st.markdown("No key concepts identified.")
    
    with col2:
        st.markdown("**Focus Areas:**")
        areas = search_plan.get('focus_areas', [])
        if areas:
            for area in areas:
                st.markdown(f"- {area}")
        else:
            st.markdown("No focus areas identified.")
        
        st.markdown("**Guidance for Agent:**")
        guidance = search_plan.get('guidance', '')
        if guidance:
            st.markdown(f"_{guidance}_")
        else:
            st.markdown("No specific guidance provided.")


# Function to display search quality metrics
def display_search_quality(search_quality: Dict[str, Any]):
    """Display search quality metrics."""
    if not search_quality:
        return
    
    st.subheader("📊 Search Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sources Found", search_quality.get('found_documents', 0))
        st.metric("Key Concept Coverage", f"{search_quality.get('key_concept_coverage', 0)}%")
    
    with col2:
        st.metric("Avg Document Length", f"{int(search_quality.get('avg_document_length', 0))} chars")
        st.metric("Quality Assessment", search_quality.get('assessment', 'Unknown'))
    
    if 'concept_coverage' in search_quality:
        st.markdown("**Concept Coverage Details:**")
        for concept, count in search_quality['concept_coverage'].items():
            st.markdown(f"- '{concept}': found in {count} sources")


# Main query input - with key to avoid conflicts
query = st.text_area(
    "Enter your blockchain question:",
    placeholder="Example: What is a blockchain consensus mechanism?",
    height=100,
    key="query_input"
)

# Process query button - with unique key
if st.button("Get Answer", type="primary", key="submit_button"):
    if not query:
        st.warning("Please enter a question.")
    else:
        # Show processing message
        with st.spinner("🧠 Processing your question..."):
            try:
                # Process the query
                result = run_async_query(query, min_length, force_refresh)
                
                # Display the answer
                st.markdown("## 🤖 Answer")
                st.markdown(result.get("final_response", "No response generated."))
                
                # Horizontal rule
                st.markdown("---")
                
                # Create tabs for the detailed information
                if verbose:
                    tab1, tab2, tab3 = st.tabs(["Search Strategy", "Search Quality", "Sources"])
                    
                    with tab1:
                        if "search_plan" in result:
                            display_search_plan(result["search_plan"])
                    
                    with tab2:
                        if "search_quality" in result:
                            display_search_quality(result["search_quality"])
                    
                    with tab3:
                        if "sources" in result:
                            keywords = []
                            if "search_plan" in result and "key_concepts" in result["search_plan"]:
                                keywords = result["search_plan"]["key_concepts"]
                            display_sources(result["sources"], keywords)
                else:
                    # Just show the sources in compact mode
                    if "sources" in result:
                        keywords = []
                        if "search_plan" in result and "key_concepts" in result["search_plan"]:
                            keywords = result["search_plan"]["key_concepts"]
                        display_sources(result["sources"], keywords)
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.write("Please check your configuration and try again.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, Google Gemini, and Supabase Vector Store.") 