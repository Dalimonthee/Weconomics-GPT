"""
LightRAG with Google Gemini integration example.

This example demonstrates how to use Google Gemini models with LightRAG
for both text generation and embeddings.

Requirements:
- A Google API key with access to Gemini models
- The google-generativeai package installed

Usage:
1. Set your GOOGLE_API_KEY environment variable
2. Run this script: python lightrag_gemini_demo.py
"""

import os
import sys
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested asyncio event loops (for Jupyter/interactive use)
nest_asyncio.apply()

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables
load_dotenv()

# Configure working directory
WORKING_DIR = "./index_gemini"
print(f"WORKING_DIR: {WORKING_DIR}")

# Make sure the Google API key is set
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable is not set.")
    print("Please set it to your Google API key with access to Gemini models.")
    sys.exit(1)

# Model configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-1.5-pro")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/embedding-001")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_DIM = 768  # Dimension for Gemini embeddings
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)


# Define LLM function for Gemini
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        kwargs["api_key"] = GOOGLE_API_KEY
        response = await gemini_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return response
    except Exception as e:
        print(f"LLM request failed: {str(e)}")
        raise


# Define embedding function for Gemini
async def embedding_func(texts):
    try:
        embeddings = await gemini_embed(
            texts,
            embed_model=EMBEDDING_MODEL,
            api_key=GOOGLE_API_KEY,
        )
        return embeddings
    except Exception as e:
        print(f"Embedding failed: {str(e)}")
        raise


async def initialize_rag():
    # Initialize LightRAG with Gemini
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )

    # Initialize storages
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Check if the sample file exists
    sample_file = "./book.txt"
    if not os.path.exists(sample_file):
        print(f"Sample file {sample_file} not found.")
        print("Creating a simple sample text for demonstration.")
        
        # Create a simple sample text
        sample_text = """
        # The Benefits of Artificial Intelligence

        Artificial Intelligence (AI) has transformed many aspects of modern life. From healthcare to finance, 
        from education to entertainment, AI technologies are creating new possibilities and efficiencies.

        ## Healthcare Applications

        In healthcare, AI helps with diagnosis, drug discovery, and personalized treatment plans. 
        Machine learning algorithms can analyze medical images to detect diseases earlier than human doctors might.
        Natural language processing can extract relevant information from medical literature to assist researchers.

        ## Business and Finance

        Financial institutions use AI for fraud detection, algorithmic trading, and customer service.
        Chatbots and virtual assistants help customers with basic questions and transactions.
        Predictive analytics help businesses forecast trends and make data-driven decisions.

        ## Everyday Life

        Smart assistants like Siri, Alexa, and Google Assistant have become part of many people's daily routines.
        Recommendation systems suggest content we might enjoy on streaming platforms and shopping sites.
        AI-powered translation tools break down language barriers in global communication.

        ## Challenges and Considerations

        Despite its benefits, AI raises important ethical questions about privacy, bias, and accountability.
        As AI systems become more powerful, ensuring they operate fairly and transparently is crucial.
        """
        
        with open("sample_text.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        sample_file = "sample_text.txt"
        print(f"Created sample file: {sample_file}")

    # Insert document text
    with open(sample_file, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"Inserting content from {sample_file} ({len(content)} characters)")
        rag.insert(content)

    # Test queries with different modes
    test_query = "What are the main applications of AI in healthcare?"
    
    print("\nNaive Search:")
    print(rag.query(test_query, param=QueryParam(mode="naive")))

    print("\nLocal Search:")
    print(rag.query(test_query, param=QueryParam(mode="local")))

    print("\nGlobal Search:")
    print(rag.query(test_query, param=QueryParam(mode="global")))

    print("\nHybrid Search:")
    print(rag.query(test_query, param=QueryParam(mode="hybrid")))


if __name__ == "__main__":
    main()
