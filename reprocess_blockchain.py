#!/usr/bin/env python
# reprocess_blockchain.py
# Script to reprocess an existing document with blockchain-specific entity types

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from lightrag.lightrag import LightRAG
from lightrag.base import DocStatus, DocProcessingStatus
from lightrag.llm.gemini import gemini_model_complete, gemini_embed, configure_gemini

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("LLM_BINDING_API_KEY")
configure_gemini(api_key)

async def embedding_func(texts):
    """Embedding function using Gemini"""
    model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    return await gemini_embed(texts, model)

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function using Gemini"""
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash-lite")
    return await gemini_model_complete(
        prompt, 
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

async def reprocess_document():
    """Reprocess the existing document with blockchain-specific entity types"""
    print("Initializing LightRAG...")
    
    # Initialize LightRAG with the same configuration
    working_dir = os.getenv("WORKING_DIR", "./working_dir")
    
    # Custom blockchain entity types
    blockchain_entity_types = ["blockchain", "company", "technology", "protocol", "token", "person", "concept", "regulation", "organization"]
    
    # Create LightRAG instance with custom addon parameters for blockchain
    lightrag = LightRAG(
        working_dir=working_dir,
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        embedding_func=embedding_func,
        llm_model_func=llm_model_func,
        addon_params={
            "language": os.getenv("SUMMARY_LANGUAGE", "English"),
            "entity_types": blockchain_entity_types,  # Use blockchain-specific entity types
            "disable_examples": True,  # Disable examples to prevent hallucination
        }
    )
    
    # First, clean up any existing entities by removing and re-adding the document
    doc_statuses = await lightrag.get_docs_by_status(DocStatus.PROCESSED)
    
    if not doc_statuses:
        print("No processed documents found!")
        return
    
    # Need to wipe out previous extracted entities
    print("Cleaning up previous graph data...")
    
    # Backup original file
    try:
        import shutil
        graph_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            backup_file = os.path.join(working_dir, "graph_chunk_entity_relation.backup.graphml")
            shutil.copy2(graph_file, backup_file)
            print(f"Backed up existing graph to {backup_file}")
    except Exception as e:
        print(f"Warning: Could not backup graph file: {e}")
    
    for doc_id, doc_data in doc_statuses.items():
        print(f"Found document: {doc_id}, chunks: {doc_data.chunks_count}")
        
        # Update document status to PENDING to force reprocessing
        print(f"Setting document {doc_id} to PENDING status...")
        await lightrag.doc_status.upsert({
            doc_id: {
                "status": DocStatus.PENDING,
                "content": doc_data.content,
                "content_summary": doc_data.content_summary,
                "content_length": doc_data.content_length,
                "created_at": doc_data.created_at,
                "updated_at": doc_data.updated_at,
                "file_path": doc_data.file_path,
                "chunks_count": doc_data.chunks_count
            }
        })
    
    # Reprocess documents with blockchain-specific entity types
    print("Starting document reprocessing with blockchain-specific entity types...")
    print("This process may take a while due to rate limiting with Gemini's free tier.")
    print("Check the logs for progress updates.")
    
    await lightrag.apipeline_process_enqueue_documents()
    
    print("Reprocessing complete!")
    await lightrag.finalize_storages()

def main():
    """Main function to run the async code"""
    asyncio.run(reprocess_document())

if __name__ == "__main__":
    main() 