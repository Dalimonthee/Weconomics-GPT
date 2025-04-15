#!/usr/bin/env python
"""
Reprocess documents for entity extraction with improved generalized entity types.
This script will clear the entity extraction cache and reprocess
a specified document to extract entities properly.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint

# Load environment variables
load_dotenv()

# Ensure we're running with Python 3.9+
if sys.version_info < (3, 9):
    print("This script requires Python 3.9 or higher.")
    sys.exit(1)

async def embedding_func(texts):
    """Embedding function that uses Google Gemini API"""
    try:
        from lightrag.llm.gemini import gemini_embed
        return await gemini_embed(texts)
    except Exception as e:
        logging.error(f"Error in embedding function: {e}")
        raise

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function that uses Google Gemini API"""
    try:
        from lightrag.llm.gemini import gemini_model_complete
        return await gemini_model_complete(prompt, system_prompt, history_messages, **kwargs)
    except Exception as e:
        logging.error(f"Error in LLM function: {e}")
        raise

async def reprocess_document():
    """Reprocess a document to extract entities with improved settings"""
    from lightrag.lightrag import LightRAG
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    
    # Specify our generalized entity types
    generalized_entity_types = [
        "Technology", "Management", "Organization", "Component", "Concept", 
        "Function", "System", "Protocol", "Infrastructure", "Platform", 
        "Standard", "Algorithm", "Business Model", "Process", "Mechanism", 
        "Stakeholder", "Policy", "Person", "Location"
    ]
    
    # Initialize LightRAG with our configurations
    rag = LightRAG(
        working_dir=os.environ.get("WORKING_DIR", "./working_dir"),
        embedding_func=embedding_func,
        llm_model_func=llm_model_func,
        enable_llm_cache=True,
        entity_extract_max_gleaning=2,  # Increase gleaning attempts for better extraction
        # Disable entity extraction to avoid rate limits - false means enabled
        DISABLE_ENTITY_EXTRACTION=False,
        addon_params={
            "language": "English",
            "entity_types": generalized_entity_types,
            "disable_examples": True,  # Disable examples to avoid hallucination
        }
    )
    
    # Clear the cache for entity extraction
    print("Clearing entity extraction cache...")
    await rag.aclear_cache(modes=["default"])
    
    # Get list of documents that need reprocessing
    processed_docs = await rag.get_docs_by_status("processed")
    
    if not processed_docs:
        print("No processed documents found to reprocess.")
        return
    
    # Get the first processed document's ID
    doc_id = next(iter(processed_docs))
    doc = processed_docs[doc_id]
    
    # Show document information
    print(f"Reprocessing document ID: {doc_id}")
    print(f"File: {doc.file_path}")
    print(f"Length: {doc.content_length} characters")
    print(f"Summary: {doc.content_summary}")
    
    # Change document status to "failed" to trigger reprocessing
    from lightrag.schema import DocStatus
    
    await rag.doc_status.upsert({
        doc_id: {
            "status": DocStatus.FAILED,
            "content": doc.content,
            "content_summary": doc.content_summary,
            "content_length": doc.content_length,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "file_path": doc.file_path,
        }
    })
    
    print("Processing document with improved entity extraction...")
    # Process the document
    await rag.apipeline_process_enqueue_documents()
    
    print("Successfully completed document reprocessing.")
    
    # Show final statistics
    print("\nReprocessing complete!")
    processing_status = await rag.get_processing_status()
    print("Document status counts:")
    pprint(processing_status)
    
    # Get updated information
    kgraph_labels = await rag.get_graph_labels()
    print("\nKnowledge graph labels:")
    pprint(kgraph_labels)
    
    # Finalize the knowledge graph database
    print("\nFinalizing storages...")
    await rag.finalize_storages()

def main():
    """Main function to run the script"""
    try:
        # Create asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the reprocessing function
        print(f"Starting document reprocessing with generalized entity extraction...")
        loop.run_until_complete(reprocess_document())
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error during reprocessing: {str(e)}")
    finally:
        # Close the event loop
        if loop and not loop.is_closed():
            loop.close()

if __name__ == "__main__":
    main() 