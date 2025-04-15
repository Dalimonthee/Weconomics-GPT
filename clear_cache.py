#!/usr/bin/env python
"""
Clear the LLM response cache to ensure fresh processing.
This helps eliminate any cached hallucinated examples and enables new direct entity queries.
"""

import os
import json
import shutil
from pathlib import Path
import glob

def clear_llm_cache():
    print("Clearing LLM cache...")
    working_dir = os.environ.get("WORKING_DIR", "./working_dir")
    
    # List of cache files to clear
    cache_files = [
        "kv_store_llm_response_cache.json",  # Main LLM cache
        "kv_store_query_cache.json",         # Query results cache
        "kv_store_keywords_cache.json"       # Keywords extraction cache
    ]
    
    for cache_file in cache_files:
        filepath = os.path.join(working_dir, cache_file)
        
        if os.path.exists(filepath):
            # Create a backup first
            backup_file = os.path.join(working_dir, f"{cache_file}.backup.json")
            try:
                shutil.copy2(filepath, backup_file)
                print(f"Created backup of {cache_file}: {backup_file}")
            except Exception as e:
                print(f"Warning: Could not create backup of {cache_file}: {e}")
            
            # Create empty cache file
            with open(filepath, 'w') as f:
                f.write('{}')
            print(f"Successfully cleared cache: {filepath}")
        else:
            print(f"No cache file found at {filepath}")
    
    # Look for any additional cache files
    additional_caches = glob.glob(os.path.join(working_dir, "kv_store_*.json"))
    for cache in additional_caches:
        if not any(file in cache for file in cache_files) and not cache.endswith(".backup.json"):
            print(f"Found additional cache: {cache}")
            backup = f"{cache}.backup"
            try:
                shutil.copy2(cache, backup)
                print(f"  Created backup: {backup}")
                with open(cache, 'w') as f:
                    f.write('{}')
                print(f"  Cleared cache: {cache}")
            except Exception as e:
                print(f"  Warning: Could not clear cache {cache}: {e}")
    
    print("All caches cleared successfully!")

if __name__ == "__main__":
    clear_llm_cache() 