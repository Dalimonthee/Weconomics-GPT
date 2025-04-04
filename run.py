#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive runner for the PDF RAG System.

This script provides a simple text-based menu to run the PDF RAG System.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    clear_screen()
    print("=" * 50)
    print("PDF RAG System - Interactive Runner")
    print("=" * 50)
    print()


def print_menu():
    """Print the main menu."""
    print("\nPlease select an option:")
    print("1. Process all PDFs in a directory")
    print("2. Process a single PDF file")
    print("3. Query the vector store")
    print("4. View system statistics")
    print("5. Setup and test the system")
    print("0. Exit")
    print()


def process_directory():
    """Process all PDFs in a directory."""
    print_header()
    print("Process all PDFs in a directory")
    print("-" * 50)
    
    # Get directory path
    default_dir = "data/books"
    directory = input(f"Enter directory path [{default_dir}]: ").strip()
    if not directory:
        directory = default_dir
    
    # Create directory if it doesn't exist
    if not Path(directory).exists():
        create = input(f"Directory {directory} doesn't exist. Create it? (y/n) ").strip().lower()
        if create == 'y':
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print("Operation canceled.")
            input("Press Enter to continue...")
            return
    
    # Get collection name
    collection = input("Enter collection name (leave empty for default): ").strip()
    
    # Build command
    cmd = ["python", "main.py", "process-dir", directory]
    if collection:
        cmd.extend(["--collection", collection])
    
    # Execute command
    print("\nProcessing PDFs...")
    subprocess.run(cmd)
    
    input("\nPress Enter to continue...")


def process_file():
    """Process a single PDF file."""
    print_header()
    print("Process a single PDF file")
    print("-" * 50)
    
    # Get file path
    file_path = input("Enter PDF file path: ").strip()
    if not file_path:
        print("No file path provided. Operation canceled.")
        input("Press Enter to continue...")
        return
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File {file_path} doesn't exist.")
        input("Press Enter to continue...")
        return
    
    # Get collection name
    collection = input("Enter collection name (leave empty for default): ").strip()
    
    # Build command
    cmd = ["python", "main.py", "process-file", file_path]
    if collection:
        cmd.extend(["--collection", collection])
    
    # Execute command
    print("\nProcessing PDF...")
    subprocess.run(cmd)
    
    input("\nPress Enter to continue...")


def query_store():
    """Query the vector store."""
    print_header()
    print("Query the vector store")
    print("-" * 50)
    
    # Get query
    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided. Operation canceled.")
        input("Press Enter to continue...")
        return
    
    # Get limit
    limit = input("Enter maximum number of results [5]: ").strip()
    if not limit:
        limit = "5"
    
    # Get collection name
    collection = input("Enter collection name (leave empty for default): ").strip()
    
    # Build command
    cmd = ["python", "main.py", "query", query, "--limit", limit]
    if collection:
        cmd.extend(["--collection", collection])
    
    # Execute command
    print("\nQuerying...")
    subprocess.run(cmd)
    
    input("\nPress Enter to continue...")


def view_stats():
    """View system statistics."""
    print_header()
    print("System Statistics")
    print("-" * 50)
    
    # Get collection name
    collection = input("Enter collection name (leave empty for default): ").strip()
    
    # Build command
    cmd = ["python", "main.py", "stats"]
    if collection:
        cmd.extend(["--collection", collection])
    
    # Execute command
    print("\nFetching statistics...")
    subprocess.run(cmd)
    
    input("\nPress Enter to continue...")


def setup_system():
    """Setup and test the system."""
    print_header()
    print("Setup and Test")
    print("-" * 50)
    
    # Execute command
    print("Running setup script...")
    subprocess.run(["python", "setup.py"])
    
    input("\nPress Enter to continue...")


def main():
    """Main function for the interactive runner."""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == '1':
                process_directory()
            elif choice == '2':
                process_file()
            elif choice == '3':
                query_store()
            elif choice == '4':
                view_stats()
            elif choice == '5':
                setup_system()
            elif choice == '0':
                print("\nExiting...")
                time.sleep(1)
                break
            else:
                print("\nInvalid choice. Please try again.")
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nOperation canceled.")
            time.sleep(1)
        except Exception as e:
            print(f"\nError: {str(e)}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main() 