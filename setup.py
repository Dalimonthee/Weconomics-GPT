#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for PDF RAG System.

This script helps users set up the PDF RAG System by:
1. Installing required dependencies
2. Testing the connection to Supabase
3. Testing the connection to Google's Gemini API
4. Verifying the system is ready to use
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Try to import required modules
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    missing = [var for var, value in required_vars.items() if not value]
    
    if missing:
        print("❌ Missing environment variables in .env file:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease create a .env file with the following variables:")
        print("""
# Supabase credentials
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Google Gemini API credentials
GOOGLE_API_KEY=your_google_api_key

# Vector Collection Name (optional)
COLLECTION_NAME=book_pages
        """)
        return False
    
    print("✅ Environment variables are set")
    return True


def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies...")
    
    # Check if requirements.txt exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def test_supabase_connection():
    """Test the connection to Supabase."""
    print("Testing Supabase connection...")
    
    try:
        from supabase import create_client
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ Supabase credentials not found in .env file")
            return False
        
        # Create the Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # Test a simple query
        response = supabase.table("vector_collections").select("*").limit(1).execute()
        
        print("✅ Supabase connection successful")
        print(f"  Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        return False


def test_gemini_connection():
    """Test the connection to Google's Gemini API."""
    print("Testing Google Gemini API connection...")
    
    try:
        import google.generativeai as genai
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("❌ Google API key not found in .env file")
            return False
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Generate a test embedding
        test_text = "This is a test to check if the Gemini API is working."
        embedding = embedding_model.embed_query(test_text)
        
        print("✅ Google Gemini API connection successful")
        print(f"  Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"❌ Google Gemini API connection failed: {str(e)}")
        return False


def test_pdf_processing():
    """Test the PDF processing functionality."""
    print("Testing PDF processing...")
    
    try:
        import pypdfium2 as pdfium
        
        # Check if the data directory exists
        data_dir = Path("data/books")
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {data_dir}")
        
        # Create a simple test PDF if none exists
        test_files = list(data_dir.glob("*.pdf"))
        if not test_files:
            print("ℹ️ No PDF files found in the data directory")
            print("ℹ️ You can add PDF files to the data/books directory later")
            
            # Try to create a sample PDF for testing
            try:
                from fpdf import FPDF
                
                print("Creating a sample PDF for testing...")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="This is a test PDF file", ln=True)
                pdf.cell(200, 10, txt="Created by the PDF RAG System setup script", ln=True)
                pdf.cell(200, 10, txt="Page 1", ln=True)
                
                pdf.add_page()
                pdf.cell(200, 10, txt="This is page 2 of the test PDF file", ln=True)
                pdf.cell(200, 10, txt="It contains additional text for testing", ln=True)
                
                sample_path = data_dir / "sample_test.pdf"
                pdf.output(str(sample_path))
                
                print(f"✅ Created sample PDF at: {sample_path}")
                
                # Test opening with pdfium
                pdf = pdfium.PdfDocument(sample_path)
                page_count = len(pdf)
                text = pdf[0].get_textpage().get_text()
                
                print(f"✅ PDF processing successful: {page_count} pages found")
                print(f"  First page text: {text[:50]}...")
                return True
            except ImportError:
                print("ℹ️ FPDF not installed, skipping sample PDF creation")
                return True
        else:
            # Test with an existing PDF
            test_file = test_files[0]
            try:
                pdf = pdfium.PdfDocument(test_file)
                page_count = len(pdf)
                text = pdf[0].get_textpage().get_text()
                
                print(f"✅ PDF processing successful: {page_count} pages found in {test_file.name}")
                print(f"  First page text: {text[:50]}...")
                return True
            except Exception as e:
                print(f"❌ PDF processing failed: {str(e)}")
                return False
    except Exception as e:
        print(f"❌ PDF processing setup failed: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("=" * 50)
    print("PDF RAG System Setup")
    print("=" * 50)
    
    # Show system information
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print("-" * 50)
    
    # Check environment variables
    env_ok = check_environment()
    print("-" * 50)
    
    # Install dependencies
    deps_ok = install_dependencies()
    print("-" * 50)
    
    if not env_ok or not deps_ok:
        print("❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    # Test Supabase connection
    supabase_ok = test_supabase_connection()
    print("-" * 50)
    
    # Test Gemini API connection
    gemini_ok = test_gemini_connection()
    print("-" * 50)
    
    # Test PDF processing
    pdf_ok = test_pdf_processing()
    print("-" * 50)
    
    # Summary
    print("Setup Summary:")
    print(f"Environment variables: {'✅' if env_ok else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"Supabase connection: {'✅' if supabase_ok else '❌'}")
    print(f"Gemini API connection: {'✅' if gemini_ok else '❌'}")
    print(f"PDF processing: {'✅' if pdf_ok else '❌'}")
    
    if env_ok and deps_ok and supabase_ok and gemini_ok and pdf_ok:
        print("\n✅ Setup completed successfully! Your system is ready to use.")
        print("\nYou can now run:")
        print("  python main.py process-dir data/books")
        print("  python main.py query \"Your search query here\"")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")


if __name__ == "__main__":
    main() 