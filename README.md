# PDF RAG System

A Retrieval-Augmented Generation (RAG) system for processing PDF books, chunking them by page, and storing them in a vector database for semantic search.

## Features

- **PDF Processing**: Uses pdfium to convert PDF files to text with high accuracy
- **Page-level Chunking**: Each page of a book becomes a semantic chunk with metadata
- **Vector Storage**: Stores chunks in Supabase Vector database with pgvector
- **Semantic Search**: Generate embeddings using Google's Gemini embedding model
- **Command-line Interface**: Easy to use CLI for processing PDFs and querying the system

## Requirements

- Python 3.8+
- Supabase account with pgvector extension enabled
- Google AI (Gemini) API key

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:

```
# Supabase credentials
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Google Gemini API credentials
GOOGLE_API_KEY=your_google_api_key

# Vector Collection Name
COLLECTION_NAME=book_pages
```

## Usage

### Process a Directory of PDFs

To process all PDF files in a directory:

```bash
python main.py process-dir path/to/pdf/directory
```

You can specify a custom collection name:

```bash
python main.py process-dir path/to/pdf/directory --collection my_custom_collection
```

### Process a Single PDF File

To process a single PDF file:

```bash
python main.py process-file path/to/file.pdf
```

### Query the Vector Store

To query the vector store for similar content:

```bash
python main.py query "Your search query here"
```

You can specify the number of results to return:

```bash
python main.py query "Your search query here" --limit 10
```

### Get Statistics

To see statistics about the vector store:

```bash
python main.py stats
```

## How It Works

### PDF Processing

1. The system uses pdfium to extract text from PDF files
2. Each page is processed individually, preserving page boundaries
3. Metadata (book name, page number) is attached to each page

### Vector Storage

1. Google's Gemini embedding model generates embeddings for each page
2. Pages are stored in Supabase with their text content, metadata, and embeddings
3. pgvector enables semantic similarity search on these embeddings

### Querying

1. User submits a query
2. System generates an embedding for the query using the same model
3. pgvector finds the most similar pages based on cosine similarity
4. Results are returned with similarity scores and content previews

## Supabase Setup

To use this system, you need a Supabase instance with pgvector enabled. The system will automatically:

1. Create a table for storing vector embeddings if it doesn't exist
2. Create the necessary indexes for similarity search
3. Store metadata alongside the embeddings

## Troubleshooting

- If you encounter errors with pdfium, make sure you have the latest version installed
- For Supabase connection issues, verify your credentials in the `.env` file
- If embeddings aren't generating, check your Google API key and quota

## License

MIT 