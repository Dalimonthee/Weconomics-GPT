# Agentic RAG System

An advanced agentic Retrieval-Augmented Generation (RAG) system built with LangChain, Google Gemini, and Supabase vector store.

## Overview

This system implements a hierarchical approach to RAG systems:

1. **Manager LLM**: Interprets user queries, formulates appropriate search queries, and presents final responses
2. **Agent LLM**: Performs the actual RAG operations, querying the vector database for relevant information
3. **RAG System**: Uses Supabase vector store to retrieve documents relevant to queries

## Architecture

- **Supabase Vector Store**: Stores and retrieves document embeddings for semantic search
- **Google Gemini LLMs**: Powers both the manager and agent components
- **LangChain**: Orchestrates the components together

## Requirements

- Python 3.8+
- Supabase account with pgvector extension enabled
- Google Gemini API access

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```
   # Supabase credentials
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   
   # Google Gemini API credentials
   GOOGLE_API_KEY=your_google_api_key
   ```

## Database Setup

This system expects a Supabase table named `documents` with the following structure:

- `id`: UUID (primary key)
- `content`: TEXT (document content)
- `embedding`: VECTOR (document embeddings)
- `metadata`: JSONB (optional metadata)

## Usage

### CLI Mode

Run the interactive session:

```bash
python -m src.main
```

### API Mode

Start the FastAPI server:

```bash
python -m src.api
```

The API will be available at http://localhost:8000 with the following endpoints:

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- `POST /query`: Submit a query to the agentic RAG system

Example API request:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?", "max_sources": 3}'
```

### Programmatic Usage

Import the components into your own application:

```python
from src.agents import AgentManager

async def my_function():
    manager = AgentManager()
    response = await manager.process_query("What is the capital of France?")
    print(response["final_response"])
```

## Architecture Diagram

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│   User Query    │──────►  │  Manager LLM    │──────►  │   Agent LLM     │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └────────┬────────┘
                                     ▲                           │
                                     │                           ▼
                                     │                  ┌─────────────────┐
                                     │                  │                 │
                                     └──────────────────┤  RAG System     │
                                                        │                 │
                                                        └────────┬────────┘
                                                                 │
                                                                 ▼
                                                        ┌─────────────────┐
                                                        │                 │
                                                        │ Supabase Vector │
                                                        │     Store       │
                                                        │                 │
                                                        └─────────────────┘
```

## API Documentation

When running the API server, Swagger UI documentation is available at:
- http://localhost:8000/docs

ReDoc documentation is available at:
- http://localhost:8000/redoc

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 