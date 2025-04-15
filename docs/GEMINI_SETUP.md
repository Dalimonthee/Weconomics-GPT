# Setting up Google Gemini with LightRAG

This guide explains how to configure LightRAG to use Google's Gemini models for text generation and embeddings.

## Prerequisites

1. A Google Cloud account with access to the Gemini API
2. An API key for Gemini ([Get an API key](https://ai.google.dev/tutorials/setup))
3. LightRAG installed on your system

## Installation

Make sure you have the Google Generative AI package installed:

```bash
pip install google-generativeai>=0.3.0
```

This package is already included in the LightRAG requirements.txt file in recent versions.

## Configuration

### Using Environment Variables

The simplest way to configure Gemini is using environment variables. You can create a `.env` file in your project directory with the following settings:

```bash
# Google Gemini LLM Configuration
LLM_BINDING=gemini
LLM_MODEL=gemini-1.5-pro
LLM_BINDING_API_KEY=your_gemini_api_key

# Google Gemini Embedding Configuration
EMBEDDING_BINDING=gemini
EMBEDDING_MODEL=models/embedding-001
EMBEDDING_DIM=768
EMBEDDING_BINDING_API_KEY=your_gemini_api_key
```

Alternatively, you can copy the included `.env.gemini.example` file and modify it:

```bash
cp .env.gemini.example .env
# Edit the .env file to set your API key
```

### Available Gemini Models

For LLM functionality:
- `gemini-1.0-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

For embedding:
- `models/embedding-001`

## Starting LightRAG with Gemini

### Using the API Server

To start LightRAG API server with Gemini:

```bash
python -m lightrag.api.lightrag_server --llm-binding gemini --embedding-binding gemini --llm-model gemini-1.5-pro --embedding-model models/embedding-001 --llm-binding-api-key YOUR_API_KEY
```

Or simply run it with environment variables set in your `.env` file:

```bash
python -m lightrag.api.lightrag_server
```

### Using LightRAG in Code

Here's a basic example of initializing LightRAG with Gemini:

```python
import os
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import EmbeddingFunc

# Set your API key
GEMINI_API_KEY = "your_api_key_here"

# Define LLM function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    kwargs["api_key"] = GEMINI_API_KEY
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

# Define embedding function
async def embedding_func(texts):
    return await gemini_embed(
        texts,
        embed_model="models/embedding-001",
        api_key=GEMINI_API_KEY,
    )

# Initialize LightRAG
rag = LightRAG(
    working_dir="./index_gemini",
    llm_model_func=llm_model_func,
    llm_model_name="gemini-1.5-pro",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,  # Dimension for Gemini embeddings
        max_token_size=8192,
        func=embedding_func,
    ),
)
```

For a full working example, see the `examples/lightrag_gemini_demo.py` file.

## Troubleshooting

### API Key Issues

If you encounter authentication errors, ensure your API key:
- Is correctly set in your environment variables or config
- Has access to the Gemini API
- Hasn't expired or reached quota limits

### Model Availability

Google may have different model availability based on your region or account tier. If you encounter model not found errors, try using a different model version.

### Embedding Dimensions

Gemini embeddings have 768 dimensions. Make sure your `embedding_dim` parameter matches this value for the `EmbeddingFunc`. 