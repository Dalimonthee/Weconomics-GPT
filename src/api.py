"""API module for the agentic RAG system."""

import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents import AgentManager, UserQuery, RAGResponse, DocumentChunk
from src.config import config


# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for the Agentic RAG System using Google Gemini and Supabase",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="The user's query")
    max_sources: Optional[int] = Field(5, description="Maximum number of sources to return")


class QueryResponse(BaseModel):
    """Query response model."""
    query: str = Field(..., description="The original query")
    manager_interpretation: str = Field(..., description="How the manager interpreted the query")
    rag_response: str = Field(..., description="Raw response from the RAG system")
    final_response: str = Field(..., description="Final formatted response for the user")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used to answer the query")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Agentic RAG API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if configurations are valid
    is_supabase_configured = bool(config.supabase.url and config.supabase.key)
    is_gemini_configured = bool(config.gemini.api_key)
    
    return {
        "status": "healthy",
        "supabase_configured": is_supabase_configured,
        "gemini_configured": is_gemini_configured
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the agentic RAG system."""
    try:
        # Create a new agent manager for each request
        # In a production system, you might want to implement session management
        manager = AgentManager()
        
        # Process the query
        response = await manager.process_query(request.query)
        
        # Limit sources if requested
        if request.max_sources and request.max_sources < len(response["sources"]):
            response["sources"] = response["sources"][:request.max_sources]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def start():
    """Start the API server using uvicorn."""
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    start() 