"""Configuration module for the agentic RAG system."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class SupabaseConfig(BaseModel):
    """Supabase configuration."""
    url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    key: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY"))
    table_name: str = "documents"  # Using the existing table as specified
    query_name: str = "match_documents"  # Default query name for Supabase

class GeminiConfig(BaseModel):
    """Google Gemini configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    model_name: str = "gemini-2.0-pro-exp"  # As specified in requirements

class Config(BaseModel):
    """Main configuration."""
    supabase: SupabaseConfig = SupabaseConfig()
    gemini: GeminiConfig = GeminiConfig()
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    )

config = Config() 