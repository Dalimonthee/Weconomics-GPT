"""
Google Gemini integration for LightRAG.

This module provides functionality to use Google Gemini models for LLM operations
in LightRAG.
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Union, Optional, AsyncIterator
import random

import pipmaster as pm

# Install required dependencies
if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")

import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import logger, locate_json_string_body_from_string
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

# Global rate limiting for free tier
# This keeps track of request times to enforce spacing
_last_request_time = 0
_request_lock = asyncio.Lock()

def configure_gemini(api_key: str):
    """
    Configure the Gemini API with the provided API key.
    
    Args:
        api_key: Google API key for Gemini
    """
    genai.configure(api_key=api_key)


async def _rate_limit_delay():
    """
    Enforces delay between API calls to respect rate limits.
    Free tier requires significant spacing between requests.
    """
    global _last_request_time
    
    async with _request_lock:
        current_time = time.time()
        elapsed = current_time - _last_request_time
        
        
        min_delay = 1.7
        base_delay = random.uniform(min_delay, 2.3)
        
        if elapsed < base_delay:
            delay = base_delay - elapsed + random.uniform(0.5, 2.0)  # Add jitter
            logger.info(f"Rate limiting: Waiting {delay:.2f}s before next API call")
            await asyncio.sleep(delay)
        
        # Update last request time
        _last_request_time = time.time()


@retry(
    # Limit retry attempts while increasing wait time
    stop=stop_after_attempt(4),
    # Much more aggressive exponential backoff: starts at 10s, doubles each time, max 240s
    wait=wait_exponential(multiplier=10, min=10, max=240),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _gemini_model_if_cache(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    **kwargs
) -> str:
    """
    Complete the prompt using Google Gemini with caching.
    
    Args:
        model_name: Name of the Gemini model to use
        prompt: The prompt to send to the model
        system_prompt: Optional system prompt for context
        history_messages: List of previous messages in the conversation
        **kwargs: Additional parameters

    Returns:
        The model's response as a string
    """
    try:
        # Apply rate limiting before making request
        await _rate_limit_delay()
        
        api_key = kwargs.get("api_key", None)
        if api_key:
            configure_gemini(api_key)
        
        # Create a generation config from kwargs with lower values
        generation_config = {
            "temperature": kwargs.get("temperature", 0.3),  # Lower temperature by default
            "top_p": kwargs.get("top_p", 0.85),  # More deterministic responses
            "top_k": kwargs.get("top_k", 20),  # More focused sampling
            "max_output_tokens": kwargs.get("max_tokens", 4096),  # Reduced tokens
        }
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        # Start a chat session
        chat = model.start_chat(history=[])
        
        # Add system prompt if present
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "parts": [system_prompt]})
        
        # Add history messages
        for msg in history_messages:
            role = "user" if msg["role"] == "user" else "model"
            formatted_messages.append({"role": role, "parts": [msg["content"]]})
        
        # Add current message
        formatted_messages.append({"role": "user", "parts": [prompt]})
        
        # Shorten prompt if it's too long to reduce token usage
        if len(prompt) > 4000:
            logger.info("Truncating prompt to reduce token usage")
            prompt = prompt[:4000]
        
        # Send the conversation to the model
        response = chat.send_message(prompt)
        
        # Return the response
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in _gemini_model_if_cache: {error_msg}")
        
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            # For rate limits, use a much longer sleep to let quota reset
            wait_time = random.uniform(20, 30)  # 20-30 seconds
            logger.info(f"Rate limit exceeded, waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
            raise RateLimitError(f"Rate limit exceeded: {error_msg}")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            await asyncio.sleep(5)  # Wait a bit for timeouts too
            raise APITimeoutError(f"Request timed out: {error_msg}")
        elif "connect" in error_msg.lower() or "connection" in error_msg.lower():
            await asyncio.sleep(5)  # Wait a bit for connection issues
            raise APIConnectionError(f"Connection error: {error_msg}")
        else:
            raise


async def gemini_model_complete(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: List[Dict[str, Any]] = [],
    keyword_extraction: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    Complete function for Google Gemini model generation.
    
    Args:
        prompt: The prompt to send to the model
        system_prompt: Optional system prompt for context
        history_messages: List of previous messages in the conversation
        keyword_extraction: Whether to extract keywords from the response
        **kwargs: Additional parameters
        
    Returns:
        The model's response as a string
    """
    # Extract and remove keyword_extraction from kwargs if present
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    
    # Get model name from config
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    
    # If keyword extraction is needed, add format instructions
    if keyword_extraction:
        prompt = f"{prompt}\nPlease format your response as a JSON object."
    
    # Call the model
    result = await _gemini_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    
    # Extract JSON if needed
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    
    return result


async def gemini_embed(
    texts: List[str],
    embed_model: str = "models/embedding-001",
    **kwargs
) -> List[List[float]]:
    """
    Generate embeddings using Google Gemini.
    
    Args:
        texts: List of texts to embed
        embed_model: Embedding model to use
        **kwargs: Additional parameters
        
    Returns:
        List of embeddings for each text
    """
    try:
        api_key = kwargs.get("api_key", None)
        if api_key:
            configure_gemini(api_key)
        
        # Get embeddings for each text
        embeddings = []
        for text in texts:
            embedding = genai.embed_content(
                model=embed_model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(embedding["embedding"])
        
        return embeddings
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in gemini_embed: {error_msg}")
        
        if "rate limit" in error_msg.lower():
            raise RateLimitError(f"Rate limit exceeded: {error_msg}")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            raise APITimeoutError(f"Request timed out: {error_msg}")
        elif "connect" in error_msg.lower() or "connection" in error_msg.lower():
            raise APIConnectionError(f"Connection error: {error_msg}")
        else:
            raise 