import google.generativeai as genai
from book_agents import BookAgent
from manager_agent import ManagerAgent, AgentResponse
import os
import logging
import time
from typing import Dict, List, Any, Optional
from functools import lru_cache
import json
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API key
API_KEY = "AIzaSyBpTDVY8Z_T_wOZDMs9UW-tvcD7qHBvx2A"  # Your key here

# Configure the Gemini API
try:
    genai.configure(api_key=API_KEY)
    logger.info("Google Generative AI API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI API: {str(e)}")
    raise

class GeminiBookAgent(BookAgent):
    def __init__(self, book_path: str, model_name: str = "gemini-2.0-pro-exp", cache_size: int = 100):
        """Initialize a book agent that uses Gemini for answering questions."""
        super().__init__(book_path, model_name, cache_size)
        
        try:
            # Initialize the model
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"Initialized Gemini model {model_name} for {self.book_title}")
            
            # Create a directory for storing cached responses
            os.makedirs("cache", exist_ok=True)
            self.cache_file = f"cache/{self.book_title}_cache.json"
            self._load_cache()
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
    
    def _load_cache(self):
        """Load the cache from file if it exists."""
        self.response_cache = {}
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.response_cache = json.load(f)
                logger.info(f"Loaded {len(self.response_cache)} cached responses for {self.book_title}")
        except Exception as e:
            logger.warning(f"Failed to load cache for {self.book_title}: {str(e)}")
    
    def _save_cache(self):
        """Save the cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f)
            logger.info(f"Saved {len(self.response_cache)} cached responses for {self.book_title}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {self.book_title}: {str(e)}")
    
    def answer_question(self, question: str) -> str:
        """Answer a question about the book using Gemini with caching.
        
        Args:
            question: The question to answer about the book
            
        Returns:
            str: The answer to the question
        """
        cache_key = self._get_cache_key(question)
        
        # Check if response is in cache
        if cache_key in self.response_cache:
            logger.info(f"Cache hit for question about {self.book_title}")
            return self.response_cache[cache_key]
        
        logger.info(f"Cache miss for question about {self.book_title}, generating response")
        
        try:
            # Combine system instructions with user question since system role is not supported
            combined_prompt = f"{self.system_prompt}\n\nQuestion: {question}"
            
            # Generate response with user role only
            response = self.model.generate_content(combined_prompt)
            answer = response.text
            
            # Cache the response
            self.response_cache[cache_key] = answer
            
            # Keep cache size under control
            if len(self.response_cache) > self.cache_size:
                # Remove oldest entries
                keys = list(self.response_cache.keys())
                for k in keys[:len(keys) - self.cache_size]:
                    del self.response_cache[k]
            
            # Save cache periodically
            if len(self.response_cache) % 5 == 0:
                self._save_cache()
                
            return answer
            
        except Exception as e:
            error_msg = f"Error generating response from Gemini for {self.book_title}: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, I encountered an error when trying to answer your question about {self.book_title}. {str(e)}"

class GeminiManagerAgent(ManagerAgent):
    def __init__(self, book_agents: List[BookAgent], model_name: str = "gemini-2.0-pro-exp", 
                 cache_size: int = 100, similarity_threshold: float = 0.7):
        """Initialize a manager agent that uses Gemini for routing and synthesis."""
        super().__init__(book_agents, model_name, cache_size, similarity_threshold)
        
        try:
            # Initialize the model
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"Initialized Gemini model {model_name} for manager agent")
            
            # Create a directory for storing cached responses
            os.makedirs("cache", exist_ok=True)
            self.cache_file = "cache/manager_cache.json"
            self._load_cache()
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model for manager: {str(e)}")
            raise
    
    def _load_cache(self):
        """Load the cache from file if it exists."""
        self.response_cache = {}
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.response_cache = json.load(f)
                logger.info(f"Loaded {len(self.response_cache)} cached responses for manager")
        except Exception as e:
            logger.warning(f"Failed to load cache for manager: {str(e)}")
    
    def _save_cache(self):
        """Save the cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f)
            logger.info(f"Saved {len(self.response_cache)} cached responses for manager")
        except Exception as e:
            logger.warning(f"Failed to save cache for manager: {str(e)}")
    
    def determine_book(self, question: str) -> str:
        """Determine which book a question is about using Gemini."""
        cache_key = f"book_determination:{question}"
        
        # Check cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        prompt = f"""
Based on the question below, determine which book it's most likely asking about.
Available books: {', '.join(self.book_agents.keys())}

Question: {question}

Return only the exact book title from the available books, or "unknown" if you can't determine.
"""
        try:
            # Using simple text prompt
            response = self.model.generate_content(prompt)
            book = response.text.strip()
            
            # Cache the result
            self.response_cache[cache_key] = book
            self._save_cache()
            
            if book in self.book_agents:
                return book
            return "unknown"
        except Exception as e:
            logger.error(f"Error determining book: {str(e)}")
            return "unknown"
    
    def _synthesize_responses(self, question: str, responses: Dict[str, AgentResponse]) -> str:
        """Synthesize responses from multiple agents using Gemini."""
        if not responses:
            return "No responses available from book agents."
        
        response_text = "\n\n".join([
            f"Expert on {resp.book_title}: {resp.answer}" 
            for _, resp in responses.items()
        ])
        
        synthesis_prompt = f"""
I asked the same question to multiple book experts and got these responses:

{response_text}

Please synthesize these responses into a comprehensive answer to the original question: {question}
Focus on comparing and contrasting the perspectives from different books when relevant.
"""
        try:
            # Using simple text prompt
            synthesis_response = self.model.generate_content(synthesis_prompt)
            return synthesis_response.text
        except Exception as e:
            logger.error(f"Error synthesizing responses: {str(e)}")
            
            # Fallback: return the individual responses
            fallback = "I couldn't synthesize the responses, but here are the individual answers:\n\n"
            for book, resp in responses.items():
                fallback += f"--- From {book} expert ---\n{resp.answer}\n\n"
            return fallback 