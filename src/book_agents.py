from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import logging
import time
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BookContent(BaseModel):
    title: str
    content: str
    embedding: Optional[List[float]] = None
    
class BookAgent:
    def __init__(self, book_path: str, model_name: str = "gemini-2.0-pro-exp", cache_size: int = 100):
        """Initialize a book agent with the content of a specific book."""
        self.book_path = book_path
        self.book_title = os.path.basename(book_path).replace(".txt", "")
        self.model_name = model_name
        self.cache_size = cache_size
        
        logger.info(f"Initializing BookAgent for '{self.book_title}'")
        
        try:
            # Load book content
            with open(book_path, "r", encoding="utf-8") as f:
                self.book_content = f.read()
            
            # Set up the system prompt with book content
            self.system_prompt = f"""You are an AI assistant specialized in the book '{self.book_title}'.
Below is the complete text of the book that you should use to answer questions:

{self.book_content}

When answering questions:
1. Cite specific sections or quotes from the book to support your answers
2. If the book doesn't cover the topic, clearly state this fact
3. Maintain the style and tone of the book in your responses when appropriate
4. Do not make up information that isn't in the book"""
            
            logger.info(f"Successfully loaded book: {self.book_title} ({len(self.book_content)} characters)")
        except Exception as e:
            logger.error(f"Failed to load book '{book_path}': {str(e)}")
            raise
    
    @lru_cache(maxsize=100)
    def answer_question(self, question: str) -> str:
        """Answer a question about the book using the specialized knowledge.
        
        This method is cached to avoid repeated API calls for the same question.
        """
        # Implementation will be provided in the specific model implementation
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def _get_cache_key(self, question: str) -> str:
        """Generate a cache key for a question."""
        return hashlib.md5(f"{self.book_title}:{question}".encode()).hexdigest()
    
    def get_book_metadata(self) -> Dict[str, Any]:
        """Return metadata about the book."""
        return {
            "title": self.book_title,
            "path": self.book_path,
            "content_length": len(self.book_content),
            "model": self.model_name
        } 