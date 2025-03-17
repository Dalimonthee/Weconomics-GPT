from typing import List, Dict, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from book_agents import BookAgent
import logging
import time
import numpy as np
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str
    book_title: Optional[str] = None
    require_all_books: bool = False
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v

class AgentResponse(BaseModel):
    book_title: str
    answer: str
    confidence: float = 1.0
    processing_time: float

class ManagerAgent:
    def __init__(self, book_agents: List[BookAgent], model_name: str = "gemini-1.5-pro", 
                 cache_size: int = 100, similarity_threshold: float = 0.7):
        """Initialize the manager agent with book agents."""
        self.book_agents = {agent.book_title: agent for agent in book_agents}
        self.model_name = model_name
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self._setup_embeddings = False
        
        # System prompt for the manager
        self.system_prompt = """You are a helpful AI assistant that coordinates specialized book agents.
Your job is to:
1. Determine which book a question is about
2. Route the question to the appropriate book agent
3. If the question is general and doesn't specify a book, choose the most relevant book agent
4. If the question requires comparing books, consult multiple book agents and synthesize their responses

Available books:
""" + "\n".join([f"- {title}" for title in self.book_agents.keys()])

        logger.info(f"Manager initialized with {len(self.book_agents)} book agents: {list(self.book_agents.keys())}")
    
    @lru_cache(maxsize=100)
    def route_question(self, question: str, specific_book: Optional[str] = None, 
                       require_all_books: bool = False) -> str:
        """Route a question to the appropriate book agent(s) and return the answer."""
        start_time = time.time()
        logger.info(f"Processing question: '{question[:50]}...' {'(All books)' if require_all_books else ''}")
        
        try:
            # If a specific book is requested and available, use it
            if specific_book and specific_book in self.book_agents:
                logger.info(f"Routing to specified book: {specific_book}")
                response = self.book_agents[specific_book].answer_question(question)
                return response
            
            # If we need answers from all books
            if require_all_books:
                logger.info("Consulting all book agents and synthesizing responses")
                return self._consult_all_agents(question)
            
            # Otherwise, determine the most relevant book
            book = self._determine_best_book(question)
            logger.info(f"Determined best book: {book}")
            
            if book != "unknown" and book in self.book_agents:
                response = self.book_agents[book].answer_question(question)
                logger.info(f"Got response from {book} agent in {time.time() - start_time:.2f}s")
                return response
            
            # If no book could be determined, consult all agents
            logger.info("No specific book identified, consulting all agents")
            return self._consult_all_agents(question)
            
        except Exception as e:
            logger.error(f"Error routing question: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
        finally:
            logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    def _determine_best_book(self, question: str) -> str:
        """Determine which book a question is about using a decision process."""
        # Simple approach - look for title mentions
        for book_title in self.book_agents:
            if book_title.lower() in question.lower():
                return book_title
        
        # If no direct mention is found, use the model to determine
        return self.determine_book(question)
    
    def determine_book(self, question: str) -> str:
        """Determine which book a question is about."""
        # This will be implemented in the model-specific class
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def _consult_all_agents(self, question: str) -> str:
        """Ask all agents and combine their responses."""
        responses = {}
        for book_title, agent in self.book_agents.items():
            try:
                start_time = time.time()
                response = agent.answer_question(question)
                processing_time = time.time() - start_time
                responses[book_title] = AgentResponse(
                    book_title=book_title,
                    answer=response,
                    processing_time=processing_time
                )
                logger.info(f"Response from {book_title} agent received in {processing_time:.2f}s")
            except Exception as e:
                logger.error(f"Error getting response from {book_title} agent: {str(e)}")
                
        # Synthesize the responses - this will be implemented in subclasses
        return self._synthesize_responses(question, responses)
    
    def _synthesize_responses(self, question: str, responses: Dict[str, AgentResponse]) -> str:
        """Synthesize responses from multiple agents."""
        # This will be implemented in the model-specific class
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def get_book_list(self) -> List[Dict[str, Any]]:
        """Return a list of available books with metadata."""
        return [agent.get_book_metadata() for agent in self.book_agents.values()] 