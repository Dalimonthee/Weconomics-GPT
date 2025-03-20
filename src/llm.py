"""LLM module for Google Gemini integration."""

import logging
import os
from typing import Any, Dict, Optional, List, Union, Callable

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold

from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_gemini_llm(
    temperature: float = 0.5, 
    model_name: str = "gemini-2.0-pro-exp", 
    additional_options: Optional[Dict[str, Any]] = None
) -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM instance.
    
    Args:
        temperature: Temperature setting for the LLM
        model_name: The Gemini model to use, defaults to "gemini-pro"
        additional_options: Any additional options to pass to the LLM
        
    Returns:
        A configured ChatGoogleGenerativeAI instance
    """
    options = {
        "temperature": temperature,
        "model": model_name,
        "safety_settings": {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Add any additional options
    if additional_options:
        options.update(additional_options)
    
    # Get the API key from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
    
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        **options
    )


# Pre-compute the guidance text to avoid lambda issues
DEFAULT_GUIDANCE = "No specific search guidance available."


def _format_search_guidance(search_plan: Optional[Any]) -> str:
    """Format the search guidance text based on the search plan.
    
    Args:
        search_plan: A SearchPlan object or None
        
    Returns:
        Formatted search guidance text
    """
    # Default guidance text for when search_plan is None
    default_guidance = """
    Search for clear and relevant information about blockchain concepts and technologies.
    Look for definitions, explanations, and practical applications that address the user's query.
    Prioritize information from authoritative sources and combine information from multiple documents
    for a comprehensive answer.
    """
    
    if not search_plan:
        return default_guidance
    
    # Extract guidance from the search plan
    try:
        guidance_text = search_plan.guidance
        if not guidance_text or guidance_text.strip() == "":
            return default_guidance
            
        # Format the guidance with additional context from the search plan
        formatted_guidance = f"""
        SEARCH GUIDANCE:
        {guidance_text}
        
        KEY CONCEPTS TO FOCUS ON:
        {', '.join(search_plan.key_concepts)}
        
        AREAS TO EXPLORE:
        {', '.join(search_plan.focus_areas)}
        """
        
        return formatted_guidance
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error formatting search guidance: {str(e)}")
        return default_guidance


def format_search_guidance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the search guidance and add it to the input dictionary.
    
    Args:
        data: Input dictionary containing the search_plan
        
    Returns:
        The input dictionary with search_guidance added
    """
    search_plan = data.get("search_plan")
    data["search_guidance"] = _format_search_guidance(search_plan)
    return data


def create_rag_prompt() -> ChatPromptTemplate:
    """Create a RAG prompt template for the Gemini model.
    
    Returns:
        A configured ChatPromptTemplate
    """
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are a blockchain technology expert assistant. Your goal is to provide accurate,
            informative responses to questions about blockchain concepts, technologies, and applications.
            
            When answering questions:
            1. Use the provided context to formulate your response
            2. Focus on being precise and factual
            3. If the context doesn't contain relevant information, say so and provide what general knowledge you have
            4. Always cite your sources when using information from the context
            5. Do not fabricate information or make up facts
            
            {search_guidance}
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """Please answer the following question based on the context provided:
            
            QUESTION: {question}
            
            CONTEXT:
            {context}
            
            Please provide a comprehensive, clear, and factual response. If the context doesn't contain 
            enough information, indicate what's missing but provide the best answer you can.
            """
        ),
    ])
    
    # Return the complete template
    return template


def create_agent_prompt() -> ChatPromptTemplate:
    """Create a prompt template for the RAG agent.
    
    Returns:
        A configured ChatPromptTemplate
    """
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are a blockchain and cryptocurrency expert. 
            Your goal is to provide accurate, informative responses to questions about blockchain
            concepts, technologies, applications, and cryptocurrencies.
            
            Always base your responses on the retrieved information first. If the retrieved information
            doesn't fully answer the question, you can incorporate your knowledge of blockchain, but
            clearly indicate when you're doing so.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            """User query: {query}
            
            Retrieved information:
            {context}
            
            Please provide a comprehensive, accurate answer based on the retrieved information.
            """
        ),
    ])
    return template


def create_manager_prompt() -> ChatPromptTemplate:
    """Create a prompt template for the agent manager.
    
    Returns:
        A configured ChatPromptTemplate
    """
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are an AI coordination manager that oversees a team of specialized agents.
            Your primary role is to:
            1. Understand user queries about blockchain and cryptocurrency
            2. Formulate an effective search strategy to find the most relevant information
            3. Provide clear guidance to specialist agents on what to look for
            4. Synthesize information from multiple sources into a coherent response
            
            You should aim to create search plans that are specific, focused, and designed to
            retrieve the most relevant information for the user's query.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            """User query: {query}
            
            Please analyze this query and determine the best approach to find relevant information.
            Identify the key concepts, terminology, and focus areas that should be searched.
            Then create a specific, targeted search plan.
            """
        ),
    ])
    return template 