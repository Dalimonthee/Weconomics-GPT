"""Agent module for the agentic RAG system."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.llm import create_gemini_llm, create_rag_prompt, create_agent_prompt, create_manager_prompt, format_search_guidance
from src.vector_store import SupabaseVectorStoreManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UserQuery:
    """Class representing a user query."""
    
    original_query: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPlan:
    """Class representing a search plan created by the manager."""
    
    reformulated_query: str
    focus_areas: List[str]
    key_concepts: List[str]
    guidance: str
    priority_filters: Dict[str, Any] = field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Document chunk model."""
    content: str = Field(..., description="Content of the document chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the document chunk")


class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str = Field(..., description="The answer to the user's query")
    sources: List[DocumentChunk] = Field(
        default_factory=list, description="Source documents used to answer the query"
    )
    search_quality: Dict[str, Any] = Field(
        default_factory=dict, description="Quality metrics for the search"
    )


class RAGAgent:
    """Agent with RAG capabilities."""
    
    def __init__(self, vector_store: Optional[SupabaseVectorStoreManager] = None):
        """Initialize the RAG agent.
        
        Args:
            vector_store: Optional custom vector store manager. If not provided,
                          a default one will be created.
        """
        logger.info("Initializing RAG Agent")
        self.vector_store = vector_store or SupabaseVectorStoreManager()
        self.llm = create_gemini_llm(temperature=0.2)  # Lower temperature for more factual responses
        self.chat_history = []
        
        # Create the chain with explicit format_search_guidance step
        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._get_context(
                    query=x["search_query"],
                    k=1000,  # Retrieve more documents initially
                    search_plan=x.get("search_plan", None)
                )
            )
            | format_search_guidance
            | create_rag_prompt()
            | self.llm
            | StrOutputParser()
        )
    
    def _get_context(self, query: str, k: int = 1000, search_plan: Optional[SearchPlan] = None) -> str:
        """Get context from vector store.
        
        Args:
            query: The query to search for
            k: Number of results to return
            search_plan: Optional search plan from the manager
            
        Returns:
            Formatted context string
        """
        logger.info(f"Getting context for query: {query}")
        
        # Ensure we have a valid query string
        effective_query = query
        if not effective_query or effective_query.strip() == "":
            logger.warning("Empty query provided, using fallback query")
            if search_plan and search_plan.reformulated_query:
                effective_query = search_plan.reformulated_query
                logger.info(f"Using reformulated query as fallback: {effective_query}")
            else:
                # Last resort fallback
                effective_query = "blockchain"
                logger.warning(f"Using hardcoded fallback query: {effective_query}")
        
        # Use search plan to enhance retrieval if available
        if search_plan:
            logger.info(f"Using search plan with reformulated query: {search_plan.reformulated_query}")
            
            # Ensure we have a valid reformulated query
            search_query = search_plan.reformulated_query
            if not search_query or search_query.strip() == "":
                search_query = effective_query
                logger.warning(f"Empty reformulated query, using original: {search_query}")
            
            # Use the reformulated query as the primary search
            documents = self.vector_store.similarity_search(search_query, k=k)
            
            # If we got fewer than k/2 results, also try the original query as a fallback
            if len(documents) < k/2:
                logger.info(f"Insufficient results with reformulated query, adding results from original query")
                additional_docs = self.vector_store.similarity_search(effective_query, k=k//2)
                # Add only documents that are not duplicates
                existing_contents = [doc.page_content for doc in documents]
                for doc in additional_docs:
                    if doc.page_content not in existing_contents:
                        documents.append(doc)
            
            # Apply priority filtering based on search plan
            documents = self._prioritize_documents(documents, search_plan)
            
            # Log stats about the retrieved documents
            search_quality = self._evaluate_search_quality(documents, search_plan)
            logger.info(f"Search quality: {search_quality}")
        else:
            # Standard search without a plan
            documents = self.vector_store.similarity_search(effective_query, k=k)
        
        # Format the context
        if documents:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = doc.page_content.strip()
                if not content:
                    continue
                
                # Extract metadata about source when available
                source_info = ""
                if hasattr(doc, 'metadata') and doc.metadata:
                    if 'file_id' in doc.metadata:
                        source_info = f" (Source: {doc.metadata['file_id']})"
                
                # Format each document with better separation and source info
                context_parts.append(f"Document {i}{source_info}:\n{content}\n")
            
            return "\n".join(context_parts)
        return "No relevant information found."
    
    def _prioritize_documents(self, documents: List[Document], search_plan: SearchPlan) -> List[Document]:
        """Prioritize and filter documents based on search plan.
        
        Args:
            documents: List of retrieved documents
            search_plan: Search plan from the manager
            
        Returns:
            Filtered and prioritized documents
        """
        if not documents:
            return documents
            
        # Score documents based on relevance to key concepts
        scored_docs = []
        for doc in documents:
            score = 0
            content = doc.page_content.lower()
            
            # Score based on key concepts presence
            for concept in search_plan.key_concepts:
                if concept.lower() in content:
                    score += 3  # Higher weight for key concepts
            
            # Score based on focus areas
            for area in search_plan.focus_areas:
                if area.lower() in content:
                    score += 2
            
            # Apply any custom filters from the search plan
            if search_plan.priority_filters:
                for filter_key, filter_value in search_plan.priority_filters.items():
                    if filter_key == "min_length" and len(content) >= filter_value:
                        score += 1
                    elif filter_key == "contains_phrase" and filter_value.lower() in content:
                        score += 3
            
            scored_docs.append((doc, score))
        
        # Sort by score (highest first) and take top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return the documents in priority order
        return [doc for doc, _ in scored_docs]
    
    def _evaluate_search_quality(self, documents: List[Document], search_plan: SearchPlan) -> Dict[str, Any]:
        """Evaluate the quality of search results against the search plan.
        
        Args:
            documents: List of retrieved documents
            search_plan: Search plan from the manager
            
        Returns:
            Dictionary with quality metrics
        """
        if not documents:
            return {
                "found_documents": 0,
                "avg_document_length": 0,
                "key_concept_coverage": 0,
                "assessment": "No documents found"
            }
        
        # Calculate basic stats
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_length = total_length / len(documents)
        
        # Check concept coverage
        concept_coverage = {}
        for concept in search_plan.key_concepts:
            concept_lower = concept.lower()
            matches = sum(1 for doc in documents if concept_lower in doc.page_content.lower())
            concept_coverage[concept] = matches
        
        # Calculate percentage of concepts covered
        concepts_found = sum(1 for count in concept_coverage.values() if count > 0)
        total_concepts = len(search_plan.key_concepts) if search_plan.key_concepts else 1
        coverage_percentage = (concepts_found / total_concepts) * 100 if total_concepts > 0 else 0
        
        # Determine quality assessment
        if coverage_percentage >= 80:
            assessment = "Excellent"
        elif coverage_percentage >= 60:
            assessment = "Good"
        elif coverage_percentage >= 40:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        return {
            "found_documents": len(documents),
            "avg_document_length": avg_length,
            "key_concept_coverage": round(coverage_percentage),
            "concept_coverage": concept_coverage,
            "assessment": assessment
        }
    
    async def process_query(self, query: UserQuery, search_plan: Optional[SearchPlan] = None) -> str:
        """Process a query and generate a response.
        
        Args:
            query: UserQuery object with the original query
            search_plan: Optional search plan from the manager
            
        Returns:
            Generated response
        """
        logger.info(f"RAG Agent processing query: {query.original_query}")
        
        # Use the search plan if available, otherwise use the original query
        search_query = search_plan.reformulated_query if search_plan and search_plan.reformulated_query else query.original_query
        
        # Ensure the search query is never empty
        if not search_query or search_query.strip() == "":
            search_query = query.original_query
            logger.warning(f"Empty search query, using original query: {search_query}")
        
        # Prepare input for chain
        chain_input = {
            "question": query.original_query,
            "search_query": search_query,
            "search_plan": search_plan
        }
        
        # Execute the chain
        try:
            logger.info("Executing RAG chain")
            response = self.chain.invoke(chain_input)
            
            # Handle AIMessage object by extracting the content if needed
            if hasattr(response, 'content'):
                response = response.content
                
            logger.info("RAG chain execution completed successfully")
            return response
        except Exception as e:
            logger.error(f"Error executing RAG chain: {str(e)}", exc_info=True)
            # Provide a fallback response if the chain fails
            return f"I'm sorry, I couldn't process your query about '{query.original_query}'. There was an error in retrieving or synthesizing the information."


class AgentManager:
    """Manager for coordinating between agents."""
    
    def __init__(self, vector_store: Optional[SupabaseVectorStoreManager] = None):
        """Initialize the agent manager.
        
        Args:
            vector_store: Optional custom vector store manager. If not provided,
                         a default one will be created and passed to the RAG agent.
        """
        logger.info("Initializing Agent Manager")
        self.vector_store = vector_store or SupabaseVectorStoreManager()
        self.rag_agent = RAGAgent(vector_store=self.vector_store)
        self.llm = create_gemini_llm(temperature=0.3)  # Lower temperature for more precise responses
        self.planning_llm = create_gemini_llm(temperature=0.2)  # Lower temperature for planning
        self.chat_history = []
        self.prompt = create_manager_prompt()
        self.planning_prompt = self._create_planning_prompt()
    
    def _create_planning_prompt(self) -> PromptTemplate:
        """Create the prompt for the planning LLM."""
        planning_template = """You are a search planning expert for a blockchain information retrieval system. Your job is to create a detailed and explicit search plan.

USER QUERY: "{query}"

IMPORTANT: You MUST create a search plan that follows this EXACT format:

```
REFORMULATED_QUERY: <Write a specific, clear search query that will retrieve relevant information>

FOCUS_AREAS:
- <Focus area 1>
- <Focus area 2>
- <Focus area 3>

KEY_CONCEPTS:
- <Key concept 1>
- <Key concept 2>
- <Key concept 3>
- <Key concept 4>

GUIDANCE:
<Write detailed guidance for analyzing and prioritizing information>
```

INSTRUCTIONS:
1. The REFORMULATED_QUERY must be clear, specific and directly related to blockchain or the user's query
2. FOCUS_AREAS should identify 3-5 specific aspects to investigate
3. KEY_CONCEPTS should list 3-7 important terms to find in documents
4. GUIDANCE should provide detailed instructions for analyzing information

Remember, the REFORMULATED_QUERY is extremely important and must NEVER be empty. 
It should be a better version of the original query that will help find relevant information.

DO NOT skip any of the sections. ALL sections MUST be present with meaningful content.
"""
        return PromptTemplate.from_template(planning_template)
    
    def _parse_search_plan(self, plan_text: str, original_query: str) -> SearchPlan:
        """Parse the search plan text into a SearchPlan object.
        
        Args:
            plan_text: Raw text output from the planning LLM
            original_query: The original user query (used as fallback)
            
        Returns:
            SearchPlan object
        """
        logger.info("Parsing search plan from LLM output")
        
        # Initialize default values
        reformulated_query = ""
        focus_areas = []
        key_concepts = []
        guidance = ""
        priority_filters = {}
        
        # Extract reformulated query
        if "REFORMULATED_QUERY:" in plan_text:
            parts = plan_text.split("REFORMULATED_QUERY:")
            if len(parts) > 1:
                query_section = parts[1].split("\n", 1)[0].strip()
                reformulated_query = query_section
                logger.info(f"Extracted reformulated query: '{reformulated_query}'")
        
        # Extract focus areas
        if "FOCUS_AREAS:" in plan_text:
            parts = plan_text.split("FOCUS_AREAS:")
            if len(parts) > 1:
                areas_section = parts[1].split("KEY_CONCEPTS:" if "KEY_CONCEPTS:" in parts[1] else "GUIDANCE:")[0]
                for line in areas_section.strip().split("\n"):
                    if line.strip().startswith("-"):
                        area = line.strip()[1:].strip()
                        if area:
                            focus_areas.append(area)
            logger.info(f"Extracted {len(focus_areas)} focus areas")
        
        # Extract key concepts
        if "KEY_CONCEPTS:" in plan_text:
            parts = plan_text.split("KEY_CONCEPTS:")
            if len(parts) > 1:
                concepts_section = parts[1].split("GUIDANCE:")[0] if "GUIDANCE:" in parts[1] else parts[1]
                for line in concepts_section.strip().split("\n"):
                    if line.strip().startswith("-"):
                        concept = line.strip()[1:].strip()
                        if concept:
                            key_concepts.append(concept)
            logger.info(f"Extracted {len(key_concepts)} key concepts")
        
        # Extract guidance
        if "GUIDANCE:" in plan_text:
            parts = plan_text.split("GUIDANCE:")
            if len(parts) > 1:
                guidance = parts[1].strip()
                logger.info("Successfully extracted guidance")
        
        # Handle case where reformulated query is empty
        if not reformulated_query or reformulated_query.strip() == "":
            logger.warning("Reformulated query is empty, creating fallback")
            
            # First try to use key concepts
            if key_concepts:
                reformulated_query = " ".join(key_concepts[:3])
                logger.info(f"Created fallback query from key concepts: '{reformulated_query}'")
            # If key concepts are also empty, just use the original query
            else:
                reformulated_query = original_query if original_query else "blockchain definition"
                logger.info(f"Using original query as fallback: '{reformulated_query}'")
        
        # Add default focus areas if none were extracted
        if not focus_areas:
            logger.warning("No focus areas found, adding defaults")
            if "definition" in reformulated_query.lower() or "what is" in reformulated_query.lower():
                focus_areas = ["Basic definitions", "Core concepts", "Fundamental principles"]
            else:
                focus_areas = ["Technical aspects", "Applications", "Key features"]
        
        # Add default key concepts if none were extracted
        if not key_concepts:
            logger.warning("No key concepts found, adding defaults")
            key_concepts = ["blockchain", "distributed ledger", "decentralization", "consensus mechanism"]
        
        # Add default guidance if none was extracted
        if not guidance or guidance.strip() == "":
            logger.warning("No guidance found, adding default")
            guidance = "Look for clear definitions and explanations. Prioritize information that directly answers the query. Combine information from multiple sources for a comprehensive answer."
        
        # Create and return the SearchPlan
        return SearchPlan(
            reformulated_query=reformulated_query,
            focus_areas=focus_areas,
            key_concepts=key_concepts,
            guidance=guidance,
            priority_filters=priority_filters
        )
    
    async def _create_search_plan(self, query: str) -> SearchPlan:
        """Create a search plan based on the user's query.
        
        Args:
            query: The user's query
            
        Returns:
            A SearchPlan object
        """
        logger.info(f"Creating search plan for query: '{query}'")
        
        # Ensure we have a valid query
        if not query or query.strip() == "":
            logger.warning("Empty query provided, using default query")
            query = "blockchain technology"
        
        # Prepare a cleaned version of the query (ensure it's spelled correctly)
        cleaned_query = query.replace("blockhain", "blockchain")
        
        try:
            # Generate the search plan using the planning LLM
            formatted_prompt = self.planning_prompt.format(query=cleaned_query)
            logger.info("Sending prompt to planning LLM")
            plan_text_response = self.planning_llm.invoke(formatted_prompt)
            
            # Handle AIMessage object by extracting the content
            if hasattr(plan_text_response, 'content'):
                plan_text = plan_text_response.content
            else:
                plan_text = str(plan_text_response)
                
            logger.info(f"Received plan from LLM ({len(plan_text)} characters)")
            
            # Parse the plan text into a SearchPlan object
            search_plan = self._parse_search_plan(plan_text, cleaned_query)
            
            # Log the created search plan
            logger.info(f"Created search plan:")
            logger.info(f"  Reformulated query: '{search_plan.reformulated_query}'")
            logger.info(f"  Focus areas: {search_plan.focus_areas}")
            logger.info(f"  Key concepts: {search_plan.key_concepts}")
            
            return search_plan
            
        except Exception as e:
            logger.error(f"Error creating search plan: {str(e)}", exc_info=True)
            # Create a fallback search plan
            return SearchPlan(
                reformulated_query=cleaned_query,
                focus_areas=["Basic concepts", "Technical aspects", "Applications"],
                key_concepts=["blockchain", "distributed ledger", "decentralization", "consensus"],
                guidance="Look for clear definitions and explanations that directly answer the query."
            )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the manager and agent.
        
        Args:
            query: The user's query
            
        Returns:
            A dictionary with the final response and debug information
        """
        logger.info(f"Manager processing query: '{query}'")
        query_obj = UserQuery(original_query=query)
        
        # Create execution trace for debugging
        trace = {
            "original_query": query,
        }
        
        try:
            # Create a search plan for this query
            search_plan = await self._create_search_plan(query)
            trace["search_plan"] = {
                "reformulated_query": search_plan.reformulated_query,
                "focus_areas": search_plan.focus_areas,
                "key_concepts": search_plan.key_concepts
            }
            
            # Get a response from the RAG agent
            agent_response = await self.rag_agent.process_query(query_obj, search_plan)
            
            # Get the vector search results
            documents = self.vector_store.similarity_search(
                search_plan.reformulated_query, 
                k=1000  # Retrieving enough documents but not too many
            )
            
            # Evaluate search quality
            search_quality = self.rag_agent._evaluate_search_quality(documents, search_plan)
            
            # Format the sources
            sources = []
            if documents:
                trace["num_sources_retrieved"] = len(documents)
                trace["avg_source_length"] = sum(len(doc.page_content) for doc in documents) / len(documents)
                
                for doc in documents:
                    source = {
                        "content": doc.page_content,
                        "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                    }
                    sources.append(source)
            
            # Return the response and metadata
            return {
                "final_response": agent_response,
                "sources": sources,
                "trace": trace,
                "search_plan": {
                    "reformulated_query": search_plan.reformulated_query,
                    "focus_areas": search_plan.focus_areas,
                    "key_concepts": search_plan.key_concepts,
                    "guidance": search_plan.guidance
                },
                "search_quality": search_quality
            }
        
        except Exception as e:
            logger.error(f"Error in manager.process_query: {str(e)}", exc_info=True)
            # Return a fallback response
            return {
                "final_response": f"I'm sorry, I couldn't process your query about '{query}'. There was an error in retrieving or synthesizing the information.",
                "sources": [],
                "trace": trace,
                "search_plan": {
                    "reformulated_query": query,
                    "focus_areas": [],
                    "key_concepts": [],
                    "guidance": "Error occurred during processing."
                },
                "search_quality": {
                    "found_documents": 0,
                    "assessment": "Error"
                }
            } 