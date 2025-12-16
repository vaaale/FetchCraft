"""
Service interfaces for the Fetchcraft MCP Server.

These abstract base classes define the contracts that service implementations
must follow. Users can provide custom implementations for dependency injection.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class FindFilesService(ABC):
    """
    Abstract interface for file finding/search functionality.
    
    Implementations should provide semantic search capabilities
    over a document collection.
    """

    @abstractmethod
    async def find_files(
        self, 
        query: str, 
        num_results: int = 10, 
        offset: int = 0
    ) -> List[Any]:
        """
        Find files using semantic search with pagination.
        
        Args:
            query: The search query
            num_results: Number of results to return
            offset: Offset for pagination
            
        Returns:
            List of NodeWithScore objects or similar results
        """
        ...


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str
    citations: Dict[str, Any] = {}
    model: str


class QueryService(ABC):
    """
    Abstract interface for RAG query functionality.
    
    Implementations should provide question-answering capabilities
    using retrieval-augmented generation.
    """

    @abstractmethod
    async def query(
        self,
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> QueryResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            top_k: Number of documents to retrieve
            include_citations: Whether to include citations in response
            
        Returns:
            QueryResponse with answer, citations, and model info
        """
        ...
