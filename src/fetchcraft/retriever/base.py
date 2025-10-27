"""
Base retriever interface for RAG framework.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Any, Dict
from pydantic import BaseModel, ConfigDict, Field

from ..node import Node, NodeWithScore

D = TypeVar('D', bound=Node)


class Retriever(BaseModel, ABC, Generic[D]):
    """
    Abstract base class for retriever implementations.
    
    Retrievers provide a high-level interface for retrieving documents
    based on text queries. Unlike vector stores which work with embeddings,
    retrievers accept natural language queries and handle the embedding
    generation internally.
    """

    resolve_parents: bool = Field(default=True, description="Whether to resolve parent nodes")
    top_k: int = Field(default=4, description="Number of results to return")
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    def retrieve(
        self, 
        query: str, 
        **kwargs
    ) -> List[NodeWithScore]:
        """
        Synchronous version of retrieve.

        Args:
            query: The query text
            **kwargs: Additional retrieval parameters

        Returns:
            List of NodeWithScore objects containing documents and their relevance scores
        """
        # import asyncio
        # try:
        #     loop = asyncio.get_event_loop()
        # except RuntimeError:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #
        # if loop.is_running():
        #     raise RuntimeError(
        #         "Cannot use sync method when async loop is already running. "
        #         "Use retrieve() or aretrieve() instead."
        #     )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aretrieve(query, **kwargs))


    @abstractmethod
    async def aretrieve(
        self, 
        query: str, 
        **kwargs
    ) -> List[NodeWithScore]:
        """
        Async version of retrieve (for consistency).
        
        Args:
            query: The query text
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of NodeWithScore objects containing documents and their relevance scores
        """
        pass
