"""
Vector index retriever implementation.
"""
from typing import List, TypeVar, Optional, Any, Dict, Annotated

from pydantic import Field, ConfigDict, SkipValidation

from .base import Retriever
from ..node import Node, NodeWithScore, ObjectMapper

D = TypeVar('D', bound=Node)


class VectorIndexRetriever(Retriever[D]):
    """
    Retriever that uses a VectorIndex for retrieval.
    
    This retriever wraps a VectorIndex to provide a simple text-to-documents
    retrieval interface. Query embedding is handled by the vector store.
    """
    
    vector_index: Annotated[Any, SkipValidation()] = Field(description="VectorIndex instance")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional search parameters")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    def __init__(
        self,
        vector_index: Any,  # VectorIndex[D] - using Any to avoid circular import
        top_k: int = 4,
        resolve_parents: bool = True,
        object_mapper: Optional[ObjectMapper] = None,
        **search_kwargs
    ):
        """
        Initialize the VectorIndexRetriever.
        
        Args:
            vector_index: The VectorIndex instance to use for retrieval (with vector store configured)
            top_k: Number of results to return (default: 4)
            resolve_parents: Whether to resolve parent nodes for SymNodes (default: True)
            **search_kwargs: Additional keyword arguments to pass to search
        """
        super().__init__(
            vector_index=vector_index,
            top_k=top_k,
            resolve_parents=resolve_parents,
            object_mapper=object_mapper,
            search_kwargs=search_kwargs
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "index_id": self.vector_index.index_id,
            "top_k": self.top_k,
            "resolve_parents": self.resolve_parents,
            "search_kwargs": self.search_kwargs
        }


    async def _retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[NodeWithScore]:
        """
        Async version of retrieve (alias for consistency).
        
        Args:
            query: The query text
            top_k: Number of results to return (overrides default if provided)
            **kwargs: Additional search parameters
            
        Returns:
            List of NodeWithScore objects containing documents and their relevance scores
        """
        # Merge kwargs with defaults
        search_params = {**self.search_kwargs, **kwargs}
        k = top_k if top_k is not None else self.top_k

        # Use vector store's search_by_text which handles embedding internally
        results = await self.vector_index.vector_store.search_by_text(
            query=query,
            k=k,
            index_id=self.vector_index.index_id,
            **search_params
        )
        
        # Resolve parents if needed
        if self.resolve_parents:
            results = await self.vector_index._resolve_parent_nodes(results)

        # Convert tuples to NodeWithScore
        return [NodeWithScore(node=doc, score=score) for doc, score in results]

    def update_config(
        self,
        top_k: Optional[int] = None,
        resolve_parents: Optional[bool] = None,
        **search_kwargs
    ) -> None:
        """
        Update retriever configuration.
        
        Args:
            top_k: New default for top_k
            resolve_parents: New default for resolve_parents
            **search_kwargs: Additional search parameters to update
        """
        if top_k is not None:
            self.top_k = top_k
        if resolve_parents is not None:
            self.resolve_parents = resolve_parents
        if search_kwargs:
            self.search_kwargs.update(search_kwargs)
    
    def __repr__(self) -> str:
        return (
            f"VectorIndexRetriever("
            f"index_id={self.vector_index.index_id}, "
            f"top_k={self.top_k}, "
            f"resolve_parents={self.resolve_parents})"
        )
