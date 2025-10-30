"""
Base retriever interface for RAG framework.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Optional, cast

from pydantic import BaseModel, ConfigDict, Field

from ..mixins import ObjectNodeMixin
from ..node import Node, NodeWithScore, NodeType, ObjectMapper, ObjectNode, DefaultObjectMapper, ObjectType

D = TypeVar('D', bound=Node)



class Retriever(BaseModel, ABC, Generic[D], ObjectNodeMixin):
    """
    Abstract base class for retriever implementations.
    
    Retrievers provide a high-level interface for retrieving documents
    based on text queries. Unlike vector stores which work with embeddings,
    retrievers accept natural language queries and handle the embedding
    generation internally.
    """

    resolve_parents: bool = Field(default=True, description="Whether to resolve parent nodes")
    object_mapper: Optional[ObjectMapper] = None
    top_k: int = Field(default=4, description="Number of results to return")
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    def __init__(self, object_mapper: Optional[ObjectMapper] = None, **kwargs):
        object_mapper = object_mapper or DefaultObjectMapper()
        super().__init__(object_mapper=object_mapper, **kwargs)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
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
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aretrieve(query, top_k=top_k, **kwargs))

    async def aretrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[NodeWithScore]:
        """
        Async version of retrieve.

        Args:
            query: The query text
            **kwargs: Additional retrieval parameters

        Returns:
            List of NodeWithScore objects containing documents and their relevance scores
        """
        nodes = await self._retrieve(query=query, top_k=top_k, **kwargs)

        # Resolve nodes
        result = []
        resolved_nodes = await self._resolve_recursively(nodes, result, query, top_k, **kwargs)
        # for scored_node in nodes:
        #     node = scored_node.node
        #     score = scored_node.score
        #     if NodeType.OBJECT == node.node_type:
        #         new_nodes = await self._resolve_recursively(node, query, top_k, **kwargs)
        #         result.extend(new_nodes)
        #     else:
        #         result.append(node)
        return resolved_nodes

    async def _resolve_recursively(
        self,
        nodes: List[Node] | List[NodeWithScore],
        resolved_nodes: List[Node] | List[NodeWithScore],
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[NodeWithScore]:
        """
        Recursively resolve a node and its children.

        Args:
            node: The node to resolve

        Returns:
            List of NodeWithScore objects containing documents and their relevance scores
        """
        for node in nodes:
            _node = node
            if NodeType.NODE_WITH_SCORE == node.node_type:
                _node = node.node

            if NodeType.OBJECT == _node.node_type:
                _node = cast(ObjectNode, _node)
                new_nodes = await self.object_mapper.resolve_object_node(_node, query, top_k, **kwargs)
                await self._resolve_recursively(new_nodes, resolved_nodes, query, top_k, **kwargs)
            else:
                resolved_nodes.append(node)
        return resolved_nodes

    @abstractmethod
    async def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
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
