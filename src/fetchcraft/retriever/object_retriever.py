from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

from pydantic import BaseModel, PrivateAttr, Field

from .base import Retriever
from ..node import NodeWithScore, ObjectNode, Node


class ObjectMapper(BaseModel, ABC):

    @abstractmethod
    def create_from_node(self, node: Node) -> Any:
        pass

class InMemoryObjectMapper(ObjectMapper):
    _object_map: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def get(self, node: Node) -> Any:
        return self._object_map.get(node.id)

    def set(self, object_id: Any, obj: Any) -> None:
        self._object_map[object_id] = obj


class ObjectRetriever(Retriever[ObjectNode]):
    retriever: Retriever[ObjectNode]
    object_factory: ObjectMapper = Field(default_factory=InMemoryObjectMapper)

    def __init__(self, retriever: Retriever[ObjectNode], object_factory: Optional[ObjectMapper] = None):
        super().__init__(retriever=retriever, object_factory=object_factory)


    async def aretrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        nodes = await self.retriever.aretrieve(query, **kwargs)
        objects = [self.object_factory.get(node.node.object_id) for node in nodes]
        return objects
