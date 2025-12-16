from abc import abstractmethod, ABC
from typing import *
from pydantic import BaseModel

from fetchcraft.node import Node


class KVStorage(BaseModel, ABC):

    @abstractmethod
    def insert_nodes(self, nodes: List[Node]):
        ...

    @abstractmethod
    def get_all(self) -> List[Node]:
        ...


class InMemoryKVStorage(KVStorage):
    _storage: Dict[str, Any]

    def __init__(self):
        super().__init__()
        self._storage = {}

    def insert_nodes(self, nodes: List[Node]):
        self._storage.update({node.id: node for node in nodes})

    def get_all(self) -> List[Node]:
        return list(self._storage.values())

