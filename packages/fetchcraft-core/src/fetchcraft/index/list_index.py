import logging
from typing import List, Optional, AsyncIterator, Tuple

from fetchcraft.index.base import BaseIndex, D
from fetchcraft.kv_storage import KVStorage
from fetchcraft.kv_storage.storage import InMemoryKVStorage
from fetchcraft.node import Node, ObjectMapper
from fetchcraft.retriever import ListIndexRetriever

logger = logging.getLogger(__name__)


class ListIndex(BaseIndex[Node]):
    storage: KVStorage

    def __init__(self, index_id: Optional[str], storage: Optional[KVStorage] = None, **kwargs):
        _storage = storage or InMemoryKVStorage()
        super().__init__(index_id=index_id, storage=_storage, **kwargs)


    async def add_nodes(self, nodes: List[D], doc: Optional[D] = None, show_progress: bool = False) -> List[str]:
        self.storage.insert_nodes(nodes)

    async def search_by_text_iter(self, query: str, query_embedding: List[float] = None, resolve_parents: bool = True, **kwargs) -> AsyncIterator[Tuple[D, float]]:
        for node in self.storage.get_all():
            score = 1.0
            if resolve_parents:
                node, score = await self._resolve_parent_node(node, 1.0)
            yield node, score

    async def get_document(self, document_id):
        pass

    async def delete_documents(self, document_ids):
        pass

    def as_retriever(self, top_k: int = 4, resolve_parents: bool = True, object_mapper: Optional[ObjectMapper] = None, **search_kwargs):
        return ListIndexRetriever(index=self, top_k=top_k, resolve_parents=resolve_parents, object_mapper=object_mapper, **search_kwargs)