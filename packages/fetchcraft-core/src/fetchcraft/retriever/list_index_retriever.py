from typing import *

from pydantic import Field, SkipValidation

from fetchcraft.node import Node, NodeWithScore, ObjectMapper
from fetchcraft.retriever import Retriever


class ListIndexRetriever(Retriever[Node]):
    index: Annotated[Any, SkipValidation()] = Field(description="ListIndex instance")

    def __init__(
        self,
        index: Any,
        top_k: int = 3,
        resolve_parents: bool = True,
        object_mapper: Optional[ObjectMapper] = None,
        filters: Optional[Union['MetadataFilter', Dict[str, Any]]] = None,
        **search_kwargs
    ):
        super().__init__(
            index=index,
            top_k=top_k,
            resolve_parents=resolve_parents,
            object_mapper=object_mapper,
            filters=filters,
            **search_kwargs
        )
        self.index = index


    async def _retrieve(self, query: str, top_k: Optional[int] = 3, **kwargs) -> List[NodeWithScore]:
        num_to_fetch = top_k or self.top_k

        result: List[NodeWithScore] = []
        async for node, score in self.index.search_by_text_iter(query, **kwargs):
            result.append(NodeWithScore(node=node, score=score))
            if len(result) >= num_to_fetch:
                break

        return result