import logging
from typing import *

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.schema import QueryType
from pydantic import BaseModel

from fetchcraft.node import NodeType, Chunk
from fetchcraft.node import NodeWithScore as FCNodeWithScore
from fetchcraft.retriever import VectorIndexRetriever

logger = logging.getLogger(__name__)

class LlamaIndexVectorIndexRetriever(BaseModel, BaseRetriever):
    retriever: VectorIndexRetriever

    def __init__(self, retriever: VectorIndexRetriever, **kwargs: Any):
        super().__init__(retriever=retriever, **kwargs)

    def fc_to_li_node(self, node_ws: FCNodeWithScore) -> NodeWithScore:
        if NodeType.CHUNK != node_ws.node.node_type:
            logger.warning(f"Unknown node type: {node_ws.node.node_type}")

        node: Chunk = cast(Chunk, node_ws.node)
        li_node = TextNode(
            text=node.text,
            id=node.id,
            start_char_idx=node.start_char_idx,
            end_char_idx=node.end_char_idx,
            metadata=node.metadata
        )
        li_node_ws = NodeWithScore(node=li_node, score=node_ws.score)
        return li_node_ws

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pass


    def retrieve(self, query_type: QueryType) -> List[NodeWithScore]:
        query = query_type if isinstance(query_type, str) else query_type.query_str

        fc_nodes = self.retriever.retrieve(query)
        li_nodes = [self.fc_to_li_node(fc_node) for fc_node in fc_nodes]
        return li_nodes

    async def aretrieve(self, query_type: QueryType) -> List[NodeWithScore]:
        query = query_type if isinstance(query_type, str) else query_type.query_str

        fc_nodes = await self.retriever.aretrieve(query)
        li_nodes = [self.fc_to_li_node(fc_node) for fc_node in fc_nodes]
        return li_nodes
