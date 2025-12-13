from typing import *

from fetchcraft.text_splitter import TextSplitter
from fetchcraft.node import Chunk, Node, DocumentNode
from fetchcraft.node_parser.base import NodeParser
from fetchcraft.text_splitter import RecursiveTextSplitter


class SimpleNodeParser(NodeParser):
    chunk_size: int = 4096
    overlap: int = 200
    text_splitter: TextSplitter

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 0,
        text_splitter: Optional[TextSplitter] = None
    ):
        if text_splitter is None:
            text_splitter = RecursiveTextSplitter()
        super().__init__(chunk_size=chunk_size, overlap=overlap, text_splitter=text_splitter)

    def get_nodes(self, documents: List[DocumentNode], metadata: Optional[Dict[str, Any]] = None) -> List[Node]:
        if metadata is None:
            metadata = {}
            
        chunk_nodes = []
        for document in documents:
            text = document.text
            chunk_tuples = self.text_splitter.split(text, self.chunk_size, self.overlap)

            for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_tuples):
                chunk = Chunk.from_text(
                    text=chunk_text,
                    chunk_index=idx,
                    start_char_idx=start_idx,
                    end_char_idx=end_idx,
                    metadata={**metadata, "total_chunks": len(chunk_tuples)}
                )

                # Set doc_id from parent_node (but don't set parent relationship for first-level chunks)
                if document:
                    chunk.doc_id = document.doc_id

                # Link to previous chunk (sibling relationship)
                if chunk_nodes:
                    prev_chunk = chunk_nodes[-1]
                    chunk.previous_id = prev_chunk.id
                    prev_chunk.next_id = chunk.id

                chunk_nodes.append(chunk)

        return chunk_nodes

    def __repr__(self) -> str:
        return f"SimpleNodeParser(chunk_size={self.chunk_size}, overlap={self.overlap})"
