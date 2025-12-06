"""
Concrete sink implementations.

This module provides ready-to-use sink implementations for common
document storage destinations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fetchcraft.index import IndexFactory
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.interfaces import Sink
from fetchcraft.ingestion.models import DocumentRecord
from fetchcraft.node import DocumentNode, Node

logger = logging.getLogger(__name__)


class VectorIndexSink(Sink):
    """
    Sink that writes documents to a vector index.
    
    This sink takes chunked documents and indexes them in a vector store
    for similarity search and retrieval.
    """
    
    def __init__(
        self,
        vector_index: Optional[VectorIndex[Node]] = None,
        index_factory: Optional[IndexFactory] = None,
        index_id: str = "vector_index"
    ):
        """
        Initialize vector index sink.
        
        Args:
            vector_index: The vector index to write to
            index_id: Identifier for this index (for logging)
        """
        self.vector_index = vector_index
        if vector_index is None and index_factory is None:
            raise ValueError("Either vector_index or index_factory must be provided")
        self.index_id = index_id
        self.index_factory = index_factory
        self._lock = asyncio.Lock()
        self._counter = 0
    
    async def write(self, record: DocumentRecord, context: Optional[dict] = None) -> None:
        """Write document chunks to vector index."""
        async with self._lock:
            try:
                # Get document and chunks
                doc = DocumentNode.model_validate(record.metadata["document"])
                chunk_dicts = record.metadata.get("chunks", [])
                vector_index = self.vector_index or self.index_factory.create_index(VectorIndex, index_id=self.index_id)

                # Add new chunks
                if chunk_dicts:
                    nodes = [Node.model_validate(c) for c in chunk_dicts]
                    await vector_index.add_nodes(doc, nodes)
                    
                    self._counter += 1
                    logger.info(
                        f"{self.index_id}: Indexed document {record.source} "
                        f"with {len(nodes)} chunks (total: {self._counter})"
                    )
                else:
                    logger.warning(
                        f"{self.index_id}: No chunks found for {record.source}"
                    )
                    
            except Exception as e:
                logger.error(
                    f"{self.index_id}: Error indexing {record.source}: {e}",
                    exc_info=True
                )
                raise
    
    def get_name(self) -> str:
        """Get sink name."""
        return f"VectorIndexSink({self.index_id})"
    
    def get_count(self) -> int:
        """Get number of documents indexed."""
        return self._counter



