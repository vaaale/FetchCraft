"""
Concrete sink implementations.

This module provides ready-to-use sink implementations for common
document storage destinations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fetchcraft.document_store import DocumentStore
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.interfaces import ISink
from fetchcraft.ingestion.models import DocumentRecord
from fetchcraft.node import DocumentNode, Node

logger = logging.getLogger(__name__)


class VectorIndexSink(ISink):
    """
    Sink that writes documents to a vector index.
    
    This sink takes chunked documents and indexes them in a vector store
    for similarity search and retrieval.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex[Node],
        index_id: str = "vector_index"
    ):
        """
        Initialize vector index sink.
        
        Args:
            vector_index: The vector index to write to
            index_id: Identifier for this index (for logging)
        """
        self.vector_index = vector_index
        self.index_id = index_id
        self._lock = asyncio.Lock()
        self._counter = 0
    
    async def write(self, record: DocumentRecord) -> None:
        """Write document chunks to vector index."""
        async with self._lock:
            try:
                # Get document and chunks
                doc = DocumentNode.model_validate(record.metadata["document"])
                chunk_dicts = record.metadata.get("chunks", [])
                
                # Add new chunks
                if chunk_dicts:
                    nodes = [Node.model_validate(c) for c in chunk_dicts]
                    await self.vector_index.add_nodes(doc, nodes)
                    
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


class DocumentStoreSink(ISink):
    """
    Sink that writes documents to a document store.
    
    This sink persists the full document to a document store (e.g., MongoDB)
    for later retrieval.
    """
    
    def __init__(self, doc_store: DocumentStore):
        """
        Initialize document store sink.
        
        Args:
            doc_store: The document store to write to
        """
        self.doc_store = doc_store
        self._lock = asyncio.Lock()
        self._counter = 0
    
    async def write(self, record: DocumentRecord) -> None:
        """Write document to document store."""
        async with self._lock:
            try:
                doc = DocumentNode.model_validate(record.metadata["document"])
                
                # Add document to store
                node_id = await self.doc_store.add_document(doc)
                doc.id = node_id
                
                # Update record with stored document ID
                record.metadata["document"] = doc.model_dump()
                record.metadata["document_store_id"] = node_id
                
                self._counter += 1
                logger.info(
                    f"DocumentStoreSink: Stored document {record.source} "
                    f"(ID: {node_id}, total: {self._counter})"
                )
                
            except Exception as e:
                logger.error(
                    f"DocumentStoreSink: Error storing {record.source}: {e}",
                    exc_info=True
                )
                raise
    
    def get_name(self) -> str:
        """Get sink name."""
        return "DocumentStoreSink"
    
    def get_count(self) -> int:
        """Get number of documents stored."""
        return self._counter


class LoggingSink(ISink):
    """
    Simple logging sink for testing and debugging.
    
    This sink just logs document information without persisting anything.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize logging sink.
        
        Args:
            log_level: Log level to use
        """
        self.log_level = log_level
        self._counter = 0
    
    async def write(self, record: DocumentRecord) -> None:
        """Log document information."""
        self._counter += 1
        logger.log(
            self.log_level,
            f"LoggingSink: Document {self._counter}: {record.source} "
            f"(status: {record.status}, metadata keys: {list(record.metadata.keys())})"
        )
    
    def get_name(self) -> str:
        """Get sink name."""
        return "LoggingSink"
    
    def get_count(self) -> int:
        """Get number of documents logged."""
        return self._counter
