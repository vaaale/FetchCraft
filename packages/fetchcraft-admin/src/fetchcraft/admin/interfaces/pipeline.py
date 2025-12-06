from __future__ import annotations

from abc import ABC, abstractmethod

from fetchcraft.document_store import DocumentStore
from fetchcraft.index import IndexFactory
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion import IngestionJob, TrackedIngestionPipeline
from fetchcraft.node import Node
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser


class IngestionPipelineFactory(ABC):
    """Interface for creating ingestion pipelines."""

    @abstractmethod
    async def create_pipeline(
        self,
        job: IngestionJob,
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        index_factory: IndexFactory,
        index_id: str = "default",
        include_source: bool = True,
    ) -> TrackedIngestionPipeline:
        """
        Create a configured ingestion pipeline.

        Args:
            job: The ingestion job
            parser_map: Map of mimetype to parser
            chunker: Node parser for chunking
            vector_index: Vector index for storing chunks
            doc_store: Document store for full documents
            index_id: Identifier for the vector index
            include_source: Whether to configure the source (False for recovery)

        Returns:
            Configured pipeline
        """
        pass
