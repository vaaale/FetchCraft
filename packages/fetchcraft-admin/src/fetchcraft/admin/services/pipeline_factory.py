"""
Pipeline factory for creating configured ingestion pipelines.

This factory encapsulates the logic for building pipelines with
the appropriate transformations and sinks based on configuration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Type

from fetchcraft.admin.interfaces import IngestionPipelineFactory
from fetchcraft.connector.filesystem import FilesystemConnector
from fetchcraft.index import BaseIndex
from fetchcraft.index import IndexFactory
from fetchcraft.index import VectorIndex
from fetchcraft.ingestion.interfaces import QueueBackend
from fetchcraft.ingestion.models import IngestionJob
from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline
from fetchcraft.ingestion.repository import JobRepository, DocumentRepository, TaskRepository
from fetchcraft.ingestion.sinks import VectorIndexSink
from fetchcraft.ingestion.sources import ConnectorSource
from fetchcraft.ingestion.transformations import (
    AsyncParsingTransformation,
    ChunkingTransformation,
    ExtractKeywords,
)
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser

logger = logging.getLogger(__name__)

class DefaultIndexFactory(IndexFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_index(self, index_cls: Type[BaseIndex], **kwargs) -> BaseIndex:
        _options = self.options.copy()
        if "index_id" in kwargs and "index_id" in _options:
            _options.pop("index_id")

        if index_cls == VectorIndex:
            return VectorIndex(**_options, **kwargs)
        else:
            raise ValueError(f"Unknown index class: {index_cls}")


class DefaultIngestionPipelineFactory(IngestionPipelineFactory):
    """
    Factory for creating ingestion pipelines with standard configuration.
    
    This factory encapsulates the logic for building pipelines with:
    - Filesystem source (when include_source=True)
    - Parsing transformation
    - Keyword extraction
    - Chunking transformation
    - Document store sink
    - Vector index sink
    """

    def __init__(
        self,
        queue_backend: QueueBackend,
        job_repo: JobRepository,
        doc_repo: DocumentRepository,
        document_root: Path,
        num_workers: int = 4,
        task_repo: Optional[TaskRepository] = None,
        callback_base_url: str = "",
    ):
        """
        Initialize the pipeline factory.
        
        Args:
            queue_backend: Queue backend for pipeline
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            document_root: Root directory for documents
            num_workers: Number of concurrent workers
            task_repo: Repository for task tracking (required for async transformations)
            callback_base_url: Base URL for async transformation callbacks
        """
        self.queue_backend = queue_backend
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.document_root = document_root
        self.num_workers = num_workers
        self.task_repo = task_repo
        self.callback_base_url = callback_base_url

        logger.info(
            f"IngestionPipelineFactory initialized with {num_workers} worker(s)"
        )

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
            index_factory: Index factory for creating vector index
            doc_store: Document store for full documents
            index_id: Identifier for the vector index
            include_source: Whether to configure the source (False for recovery)
            
        Returns:
            Configured pipeline
        """
        logger.info(f"Creating pipeline for job '{job.name}' (ID: {job.id})")

        full_source_path = self.document_root / job.source_path
        connector = FilesystemConnector(
            path=full_source_path,
            filter=None
        )

        directories = await connector.list_directories()

        # Create base pipeline
        pipeline = TrackedIngestionPipeline(
            job=job,
            backend=self.queue_backend,
            job_repo=self.job_repo,
            doc_repo=self.doc_repo,
            task_repo=self.task_repo,
            num_workers=self.num_workers,
            callback_base_url=self.callback_base_url,
            context={"directories": directories}
        )

        # Configure source if requested (skip for recovery scenarios)
        if include_source:
            source = ConnectorSource(
                connector=connector,
                document_root=self.document_root,
            )
            pipeline.source(source)

        pipeline.add_transformation(AsyncParsingTransformation(parser_map=parser_map))
        pipeline.add_transformation(ExtractKeywords())
        pipeline.add_transformation(ChunkingTransformation(chunker=chunker))

        # Add sinks
        pipeline.add_sink(VectorIndexSink(index_factory=index_factory, index_id=index_id))

        logger.info(f"Pipeline configured for job '{job.name}'")
        return pipeline


# Backwards compatibility alias
IIngestionPipelineFactory = IngestionPipelineFactory
