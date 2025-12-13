"""
Pipeline factory for creating configured ingestion pipelines.

This module provides the base class for creating ingestion pipelines
with a user-friendly interface for customization.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Type, TYPE_CHECKING

from pydantic import BaseModel

from fetchcraft.connector.filesystem import FilesystemConnector
from fetchcraft.index import BaseIndex, IndexFactory, VectorIndex
from fetchcraft.ingestion.interfaces import QueueBackend, Source
from fetchcraft.ingestion.models import IngestionJob
from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline
from fetchcraft.ingestion.repository import JobRepository, DocumentRepository, TaskRepository
from fetchcraft.ingestion.sources import ConnectorSource

if TYPE_CHECKING:
    from fetchcraft.admin.context import HandlerContext

logger = logging.getLogger(__name__)


class DefaultIndexFactory(IndexFactory):
    """Default index factory implementation."""
    
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


class FetchcraftIngestionPipelineFactory(BaseModel):
    """
    Base class for creating ingestion pipelines.
    
    Users should subclass this and implement the `configure_pipeline` method
    to add their custom transformations and sinks. The framework handles
    all the plumbing (queue backend, repositories, etc.).
    
    Subclasses should define their own fields for parser_map, chunker,
    index_factory, etc. as needed for their specific pipeline configuration.
    
    Example:
        class MyPipelineFactory(FetchcraftIngestionPipelineFactory):
            parser_map: Dict[str, DocumentParser]
            chunker: NodeParser
            index_factory: IndexFactory
            
            async def configure_pipeline(self, pipeline: TrackedIngestionPipeline) -> None:
                pipeline.add_transformation(ParsingTransformation(parser_map=self.parser_map))
                pipeline.add_transformation(ExtractKeywords())
                pipeline.add_transformation(ChunkingTransformation(chunker=self.chunker))
                pipeline.add_sink(VectorIndexSink(index_factory=self.index_factory))
    
    Attributes available in subclasses:
        self.document_root: Path - Root directory for documents
        self.queue_backend: QueueBackend - Queue backend for message passing
        self.job_repo: JobRepository - Repository for job tracking
        self.doc_repo: DocumentRepository - Repository for document tracking
        self.task_repo: TaskRepository - Repository for task tracking
    """
    
    def __init__(self, **kwargs):
        """Initialize the pipeline factory."""
        super().__init__(**kwargs)
        # These will be set by the handler during initialization
        self._context: Optional[HandlerContext] = None
        self._document_root: Optional[Path] = None
    
    def initialize(
        self,
        context: "HandlerContext",
        document_root: Path,
    ) -> None:
        """
        Initialize the factory with shared resources.
        
        This is called by the handler during startup. Users should not
        call this method directly.
        
        Args:
            context: Handler context with shared resources
            document_root: Root directory for documents
        """
        self._context = context
        self._document_root = document_root
        logger.info(
            f"PipelineFactory initialized with {context.config.num_workers} worker(s)"
        )
    
    @property
    def queue_backend(self) -> QueueBackend:
        """Get the queue backend."""
        return self._context.queue_backend
    
    @property
    def job_repo(self) -> JobRepository:
        """Get the job repository."""
        return self._context.job_repo
    
    @property
    def doc_repo(self) -> DocumentRepository:
        """Get the document repository."""
        return self._context.doc_repo
    
    @property
    def task_repo(self) -> TaskRepository:
        """Get the task repository."""
        return self._context.task_repo
    
    @property
    def num_workers(self) -> int:
        """Get the number of workers."""
        return self._context.config.num_workers
    
    @property
    def callback_base_url(self) -> str:
        """Get the callback base URL."""
        return self._context.config.callback_base_url
    
    @property
    def document_root(self) -> Path:
        """Get the document root path."""
        return self._document_root
    
    @abstractmethod
    async def create_source(self, ingestion_root: Path, sub_path: Path) -> Source:
        """Create a source for the pipeline."""
        pass

    @abstractmethod
    async def configure_pipeline(self, pipeline: TrackedIngestionPipeline) -> None:
        """
        Configure the pipeline with transformations and sinks.
        
        This method is called after the pipeline is created with all system
        dependencies. Users should add their transformations and sinks here.
        
        Subclasses should define their own fields (parser_map, chunker, etc.)
        and use them in this method.
        
        Args:
            pipeline: Pre-configured pipeline instance
        
        Example:
            async def configure_pipeline(self, pipeline: TrackedIngestionPipeline) -> None:
                pipeline.add_transformation(ParsingTransformation(parser_map=self.parser_map))
                pipeline.add_transformation(ExtractKeywords())
                pipeline.add_transformation(ChunkingTransformation(chunker=self.chunker))
                pipeline.add_sink(VectorIndexSink(index_factory=self.index_factory))
        """
        pass
    
    async def create_pipeline(
        self,
        job: IngestionJob,
        include_source: bool = True,
    ) -> TrackedIngestionPipeline:
        """
        Create a configured ingestion pipeline.
        
        This method creates the pipeline with all system dependencies,
        then calls configure_pipeline for user customization.
        
        Args:
            job: The ingestion job
            include_source: Whether to configure the source (False for recovery)
            
        Returns:
            Configured pipeline
        """
        logger.info(f"Creating pipeline for job '{job.name}' (ID: {job.id})")
        
        ingest_from_path = self._document_root / job.source_path

        # Create base pipeline with system dependencies
        pipeline = TrackedIngestionPipeline(
            job=job,
            backend=self.queue_backend,
            job_repo=self.job_repo,
            doc_repo=self.doc_repo,
            task_repo=self.task_repo,
            num_workers=self.num_workers,
            callback_base_url=self.callback_base_url,
            context={}
        )
        
        # Configure source if requested (skip for recovery scenarios)
        if include_source:
            source = await self.create_source(ingestion_root=self._document_root, sub_path=ingest_from_path)
            pipeline.source(source)
        
        # Call user's configure_pipeline to add transformations and sinks
        await self.configure_pipeline(pipeline)
        
        logger.info(f"Pipeline configured for job '{job.name}'")
        return pipeline
