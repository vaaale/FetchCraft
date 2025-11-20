"""
Pipeline factory for creating configured ingestion pipelines.

This factory encapsulates the logic for building pipelines with
the appropriate transformations and sinks based on configuration.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from fetchcraft.connector.filesystem import FilesystemConnector
from fetchcraft.document_store import DocumentStore
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.interfaces import IQueueBackend
from fetchcraft.ingestion.models import IngestionJob
from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline
from fetchcraft.ingestion.repository import IJobRepository, IDocumentRepository
from fetchcraft.ingestion.sinks import VectorIndexSink, DocumentStoreSink
from fetchcraft.ingestion.sources import ConnectorSource
from fetchcraft.ingestion.transformations import (
    ParsingTransformation,
    ChunkingTransformation,
    ExtractKeywords,
)
from fetchcraft.node import Node
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser

logger = logging.getLogger(__name__)


class IIngestionPipelineFactory(ABC):
    """Interface for creating ingestion pipelines."""
    
    @abstractmethod
    def create_pipeline(
        self,
        job: IngestionJob,
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        vector_index: VectorIndex[Node],
        doc_store: DocumentStore,
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


class IngestionPipelineFactory(IIngestionPipelineFactory):
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
        queue_backend: IQueueBackend,
        job_repo: IJobRepository,
        doc_repo: IDocumentRepository,
        document_root: Path,
        num_workers: int = 4,
    ):
        """
        Initialize the pipeline factory.
        
        Args:
            queue_backend: Queue backend for pipeline
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            document_root: Root directory for documents
            num_workers: Number of concurrent workers
        """
        self.queue_backend = queue_backend
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.document_root = document_root
        self.num_workers = num_workers
        
        logger.info(
            f"IngestionPipelineFactory initialized with {num_workers} worker(s)"
        )
    
    def create_pipeline(
        self,
        job: IngestionJob,
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        vector_index: VectorIndex[Node],
        doc_store: DocumentStore,
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
        logger.info(f"Creating pipeline for job '{job.name}' (ID: {job.id})")
        
        # Create base pipeline
        pipeline = TrackedIngestionPipeline(
            job=job,
            backend=self.queue_backend,
            job_repo=self.job_repo,
            doc_repo=self.doc_repo,
            num_workers=self.num_workers,
        )
        
        # Configure source if requested (skip for recovery scenarios)
        if include_source:
            full_source_path = self.document_root / job.source_path
            
            connector = FilesystemConnector(
                path=full_source_path,
                filter=None
            )
            
            source = ConnectorSource(
                connector=connector,
                document_root=self.document_root,
            )
            
            pipeline.source(source)
        
        # Add transformations
        pipeline.add_transformation(ParsingTransformation(parser_map=parser_map))
        pipeline.add_transformation(ExtractKeywords())
        pipeline.add_transformation(ChunkingTransformation(chunker=chunker))
        
        # Add sinks
        pipeline.add_sink(VectorIndexSink(vector_index=vector_index, index_id=index_id))
        
        logger.info(f"Pipeline configured for job '{job.name}'")
        return pipeline
