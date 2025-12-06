"""Services package for Fetchcraft Admin."""

from .ingestion_service import IngestionService
from .pipeline_factory import (
    IngestionPipelineFactoryInterface,
    IngestionPipelineFactory,
    IIngestionPipelineFactory,  # Backwards compatibility
)
from .job_service import JobService
from .document_service import DocumentService
from .worker_manager import WorkerManager

__all__ = [
    "IngestionService",
    "IngestionPipelineFactoryInterface",
    "IngestionPipelineFactory",
    "IIngestionPipelineFactory",  # Backwards compatibility
    "JobService",
    "DocumentService",
    "WorkerManager",
]
