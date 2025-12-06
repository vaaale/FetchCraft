"""Services package for Fetchcraft Admin."""

from .ingestion_service import IngestionService
from .pipeline_factory import (
    DefaultIngestionPipelineFactory,
    IIngestionPipelineFactory,  # Backwards compatibility
)
from ..interfaces.pipeline import IngestionPipelineFactory
from .job_service import JobService
from .document_service import DocumentService
from .worker_manager import WorkerManager

__all__ = [
    "IngestionService",
    "DefaultIngestionPipelineFactory",
    "IIngestionPipelineFactory",  # Backwards compatibility
    "JobService",
    "DocumentService",
    "WorkerManager",
]
