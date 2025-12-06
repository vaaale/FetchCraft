"""
Ingestion API endpoints package.
"""
from fetchcraft.admin.ingestion.api.router import create_ingestion_router
from fetchcraft.admin.ingestion.api.schema import (
    CreateJobRequest,
    JobResponse,
    JobListResponse,
    DocumentResponse,
    DocumentListResponse,
    DirectoryItem,
    DirectoryListResponse,
    RetryResponse,
    MessageResponse,
    MessagesListResponse,
    QueueStatsResponse,
    IngestionStatusResponse,
    CallbackMessage,
    CallbackResponse,
    HealthResponse,
)

__all__ = [
    "create_ingestion_router",
    "CreateJobRequest",
    "JobResponse",
    "JobListResponse",
    "DocumentResponse",
    "DocumentListResponse",
    "DirectoryItem",
    "DirectoryListResponse",
    "RetryResponse",
    "MessageResponse",
    "MessagesListResponse",
    "QueueStatsResponse",
    "IngestionStatusResponse",
    "CallbackMessage",
    "CallbackResponse",
    "HealthResponse",
]
