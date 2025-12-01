"""API models for requests and responses."""
from fetchcraft.admin.api.models.requests import (
    CreateJobRequest,
    DoclingNodeCallback,
    DoclingCompletionCallback,
)
from fetchcraft.admin.api.models.responses import (
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
)

__all__ = [
    # Requests
    "CreateJobRequest",
    "DoclingNodeCallback",
    "DoclingCompletionCallback",
    # Responses
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
]
