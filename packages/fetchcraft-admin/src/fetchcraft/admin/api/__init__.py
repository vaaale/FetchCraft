"""API package for the Admin application."""
from fetchcraft.admin.api.schema import (
    # Requests
    CreateJobRequest,
    CallbackMessage,
    CallbackResponse,
    # Responses
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
    HealthResponse,
    JobStatusEnum,
    ParseResponse,
    BatchParseResponse,
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse,
)
from fetchcraft.admin.api.endpoints import main_router, ui_router

__all__ = [
    # Routers
    "main_router",
    "ui_router",
    # Requests
    "CreateJobRequest",
    "CallbackMessage",
    "CallbackResponse",
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
    "HealthResponse",
    "JobStatusEnum",
    "ParseResponse",
    "BatchParseResponse",
    "JobSubmitResponse",
    "JobStatusResponse",
    "JobResultResponse",
]
