"""Request and response models for the Ingestion API."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class JobStatusEnum(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Request Models
# =============================================================================

class CreateJobRequest(BaseModel):
    """Request to create a new ingestion job."""
    name: str
    source_path: str


class CallbackMessage(BaseModel):
    """Generic callback message for async transformations."""
    task_id: str = Field(..., description="Task ID for correlation")
    status: str = Field(..., description="Status: PROCESSING, COMPLETED, or FAILED")
    message: Dict[str, Any] = Field(default_factory=dict, description="Callback payload")
    error: Optional[str] = Field(None, description="Error message if status is FAILED")


class CallbackResponse(BaseModel):
    """Response for callback endpoint."""
    success: bool
    message: str
    task_id: Optional[str] = None


# =============================================================================
# Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    config: Dict[str, Any] = Field(description="Server configuration")
    environment: Dict[str, Any] = Field(description="Environment variables")


class JobResponse(BaseModel):
    """Response model for a single job."""
    id: str
    name: str
    status: str
    source_path: str
    document_root: str
    pipeline_steps: List[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


class JobListResponse(BaseModel):
    """Response model for a list of jobs."""
    jobs: List[JobResponse]
    total: int


class DocumentResponse(BaseModel):
    """Response model for a single document."""
    id: str
    job_id: str
    source: str
    status: str
    current_step: Optional[str]
    step_statuses: Dict[str, str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    error_step: Optional[str]
    retry_count: int


class DocumentListResponse(BaseModel):
    """Response model for a list of documents."""
    documents: List[DocumentResponse]
    total: int


class DirectoryItem(BaseModel):
    """A single directory item."""
    name: str
    path: str
    is_directory: bool


class DirectoryListResponse(BaseModel):
    """Response model for directory listing."""
    items: List[DirectoryItem]
    current_path: str


class RetryResponse(BaseModel):
    """Response for retry operations."""
    retried_count: int


class MessageResponse(BaseModel):
    """Response model for a queue message with document details."""
    id: str
    job_id: str
    job_name: str
    source: str
    status: str
    current_step: Optional[str]
    step_statuses: Dict[str, str]
    pipeline_steps: List[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    error_step: Optional[str]
    retry_count: int


class MessagesListResponse(BaseModel):
    """Response model for a list of queue messages."""
    messages: List[MessageResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics."""
    total_messages: int
    by_state: Dict[str, int]
    by_queue: Dict[str, int]
    failed_messages: int
    oldest_pending: Optional[str]


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion service status."""
    status: str
    pid: Optional[int]
