"""Request and response models for the Admin API."""
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class JobStatusEnum(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Request Models
# =============================================================================

class CreateJobRequest(BaseModel):
    """Request to create a new ingestion job."""
    name: str
    source_path: str  # Relative to document root


class CallbackMessage(BaseModel):
    """
    Generic callback message for async transformations.
    
    This is the standard format for callbacks from external services
    (e.g., docling parsing server) to the admin server.
    """
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

class ParseResponse(BaseModel):
    """Response model for document parsing."""
    filename: str = Field(description="Name of the parsed file")
    success: bool = Field(description="Whether parsing was successful")
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DocumentNode objects as dictionaries"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if parsing failed"
    )
    num_nodes: int = Field(description="Number of nodes created")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class BatchParseResponse(BaseModel):
    """Response model for batch document parsing."""
    results: List[ParseResponse] = Field(description="Results for each file")
    total_files: int = Field(description="Total number of files processed")
    successful: int = Field(description="Number of successfully parsed files")
    failed: int = Field(description="Number of failed files")
    total_nodes: int = Field(description="Total number of nodes created")
    total_processing_time_ms: float = Field(description="Total processing time in milliseconds")


class JobSubmitResponse(BaseModel):
    """Response model for job submission."""
    job_id: str = Field(description="Unique identifier for the submitted job")
    status: JobStatusEnum = Field(description="Initial job status")
    message: str = Field(description="Confirmation message")


class JobStatusResponse(BaseModel):
    """Response model for job status check."""
    job_id: str = Field(description="Job identifier")
    status: JobStatusEnum = Field(description="Current job status")
    submitted_at: float = Field(description="Timestamp when job was submitted")
    started_at: Optional[float] = Field(default=None, description="Timestamp when processing started")
    completed_at: Optional[float] = Field(default=None, description="Timestamp when processing completed")
    error: Optional[str] = Field(default=None, description="Error message if job failed")


class JobResultResponse(BaseModel):
    """Response model for job results."""
    job_id: str = Field(description="Job identifier")
    status: JobStatusEnum = Field(description="Job status")
    results: Optional[BatchParseResponse] = Field(default=None, description="Parsing results if completed")
    error: Optional[str] = Field(default=None, description="Error message if job failed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
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
    status: str  # "running", "stopped", or "error"
    pid: Optional[int]
