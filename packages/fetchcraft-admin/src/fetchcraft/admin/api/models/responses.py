"""Response models for the Admin API."""
import time
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class JobStatusEnum(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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


class Job:
    """Represents a parsing job."""

    def __init__(
        self,
        job_id: str,
        files: List[str],
        status: JobStatusEnum = JobStatusEnum.PENDING,
        submitted_at: Optional[float] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        results: Optional[BatchParseResponse] = None,
        error: Optional[str] = None,
        callback_url: Optional[str] = None
    ):
        self.job_id = job_id
        self.files = files
        self.status = status
        self.submitted_at = submitted_at or time.time()
        self.started_at = started_at
        self.completed_at = completed_at
        self.results = results
        self.error = error
        self.callback_url = callback_url

    def to_dict(self) -> dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "files": self.files,
            "status": self.status.value if isinstance(self.status, JobStatusEnum) else self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "results": self.results.model_dump() if self.results else None,
            "callback_url": self.callback_url
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create job from dictionary."""
        results = None
        if data.get("results"):
            results = BatchParseResponse(**data["results"])

        return cls(
            job_id=data["job_id"],
            files=data.get("files", []),
            status=JobStatusEnum(data["status"]),
            submitted_at=data["submitted_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            results=results,
            error=data.get("error"),
            callback_url=data.get("callback_url")
        )
