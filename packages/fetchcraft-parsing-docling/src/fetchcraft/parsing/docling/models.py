"""
Pydantic models for the Docling parsing API.
"""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class UnsupportedFileFormatError(Exception):
    """Exception raised when an unsupported file format is encountered."""
    def __init__(self, message: str, filename: str):
        super().__init__(message)
        self.filename = filename


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
