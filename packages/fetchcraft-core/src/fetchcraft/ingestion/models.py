"""
Data models for ingestion pipeline tracking.

This module defines the core data structures for tracking ingestion jobs
and documents as they flow through the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Dict, List
from uuid import uuid4


def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(tz=timezone.utc)


class JobStatus(str, Enum):
    """Status of an ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentStatus(str, Enum):
    """Status of a document in the pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionJob:
    """
    Represents an ingestion job.
    
    An ingestion job tracks a batch of documents being processed through
    the pipeline. It contains metadata about the job execution and references
    to all documents being processed.
    
    Attributes:
        id: Unique job identifier
        name: Human-readable job name
        status: Current job status
        source_path: Path to source documents (relative to document root)
        document_root: Root directory for all documents
        pipeline_steps: Ordered list of pipeline step names
        created_at: Job creation timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
        error_message: Error message if job failed
        metadata: Additional job metadata
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    source_path: str = ""
    document_root: str = ""
    pipeline_steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "source_path": self.source_path,
            "document_root": self.document_root,
            "pipeline_steps": self.pipeline_steps,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class DocumentRecord:
    """
    Represents a document being processed in the pipeline.
    
    Tracks a single document's progress through each step of the pipeline,
    including timing information and errors.
    
    Attributes:
        id: Unique document identifier
        job_id: ID of the parent ingestion job
        source: Document source path or identifier (displayed in UI)
        status: Current document status
        current_step: Name of the current pipeline step
        step_statuses: Status of each pipeline step
        created_at: Document record creation timestamp
        started_at: Processing start timestamp
        completed_at: Processing completion timestamp
        error_message: Error message if processing failed
        error_step: Step where error occurred
        retry_count: Number of retry attempts
        metadata: Additional document metadata
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    source: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    current_step: Optional[str] = None
    step_statuses: Dict[str, str] = field(default_factory=dict)  # step_name -> "pending" | "processing" | "completed" | "failed"
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_step: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "source": self.source,
            "status": self.status.value,
            "current_step": self.current_step,
            "step_statuses": self.step_statuses,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_step": self.error_step,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }
