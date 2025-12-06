"""Domain models for the Admin application."""
import time
from typing import List, Optional

from fetchcraft.admin.api.schema import BatchParseResponse, JobStatusEnum


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
