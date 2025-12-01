"""
Repository layer for job persistence.

This module defines the interface and implementations for job storage.
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from ..models import BatchParseResponse, JobStatusEnum


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


class JobRepository(ABC):
    """Abstract interface for job persistence."""
    
    @abstractmethod
    def save_job(self, job: Job) -> None:
        """Save a single job."""
        pass
    
    @abstractmethod
    def save_all_jobs(self, jobs: Dict[str, Job]) -> None:
        """Save all jobs."""
        pass
    
    @abstractmethod
    def load_all_jobs(self) -> Dict[str, Job]:
        """Load all jobs from storage."""
        pass
    
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        pass
    
    @abstractmethod
    def save_file(self, job_id: str, filename: str, content: bytes) -> Path:
        """Save a file for a job."""
        pass
    
    @abstractmethod
    def get_job_directory(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        pass
    
    @abstractmethod
    def file_exists(self, job_id: str, filename: str) -> bool:
        """Check if a file exists for a job."""
        pass


class FileSystemJobRepository(JobRepository):
    """File system based implementation of JobRepository."""
    
    def __init__(self, data_dir: str, jobs_file: str):
        """
        Initialize the file system repository.
        
        Args:
            data_dir: Base directory for data storage
            jobs_file: Name of the JSON file for job metadata
        """
        self.data_dir = Path(data_dir)
        self.jobs_file = jobs_file
        self._ensure_data_dir()
    
    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_jobs_file_path(self) -> Path:
        """Get the path to the jobs JSON file."""
        return self.data_dir / self.jobs_file
    
    def get_job_directory(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        job_dir = self.data_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir
    
    def save_job(self, job: Job) -> None:
        """Save a single job by reloading and saving all jobs."""
        jobs = self.load_all_jobs()
        jobs[job.job_id] = job
        self.save_all_jobs(jobs)
    
    def save_all_jobs(self, jobs: Dict[str, Job]) -> None:
        """Save all jobs to JSON file."""
        jobs_file = self._get_jobs_file_path()
        jobs_data = {
            job_id: job.to_dict()
            for job_id, job in jobs.items()
        }
        
        try:
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            print(f"💾 Saved {len(jobs_data)} jobs to {jobs_file}")
        except Exception as e:
            print(f"⚠️  Error saving jobs: {e}")
    
    def load_all_jobs(self) -> Dict[str, Job]:
        """Load all jobs from JSON file."""
        jobs_file = self._get_jobs_file_path()
        
        if not jobs_file.exists():
            print("📂 No existing jobs file found")
            return {}
        
        try:
            with open(jobs_file, 'r') as f:
                jobs_data = json.load(f)
            
            jobs = {}
            for job_id, job_dict in jobs_data.items():
                jobs[job_id] = Job.from_dict(job_dict)
            
            print(f"📂 Loaded {len(jobs)} jobs from {jobs_file}")
            return jobs
        except Exception as e:
            print(f"⚠️  Error loading jobs: {e}")
            return {}
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        jobs = self.load_all_jobs()
        return jobs.get(job_id)
    
    def save_file(self, job_id: str, filename: str, content: bytes) -> Path:
        """
        Save a file for a job.
        
        Args:
            job_id: The job ID
            filename: Original filename
            content: File content bytes
            
        Returns:
            Path to the saved file
        """
        job_dir = self.get_job_directory(job_id)
        file_path = job_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return file_path
    
    def file_exists(self, job_id: str, filename: str) -> bool:
        """Check if a file exists for a job."""
        job_dir = self.get_job_directory(job_id)
        file_path = job_dir / filename
        return file_path.exists()
