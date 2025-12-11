"""
Repository interfaces and implementations for ingestion tracking.

This module follows a repository pattern with interface definitions and
concrete implementations for different storage backends.

Naming Convention:
- Interfaces do NOT use 'I' prefix (e.g., JobRepository, DocumentRepository)
- Concrete implementations use descriptive prefixes (e.g., PostgresJobRepository)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    TaskRecord,
    JobStatus,
    DocumentStatus,
    TaskStatus,
)


def _normalize_job_status(status_str: str) -> JobStatus:
    """
    Normalize job status string to JobStatus enum.
    
    Handles legacy status values for backwards compatibility:
    - 'processing' -> JobStatus.RUNNING
    """
    if status_str == "processing":
        return JobStatus.RUNNING
    return JobStatus(status_str)


def _sanitize_for_postgres(obj: Any) -> Any:
    """
    Recursively remove null bytes from strings in an object.
    
    PostgreSQL cannot store null bytes (\u0000) in text fields.
    This function recursively sanitizes strings by removing null bytes.
    """
    if isinstance(obj, str):
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {k: _sanitize_for_postgres(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_postgres(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_postgres(item) for item in obj)
    else:
        return obj


class JobRepository(ABC):
    """
    Interface for ingestion job persistence.
    
    This interface defines the contract for storing and retrieving ingestion
    jobs. Implementations can use different storage backends (PostgreSQL, MongoDB, etc.).
    """
    
    @abstractmethod
    async def create_job(self, job: IngestionJob) -> str:
        """
        Create a new ingestion job.
        
        Args:
            job: The job to create
            
        Returns:
            The created job ID
        """
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            The job if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionJob]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by job status
            limit: Maximum number of jobs to return
            offset: Offset for pagination
            
        Returns:
            List of jobs
        """
        pass
    
    @abstractmethod
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update a job's status.
        
        Args:
            job_id: The job ID
            status: New status
            error_message: Optional error message
        """
        pass
    
    @abstractmethod
    async def set_job_started(self, job_id: str) -> None:
        """Mark job as started with timestamp."""
        pass
    
    @abstractmethod
    async def set_job_completed(self, job_id: str) -> None:
        """Mark job as completed with timestamp."""
        pass
    
    @abstractmethod
    async def delete_job(self, job_id: str) -> None:
        """Delete a job and all associated documents."""
        pass


class DocumentRepository(ABC):
    """
    Interface for document tracking persistence.
    
    This interface defines the contract for storing and retrieving document
    processing records within ingestion jobs.
    """
    
    @abstractmethod
    async def create_document(self, doc: DocumentRecord) -> str:
        """
        Create a new document record.
        
        Args:
            doc: The document record to create
            
        Returns:
            The created document ID
        """
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            The document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_documents(
        self,
        job_id: str,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        List documents for a job with optional filtering.
        
        Args:
            job_id: Filter by job ID
            status: Filter by document status
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    async def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        error_step: Optional[str] = None
    ) -> None:
        """
        Update a document's status.
        
        Args:
            doc_id: The document ID
            status: New status
            current_step: Current pipeline step
            error_message: Optional error message
            error_step: Step where error occurred
        """
        pass
    
    @abstractmethod
    async def update_step_status(
        self,
        doc_id: str,
        step_name: str,
        step_status: str
    ) -> None:
        """
        Update the status of a specific pipeline step.
        
        Args:
            doc_id: The document ID
            step_name: Name of the pipeline step
            step_status: Status ("pending", "processing", "completed", "failed")
        """
        pass
    
    @abstractmethod
    async def update_document_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update a document's metadata.
        
        Args:
            doc_id: The document ID
            metadata: New metadata dictionary
        """
        pass
    
    @abstractmethod
    async def update_document_content(
        self,
        doc_id: str,
        content: Optional[str]
    ) -> None:
        """
        Update a document's content.
        
        Args:
            doc_id: The document ID
            content: Base64-encoded content string
        """
        pass
    
    @abstractmethod
    async def increment_retry_count(self, doc_id: str) -> int:
        """
        Increment document retry count.
        
        Args:
            doc_id: The document ID
            
        Returns:
            New retry count
        """
        pass
    
    @abstractmethod
    async def get_documents_by_status(
        self,
        job_id: str,
        status: DocumentStatus
    ) -> List[DocumentRecord]:
        """
        Get all documents in a specific status for a job.
        
        Args:
            job_id: The job ID
            status: Document status to filter by
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    async def list_all_documents(
        self,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        List all documents across all jobs with optional filtering.
        
        Args:
            status: Filter by document status
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a document.
        
        Args:
            doc_id: The document ID to delete
        """
        pass
    
    @abstractmethod
    async def find_document_by_metadata(
        self,
        key: str,
        value: str,
        additional_filters: Optional[Dict[str, str]] = None
    ) -> Optional[DocumentRecord]:
        """
        Find a document by metadata key-value pair.
        
        Args:
            key: Metadata key to search
            value: Value to match
            additional_filters: Optional additional metadata filters
            
        Returns:
            The document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_job_pipeline_steps(self, job_id: str) -> Optional[List[str]]:
        """
        Get pipeline steps for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            List of pipeline step names or None if job not found
        """
        pass
    
    @abstractmethod
    async def count_children_by_parent(
        self,
        parent_doc_id: str,
        status: Optional[DocumentStatus] = None
    ) -> int:
        """
        Count child documents for a parent document.
        
        Args:
            parent_doc_id: The parent document ID
            status: Optional status filter
            
        Returns:
            Count of child documents
        """
        pass
    
    @abstractmethod
    async def count_documents_by_status(self) -> Dict[str, int]:
        """
        Count all documents grouped by status.
        
        Returns:
            Dictionary mapping status string to count
        """
        pass


class TaskRepository(ABC):
    """
    Interface for task tracking persistence.
    
    Tasks represent individual pipeline step executions for documents.
    This enables fine-grained tracking and async callback correlation.
    """
    
    @abstractmethod
    async def create_task(self, task: TaskRecord) -> str:
        """
        Create a new task record.
        
        Args:
            task: The task record to create
            
        Returns:
            The created task ID
        """
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """
        Retrieve a task by ID.
        
        Args:
            task_id: The task ID
            
        Returns:
            The task if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_task_by_job_and_document(
        self,
        job_id: str,
        document_id: str,
        transformation_name: str
    ) -> Optional[TaskRecord]:
        """
        Find a task by job, document, and transformation.
        
        Args:
            job_id: The job ID
            document_id: The document ID
            transformation_name: Name of the transformation
            
        Returns:
            The task if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_tasks_for_document(
        self,
        document_id: str
    ) -> List[TaskRecord]:
        """
        List all tasks for a document.
        
        Args:
            document_id: The document ID
            
        Returns:
            List of tasks ordered by step_index
        """
        pass
    
    @abstractmethod
    async def list_tasks_for_job(
        self,
        job_id: str,
        status: Optional[TaskStatus] = None
    ) -> List[TaskRecord]:
        """
        List all tasks for a job with optional status filter.
        
        Args:
            job_id: The job ID
            status: Optional status filter
            
        Returns:
            List of tasks
        """
        pass
    
    @abstractmethod
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update a task's status.
        
        Args:
            task_id: The task ID
            status: New status
            error_message: Optional error message
        """
        pass
    
    @abstractmethod
    async def set_task_started(self, task_id: str) -> None:
        """Mark task as started with timestamp."""
        pass
    
    @abstractmethod
    async def set_task_submitted(self, task_id: str) -> None:
        """Mark async task as submitted with timestamp."""
        pass
    
    @abstractmethod
    async def set_task_completed(self, task_id: str) -> None:
        """Mark task as completed with timestamp."""
        pass
    
    @abstractmethod
    async def update_task_metadata(
        self,
        task_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update a task's metadata.
        
        Args:
            task_id: The task ID
            metadata: New metadata dictionary
        """
        pass
    
    @abstractmethod
    async def get_pending_async_tasks(
        self,
        job_id: Optional[str] = None
    ) -> List[TaskRecord]:
        """
        Get all async tasks that are waiting for callbacks.
        
        Args:
            job_id: Optional job ID filter
            
        Returns:
            List of tasks with status SUBMITTED or WAITING
        """
        pass
    
    @abstractmethod
    async def delete_tasks_for_document(self, document_id: str) -> None:
        """
        Delete all tasks for a document.
        
        Args:
            document_id: The document ID
        """
        pass
    
    @abstractmethod
    async def delete_tasks_for_job(self, job_id: str) -> None:
        """
        Delete all tasks for a job.
        
        Args:
            job_id: The job ID
        """
        pass


class PostgresJobRepository(JobRepository):
    """PostgreSQL implementation of job repository."""
    
    def __init__(self, pool):
        """
        Initialize repository with database pool.
        
        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool
    
    async def _ensure_schema(self):
        """Create tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    document_root TEXT NOT NULL,
                    pipeline_steps JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            # Create indexes - handle concurrent creation
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status)"
                )
            except Exception:
                pass  # Index already exists
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON ingestion_jobs(created_at DESC)"
                )
            except Exception:
                pass  # Index already exists
    
    async def create_job(self, job: IngestionJob) -> str:
        await self._ensure_schema()
        
        # Sanitize data to remove null bytes
        sanitized_metadata = _sanitize_for_postgres(job.metadata)
        sanitized_pipeline_steps = _sanitize_for_postgres(job.pipeline_steps)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_jobs 
                (id, name, status, source_path, document_root, pipeline_steps, 
                 created_at, started_at, completed_at, error_message, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                job.id,
                _sanitize_for_postgres(job.name),
                job.status.value,
                _sanitize_for_postgres(job.source_path),
                _sanitize_for_postgres(job.document_root),
                json.dumps(sanitized_pipeline_steps),
                job.created_at,
                job.started_at,
                job.completed_at,
                _sanitize_for_postgres(job.error_message),
                json.dumps(sanitized_metadata),
            )
        return job.id
    
    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ingestion_jobs WHERE id = $1", job_id
            )
            if not row:
                return None
            
            return IngestionJob(
                id=row["id"],
                name=row["name"],
                status=_normalize_job_status(row["status"]),
                source_path=row["source_path"],
                document_root=row["document_root"],
                pipeline_steps=json.loads(row["pipeline_steps"]) if isinstance(row["pipeline_steps"], str) else row["pipeline_steps"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                error_message=row["error_message"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            )
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionJob]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_jobs 
                    WHERE status = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                    """,
                    status.value,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_jobs 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )
            
            jobs = []
            for row in rows:
                jobs.append(
                    IngestionJob(
                        id=row["id"],
                        name=row["name"],
                        status=_normalize_job_status(row["status"]),
                        source_path=row["source_path"],
                        document_root=row["document_root"],
                        pipeline_steps=json.loads(row["pipeline_steps"]) if isinstance(row["pipeline_steps"], str) else row["pipeline_steps"],
                        created_at=row["created_at"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        error_message=row["error_message"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                    )
                )
            return jobs
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs 
                SET status = $1, error_message = $2 
                WHERE id = $3
                """,
                status.value,
                error_message,
                job_id,
            )
    
    async def set_job_started(self, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs 
                SET status = $1, started_at = NOW() 
                WHERE id = $2
                """,
                JobStatus.RUNNING.value,
                job_id,
            )
    
    async def set_job_completed(self, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs 
                SET status = $1, completed_at = NOW() 
                WHERE id = $2
                """,
                JobStatus.COMPLETED.value,
                job_id,
            )
    
    async def delete_job(self, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            # Delete related messages from the queue
            await conn.execute(
                "DELETE FROM messages WHERE body->>'job_id' = $1",
                job_id
            )
            # Delete related tasks
            await conn.execute(
                "DELETE FROM ingestion_tasks WHERE job_id = $1",
                job_id
            )
            # Delete the job itself
            await conn.execute("DELETE FROM ingestion_jobs WHERE id = $1", job_id)


class PostgresDocumentRepository(DocumentRepository):
    """PostgreSQL implementation of document repository."""
    
    def __init__(self, pool):
        """
        Initialize repository with database pool.
        
        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool
    
    async def _ensure_schema(self):
        """Create tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_documents (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    step_statuses JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    error_step TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            # Create indexes - handle concurrent creation
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_docs_job_id ON ingestion_documents(job_id)"
                )
            except Exception:
                pass  # Index already exists
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_docs_status ON ingestion_documents(status)"
                )
            except Exception:
                pass  # Index already exists
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_docs_job_status ON ingestion_documents(job_id, status)"
                )
            except Exception:
                pass  # Index already exists
    
    async def create_document(self, doc: DocumentRecord) -> str:
        await self._ensure_schema()
        
        # Sanitize data to remove null bytes
        sanitized_metadata = _sanitize_for_postgres(doc.metadata)
        sanitized_step_statuses = _sanitize_for_postgres(doc.step_statuses)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_documents 
                (id, job_id, source, content, status, current_step, step_statuses,
                 created_at, started_at, completed_at, error_message, 
                 error_step, retry_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                doc.id,
                doc.job_id,
                _sanitize_for_postgres(doc.source),
                doc.content,
                doc.status.value,
                _sanitize_for_postgres(doc.current_step),
                json.dumps(sanitized_step_statuses),
                doc.created_at,
                doc.started_at,
                doc.completed_at,
                _sanitize_for_postgres(doc.error_message),
                _sanitize_for_postgres(doc.error_step),
                doc.retry_count,
                json.dumps(sanitized_metadata),
            )
        return doc.id
    
    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ingestion_documents WHERE id = $1", doc_id
            )
            if not row:
                return None
            
            return DocumentRecord(
                id=row["id"],
                job_id=row["job_id"],
                source=row["source"],
                content=row["content"],
                status=DocumentStatus(row["status"]),
                current_step=row["current_step"],
                step_statuses=json.loads(row["step_statuses"]) if isinstance(row["step_statuses"], str) else row["step_statuses"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                error_message=row["error_message"],
                error_step=row["error_step"],
                retry_count=row["retry_count"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            )
    
    async def list_documents(
        self,
        job_id: str,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_documents 
                    WHERE job_id = $1 AND status = $2 
                    ORDER BY created_at ASC 
                    LIMIT $3 OFFSET $4
                    """,
                    job_id,
                    status.value,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_documents 
                    WHERE job_id = $1 
                    ORDER BY created_at ASC 
                    LIMIT $2 OFFSET $3
                    """,
                    job_id,
                    limit,
                    offset,
                )
            
            docs = []
            for row in rows:
                docs.append(
                    DocumentRecord(
                        id=row["id"],
                        job_id=row["job_id"],
                        source=row["source"],
                        content=row["content"],
                        status=DocumentStatus(row["status"]),
                        current_step=row["current_step"],
                        step_statuses=json.loads(row["step_statuses"]) if isinstance(row["step_statuses"], str) else row["step_statuses"],
                        created_at=row["created_at"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        error_message=row["error_message"],
                        error_step=row["error_step"],
                        retry_count=row["retry_count"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                    )
                )
            return docs
    
    async def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        error_step: Optional[str] = None
    ) -> None:
        async with self.pool.acquire() as conn:
            updates = ["status = $1"]
            params = [status.value]
            param_idx = 2
            
            if current_step is not None:
                updates.append(f"current_step = ${param_idx}")
                params.append(_sanitize_for_postgres(current_step))
                param_idx += 1
            
            if error_message is not None:
                updates.append(f"error_message = ${param_idx}")
                params.append(_sanitize_for_postgres(error_message))
                param_idx += 1
            
            if error_step is not None:
                updates.append(f"error_step = ${param_idx}")
                params.append(_sanitize_for_postgres(error_step))
                param_idx += 1
            
            # Set started_at if transitioning to processing
            if status == DocumentStatus.PROCESSING:
                updates.append("started_at = COALESCE(started_at, NOW())")
            
            # Set completed_at if transitioning to completed or failed
            if status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED):
                updates.append("completed_at = NOW()")
            
            params.append(doc_id)
            query = f"UPDATE ingestion_documents SET {', '.join(updates)} WHERE id = ${param_idx}"
            await conn.execute(query, *params)
    
    async def update_step_status(
        self,
        doc_id: str,
        step_name: str,
        step_status: str
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_documents 
                SET step_statuses = jsonb_set(
                    COALESCE(step_statuses, '{}'::jsonb),
                    $1::text[],
                    $2::jsonb
                )
                WHERE id = $3
                """,
                [step_name],
                json.dumps(step_status),
                doc_id,
            )
    
    async def update_document_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_documents 
                SET metadata = $1::jsonb
                WHERE id = $2
                """,
                json.dumps(metadata),
                doc_id,
            )
    
    async def update_document_content(
        self,
        doc_id: str,
        content: Optional[str]
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_documents 
                SET content = $1
                WHERE id = $2
                """,
                content,
                doc_id,
            )
    
    async def increment_retry_count(self, doc_id: str) -> int:
        async with self.pool.acquire() as conn:
            new_count = await conn.fetchval(
                """
                UPDATE ingestion_documents 
                SET retry_count = retry_count + 1 
                WHERE id = $1 
                RETURNING retry_count
                """,
                doc_id,
            )
        return new_count
    
    async def get_documents_by_status(
        self,
        job_id: str,
        status: DocumentStatus
    ) -> List[DocumentRecord]:
        return await self.list_documents(job_id, status)
    
    async def list_all_documents(
        self,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """List all documents across all jobs."""
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_documents 
                    WHERE status = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                    """,
                    status.value,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_documents 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )
            
            docs = []
            for row in rows:
                docs.append(
                    DocumentRecord(
                        id=row["id"],
                        job_id=row["job_id"],
                        source=row["source"],
                        content=row["content"],
                        status=DocumentStatus(row["status"]),
                        current_step=row["current_step"],
                        step_statuses=json.loads(row["step_statuses"]) if isinstance(row["step_statuses"], str) else row["step_statuses"],
                        created_at=row["created_at"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        error_message=row["error_message"],
                        error_step=row["error_step"],
                        retry_count=row["retry_count"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                    )
                )
            return docs
    
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM ingestion_documents WHERE id = $1", doc_id)
    
    async def find_document_by_metadata(
        self,
        key: str,
        value: str,
        additional_filters: Optional[Dict[str, str]] = None
    ) -> Optional[DocumentRecord]:
        """Find a document by metadata key-value pair."""
        async with self.pool.acquire() as conn:
            # Build query with additional filters
            query = """
                SELECT * FROM ingestion_documents
                WHERE metadata->>$1 = $2
            """
            params = [key, value]
            param_idx = 3
            
            if additional_filters:
                for filter_key, filter_value in additional_filters.items():
                    query += f" AND metadata->>${param_idx} = ${param_idx + 1}"
                    params.extend([filter_key, filter_value])
                    param_idx += 2
            
            query += " ORDER BY created_at DESC LIMIT 1"
            
            row = await conn.fetchrow(query, *params)
            
            if not row:
                return None
            
            return DocumentRecord(
                id=row["id"],
                job_id=row["job_id"],
                source=row["source"],
                content=row["content"],
                status=DocumentStatus(row["status"]),
                current_step=row["current_step"],
                step_statuses=json.loads(row["step_statuses"]) if isinstance(row["step_statuses"], str) else row["step_statuses"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                error_message=row["error_message"],
                error_step=row["error_step"],
                retry_count=row["retry_count"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            )
    
    async def get_job_pipeline_steps(self, job_id: str) -> Optional[List[str]]:
        """Get pipeline steps for a job."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT pipeline_steps FROM ingestion_jobs WHERE id = $1",
                job_id
            )
            if not row:
                return None
            
            pipeline_steps = row['pipeline_steps']
            if isinstance(pipeline_steps, str):
                return json.loads(pipeline_steps)
            return pipeline_steps
    
    async def count_children_by_parent(
        self,
        parent_doc_id: str,
        status: Optional[DocumentStatus] = None
    ) -> int:
        """Count child documents for a parent document."""
        async with self.pool.acquire() as conn:
            if status:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM ingestion_documents
                    WHERE metadata->>'parent_document_id' = $1
                    AND status = $2
                    """,
                    parent_doc_id,
                    status.value
                )
            else:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM ingestion_documents
                    WHERE metadata->>'parent_document_id' = $1
                    """,
                    parent_doc_id
                )
            return count or 0
    
    async def count_documents_by_status(self) -> Dict[str, int]:
        """Count all documents grouped by status."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, COUNT(*) as count FROM ingestion_documents GROUP BY status"
            )
            return {row["status"]: row["count"] for row in rows}


class PostgresTaskRepository(TaskRepository):
    """PostgreSQL implementation of task repository."""
    
    def __init__(self, pool):
        """
        Initialize repository with database pool.
        
        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool
    
    async def _ensure_schema(self):
        """Create tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_tasks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    transformation_name TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    is_async BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    submitted_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            # Create indexes
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON ingestion_tasks(job_id)"
                )
            except Exception:
                pass
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_document_id ON ingestion_tasks(document_id)"
                )
            except Exception:
                pass
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON ingestion_tasks(status)"
                )
            except Exception:
                pass
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_job_doc_transform ON ingestion_tasks(job_id, document_id, transformation_name)"
                )
            except Exception:
                pass
    
    async def create_task(self, task: TaskRecord) -> str:
        await self._ensure_schema()
        
        sanitized_metadata = _sanitize_for_postgres(task.metadata)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_tasks 
                (id, job_id, document_id, transformation_name, step_index, status,
                 is_async, created_at, started_at, submitted_at, completed_at, 
                 error_message, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                task.id,
                task.job_id,
                task.document_id,
                _sanitize_for_postgres(task.transformation_name),
                task.step_index,
                task.status.value,
                task.is_async,
                task.created_at,
                task.started_at,
                task.submitted_at,
                task.completed_at,
                _sanitize_for_postgres(task.error_message),
                json.dumps(sanitized_metadata),
            )
        return task.id
    
    def _row_to_task(self, row) -> TaskRecord:
        """Convert database row to TaskRecord."""
        return TaskRecord(
            id=row["id"],
            job_id=row["job_id"],
            document_id=row["document_id"],
            transformation_name=row["transformation_name"],
            step_index=row["step_index"],
            status=TaskStatus(row["status"]),
            is_async=row["is_async"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            submitted_at=row["submitted_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
        )
    
    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ingestion_tasks WHERE id = $1", task_id
            )
            if not row:
                return None
            return self._row_to_task(row)
    
    async def get_task_by_job_and_document(
        self,
        job_id: str,
        document_id: str,
        transformation_name: str
    ) -> Optional[TaskRecord]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM ingestion_tasks 
                WHERE job_id = $1 AND document_id = $2 AND transformation_name = $3
                """,
                job_id,
                document_id,
                transformation_name,
            )
            if not row:
                return None
            return self._row_to_task(row)
    
    async def list_tasks_for_document(self, document_id: str) -> List[TaskRecord]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM ingestion_tasks 
                WHERE document_id = $1 
                ORDER BY step_index ASC
                """,
                document_id,
            )
            return [self._row_to_task(row) for row in rows]
    
    async def list_tasks_for_job(
        self,
        job_id: str,
        status: Optional[TaskStatus] = None
    ) -> List[TaskRecord]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_tasks 
                    WHERE job_id = $1 AND status = $2 
                    ORDER BY created_at ASC
                    """,
                    job_id,
                    status.value,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_tasks 
                    WHERE job_id = $1 
                    ORDER BY created_at ASC
                    """,
                    job_id,
                )
            return [self._row_to_task(row) for row in rows]
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_tasks 
                SET status = $1, error_message = $2 
                WHERE id = $3
                """,
                status.value,
                _sanitize_for_postgres(error_message),
                task_id,
            )
    
    async def set_task_started(self, task_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_tasks 
                SET status = $1, started_at = NOW() 
                WHERE id = $2
                """,
                TaskStatus.PROCESSING.value,
                task_id,
            )
    
    async def set_task_submitted(self, task_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_tasks 
                SET status = $1, submitted_at = NOW() 
                WHERE id = $2
                """,
                TaskStatus.SUBMITTED.value,
                task_id,
            )
    
    async def set_task_completed(self, task_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_tasks 
                SET status = $1, completed_at = NOW() 
                WHERE id = $2
                """,
                TaskStatus.COMPLETED.value,
                task_id,
            )
    
    async def update_task_metadata(
        self,
        task_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_tasks 
                SET metadata = $1::jsonb
                WHERE id = $2
                """,
                json.dumps(_sanitize_for_postgres(metadata)),
                task_id,
            )
    
    async def get_pending_async_tasks(
        self,
        job_id: Optional[str] = None
    ) -> List[TaskRecord]:
        async with self.pool.acquire() as conn:
            if job_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_tasks 
                    WHERE job_id = $1 
                    AND is_async = TRUE 
                    AND status IN ($2, $3)
                    ORDER BY created_at ASC
                    """,
                    job_id,
                    TaskStatus.SUBMITTED.value,
                    TaskStatus.WAITING.value,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ingestion_tasks 
                    WHERE is_async = TRUE 
                    AND status IN ($1, $2)
                    ORDER BY created_at ASC
                    """,
                    TaskStatus.SUBMITTED.value,
                    TaskStatus.WAITING.value,
                )
            return [self._row_to_task(row) for row in rows]
    
    async def delete_tasks_for_document(self, document_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ingestion_tasks WHERE document_id = $1",
                document_id,
            )
    
    async def delete_tasks_for_job(self, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ingestion_tasks WHERE job_id = $1",
                job_id,
            )


# Backwards compatibility aliases (deprecated - will be removed in future version)
IJobRepository = JobRepository
IDocumentRepository = DocumentRepository
ITaskRepository = TaskRepository
