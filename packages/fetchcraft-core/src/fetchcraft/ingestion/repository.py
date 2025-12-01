"""
Repository interfaces and implementations for ingestion tracking.

This module follows a repository pattern with interface definitions and
concrete implementations for different storage backends.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from fetchcraft.ingestion.models import IngestionJob, DocumentRecord, JobStatus, DocumentStatus


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


class IJobRepository(ABC):
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


class IDocumentRepository(ABC):
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


class PostgresJobRepository(IJobRepository):
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
                status=JobStatus(row["status"]),
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
                        status=JobStatus(row["status"]),
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
            await conn.execute("DELETE FROM ingestion_jobs WHERE id = $1", job_id)


class PostgresDocumentRepository(IDocumentRepository):
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
                (id, job_id, source, status, current_step, step_statuses,
                 created_at, started_at, completed_at, error_message, 
                 error_step, retry_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                doc.id,
                doc.job_id,
                _sanitize_for_postgres(doc.source),
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
