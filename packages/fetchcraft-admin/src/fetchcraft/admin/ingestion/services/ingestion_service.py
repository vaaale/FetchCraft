"""
Business logic layer for ingestion management.

This service provides high-level operations for managing ingestion jobs
and documents, coordinating between repositories and the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from typing import TYPE_CHECKING

from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion import TrackedIngestionPipeline
from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    JobStatus,
    DocumentStatus,
    utcnow,
)
from fetchcraft.ingestion.repository import (
    JobRepository,
    DocumentRepository,
)
from fetchcraft.node import Node

if TYPE_CHECKING:
    from fetchcraft.admin.ingestion.services.worker_manager import WorkerManager
    from fetchcraft.admin.ingestion.pipeline_factory import FetchcraftIngestionPipelineFactory
    from fetchcraft.ingestion.interfaces import QueueBackend

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for managing ingestion jobs and documents.
    
    This service coordinates between repositories, the pipeline, and
    external components to provide high-level ingestion operations.
    """

    def __init__(
        self,
        job_repo: JobRepository,
        doc_repo: DocumentRepository,
        pipeline_factory: "FetchcraftIngestionPipelineFactory",
        document_root: Path,
        queue_backend: "QueueBackend",
        worker_manager: Optional["WorkerManager"] = None,
    ):
        """
        Initialize the ingestion service.
        
        Args:
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            pipeline_factory: Factory for creating configured pipelines
            document_root: Root directory for documents
            queue_backend: Queue backend for message passing
            worker_manager: Worker manager for registering pipelines (for callback handling)
        """
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.pipeline_factory = pipeline_factory
        self.document_root = document_root
        self.queue_backend = queue_backend
        self.worker_manager = worker_manager
        self._background_tasks: set[asyncio.Task] = set()
        self._pipelines: dict[str, TrackedIngestionPipeline] = {}

        logger.info(f"IngestionService initialized with document root: {document_root}")

    async def get_pipeline_steps(self, job_id: str) -> list[str]:
        """
        Get the current pipeline step configuration.
        
        Returns:
            List of pipeline step names in execution order
        """
        job = await self.job_repo.get_job(job_id)
        names = job.pipeline_steps
        return names

    async def create_job(
        self,
        name: str,
        source_path: str | Path,
    ) -> str:
        """
        Create a new ingestion job.
        
        Args:
            name: Human-readable job name
            source_path: Path to source documents (relative to document_root)
            
        Returns:
            Job ID
        """
        full_source_path = self.document_root / source_path

        if not full_source_path.exists():
            raise ValueError(f"Source path does not exist: {full_source_path}")

        job = IngestionJob(
            name=name or f"Ingestion {utcnow().isoformat()}",
            source_path=str(source_path),
            document_root=str(self.document_root),
            status=JobStatus.PENDING,
        )

        logger.info(f"Creating job '{job.name}' from {source_path}")

        pipeline = await self.pipeline_factory.create_pipeline(
            job=job,
            include_source=True,
        )

        await self.job_repo.create_job(job)
        logger.info(f"Job '{job.name}' persisted with ID: {job.id}")

        self._pipelines[job.id] = pipeline

        if self.worker_manager:
            self.worker_manager.register_pipeline(job.id, pipeline)

        task = asyncio.create_task(self._run_job(pipeline))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        logger.info(f"Job '{job.name}' started in background")
        return job.id

    async def _run_job(self, pipeline: TrackedIngestionPipeline):
        """Run a job in the background."""
        try:
            await pipeline.run_job()
        except Exception as e:
            logger.error(f"Job execution failed: {e}", exc_info=True)
        finally:
            self._pipelines.pop(pipeline.job.id, None)
            if self.worker_manager:
                self.worker_manager.unregister_pipeline(pipeline.job.id)

    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Get a job by ID."""
        return await self.job_repo.get_job(job_id)

    async def stop_job(self, job_id: str) -> bool:
        """Stop a running job."""
        job = await self.job_repo.get_job(job_id)
        if not job:
            logger.warning(f"Cannot stop job {job_id}: Job not found")
            return False

        if job.status not in [JobStatus.RUNNING, JobStatus.PENDING]:
            logger.warning(
                f"Cannot stop job {job_id}: Job is in '{job.status.value}' state."
            )
            return False

        pipeline = self._pipelines.get(job_id)
        if not pipeline:
            logger.warning(f"Cannot stop job {job_id}: Pipeline not found")
            await self.job_repo.update_job_status(
                job_id,
                JobStatus.CANCELLED,
                error_message="Job stopped by user (pipeline not found)"
            )
            return True

        logger.info(f"Stopping job '{job.name}' (ID: {job_id})")

        try:
            await pipeline.shutdown()
            logger.info(f"Pipeline shutdown complete for job '{job.name}'")

            await self.job_repo.update_job_status(
                job_id,
                JobStatus.CANCELLED,
                error_message="Job stopped by user"
            )

            self._pipelines.pop(job_id, None)

            logger.info(f"✅ Job '{job.name}' stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
            await self.job_repo.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=f"Error while stopping job: {e}"
            )
            return False

    async def delete_job(
        self,
        job_id: str,
        delete_documents: bool = True
    ) -> bool:
        """Delete a job and optionally its documents."""
        job = await self.job_repo.get_job(job_id)
        if not job:
            return False

        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(
                f"Cannot delete job in '{job.status.value}' state. "
                "Stop the job first or wait for it to complete."
            )

        logger.info(f"Deleting job '{job.name}' (ID: {job_id})")

        if delete_documents:
            docs = await self.doc_repo.list_documents(job_id, status=None, limit=10000)
            doc_count = len(docs)

            for doc in docs:
                await self.doc_repo.delete_document(doc.id)

            logger.info(f"Deleted {doc_count} document(s) for job {job_id}")

        await self.job_repo.delete_job(job_id)
        logger.info(f"Job '{job.name}' deleted successfully")

        return True

    async def restart_job(self, job_id: str) -> str:
        """Restart a completed or failed job."""
        job = await self.job_repo.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(
                f"Cannot restart job in '{job.status.value}' state."
            )

        logger.info(f"Restarting job '{job.name}' (ID: {job_id})")

        new_job_id = await self.create_job(
            name=f"{job.name} (restarted)",
            source_path=job.source_path,
        )

        logger.info(f"Job '{job.name}' restarted as new job {new_job_id}")
        return new_job_id

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionJob]:
        """List jobs with optional filtering."""
        return await self.job_repo.list_jobs(status, limit, offset)

    async def get_job_documents(
        self,
        job_id: str,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """Get documents for a job."""
        return await self.doc_repo.list_documents(job_id, status, limit, offset)

    async def retry_failed_documents(self, job_id: str) -> int:
        """Retry all failed documents for a job."""
        failed_docs = await self.doc_repo.get_documents_by_status(
            job_id,
            DocumentStatus.FAILED
        )

        count = 0
        for doc in failed_docs:
            await self.doc_repo.update_document_status(
                doc.id,
                DocumentStatus.PENDING,
                current_step=None,
                error_message=None,
                error_step=None
            )

            await self.doc_repo.increment_retry_count(doc.id)

            await self.queue_backend.enqueue(
                "ingest.main",
                body={
                    "type": "document",
                    "doc_id": doc.id,
                    "job_id": job_id,
                    "current_step_idx": 0,
                }
            )
            count += 1

        logger.info(f"Retried {count} failed documents for job {job_id}")
        return count

    async def get_all_documents(
        self,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> tuple[List[DocumentRecord], dict[str, IngestionJob]]:
        """Get all documents across all jobs with optional filtering."""
        docs = await self.doc_repo.list_all_documents(status, limit, offset)

        job_ids = set(doc.job_id for doc in docs)

        jobs_dict = {}
        for job_id in job_ids:
            job = await self.job_repo.get_job(job_id)
            if job:
                jobs_dict[job_id] = job

        return docs, jobs_dict
