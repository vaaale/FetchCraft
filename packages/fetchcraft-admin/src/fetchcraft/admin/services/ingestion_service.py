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

from fetchcraft.document_store import DocumentStore
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    JobStatus,
    DocumentStatus,
    utcnow,
)
from fetchcraft.ingestion.repository import (
    IJobRepository,
    IDocumentRepository,
)
from fetchcraft.node import Node
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser
from .pipeline_factory import IIngestionPipelineFactory
from fetchcraft.ingestion import TrackedIngestionPipeline

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for managing ingestion jobs and documents.
    
    This service coordinates between repositories, the pipeline, and
    external components to provide high-level ingestion operations.
    """

    def __init__(
        self,
        job_repo: IJobRepository,
        doc_repo: IDocumentRepository,
        pipeline_factory: IIngestionPipelineFactory,
        document_root: Path,
    ):
        """
        Initialize the ingestion service.
        
        Args:
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            pipeline_factory: Factory for creating configured pipelines
            document_root: Root directory for documents
        """
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.pipeline_factory = pipeline_factory
        self.document_root = document_root
        self._background_tasks: set[asyncio.Task] = set()  # Track background tasks to prevent GC
        self._pipelines: dict[str, TrackedIngestionPipeline] = {}  # Track pipelines by job_id

        logger.info(f"IngestionService initialized with document root: {document_root}")

    async def get_pipeline_steps(self, job_id: str) -> list[str]:
        """
        Get the current pipeline step configuration.
        
        This defines the canonical pipeline steps that all jobs use.
        Update this method when you add or remove pipeline steps.
        
        NOTE: These names must match the transformation class names
        (or their get_name() return values) to properly track step statuses.
        
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
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        vector_index: VectorIndex[Node],
        doc_store: DocumentStore,
        index_id: str = "default",
    ) -> str:
        """
        Create a new ingestion job.
        
        Args:
            name: Human-readable job name
            source_path: Path to source documents (relative to document_root)
            parser_map: Map of mimetype to parser
            chunker: Node parser for chunking
            vector_index: Vector index for storing chunks
            doc_store: Document store for full documents
            index_id: Identifier for the vector index
            
        Returns:
            Job ID
        """
        # Resolve source path (make it relative to document_root)
        full_source_path = self.document_root / source_path

        if not full_source_path.exists():
            raise ValueError(f"Source path does not exist: {full_source_path}")

        # Create job
        job = IngestionJob(
            name=name or f"Ingestion {utcnow().isoformat()}",
            source_path=str(source_path),
            document_root=str(self.document_root),
            status=JobStatus.PENDING,
        )

        logger.info(f"Creating job '{job.name}' from {source_path}")

        # Build pipeline using factory
        pipeline = self.pipeline_factory.create_pipeline(
            job=job,
            parser_map=parser_map,
            chunker=chunker,
            vector_index=vector_index,
            doc_store=doc_store,
            index_id=index_id,
            include_source=True,
        )

        # Persist job to database before starting background task
        await self.job_repo.create_job(job)
        logger.info(f"Job '{job.name}' persisted with ID: {job.id}")

        # Track pipeline for this job
        self._pipelines[job.id] = pipeline

        # Start job in background
        # Track the task to prevent garbage collection
        task = asyncio.create_task(self._run_job(pipeline))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        logger.info(f"Job '{job.name}' started in background")
        return job.id

    async def _run_job(self, pipeline: TrackedIngestionPipeline):
        """
        Run a job in the background.
        
        Args:
            pipeline: The configured pipeline to run
        """
        try:
            await pipeline.run_job()
        except Exception as e:
            logger.error(f"Job execution failed: {e}", exc_info=True)
        finally:
            # Remove pipeline from tracking when job completes
            self._pipelines.pop(pipeline.job.id, None)

    async def recover_running_jobs(
        self,
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        vector_index: VectorIndex[Node],
        doc_store: DocumentStore,
        index_id: str = "default",
    ) -> int:
        """
        Recover jobs that were running when the server crashed/restarted.
        
        This method:
        1. Finds jobs with status 'running' or 'pending'
        2. Reconstructs pipelines with proper transformations and sinks
        3. Starts workers to resume processing queued documents
        4. Marks jobs as running if they were pending
        
        Args:
            parser_map: Map of mimetype to parser
            chunker: Node parser for chunking
            vector_index: Vector index for storing chunks
            doc_store: Document store for full documents
            index_id: Identifier for the vector index
        
        Returns:
            Number of jobs recovered
        """
        logger.info("🔍 Checking for jobs to recover...")

        # Find jobs that were running or pending
        running_jobs = await self.job_repo.list_jobs(status=JobStatus.RUNNING, limit=1000)
        pending_jobs = await self.job_repo.list_jobs(status=JobStatus.PENDING, limit=1000)

        jobs_to_recover = running_jobs + pending_jobs

        if not jobs_to_recover:
            logger.info("✓ No jobs to recover")
            return 0

        logger.info(f"📋 Found {len(jobs_to_recover)} job(s) to recover:")
        for job in jobs_to_recover:
            logger.info(f"   - {job.name} (ID: {job.id}, Status: {job.status})")

        for job in jobs_to_recover:
            try:
                logger.info(f"Recovering job '{job.name}' (ID: {job.id})")

                # Create pipeline with full configuration using factory
                # We need transformations and sinks even though documents are already queued
                # Don't include source since documents are already queued
                pipeline = self.pipeline_factory.create_pipeline(
                    job=job,
                    parser_map=parser_map,
                    chunker=chunker,
                    vector_index=vector_index,
                    doc_store=doc_store,
                    index_id=index_id,
                    include_source=False,
                )

                # Mark job as running if it was pending
                if job.status == JobStatus.PENDING:
                    await self.job_repo.set_job_started(job.id)

                # Track pipeline for this job
                self._pipelines[job.id] = pipeline

                # Start workers in background to process queued documents
                # Track the task to prevent garbage collection
                task = asyncio.create_task(self._recover_job(pipeline))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

                logger.info(f"Job '{job.name}' recovery initiated")

            except Exception as e:
                logger.error(f"Failed to recover job '{job.name}': {e}", exc_info=True)
                # Mark job as failed
                await self.job_repo.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message=f"Recovery failed: {str(e)}"
                )

        return len(jobs_to_recover)

    async def _recover_job(self, pipeline: TrackedIngestionPipeline):
        """
        Recover a job by starting workers to process existing queued documents.
        
        Args:
            pipeline: The pipeline with job context
        """
        try:
            logger.info(f"[Recovery] Starting workers for job: {pipeline.job.name}")

            # Start workers
            await pipeline._start_workers()
            logger.info(f"[Recovery] Workers started for job: {pipeline.job.name}")

            # Wait until all queued work is processed
            logger.info(f"[Recovery] Waiting for queued work to complete for job: {pipeline.job.name}")
            await pipeline._wait_until_idle()
            logger.info(f"[Recovery] All queued work completed for job: {pipeline.job.name}")

            # Mark job as completed
            await pipeline.job_repo.set_job_completed(pipeline.job.id)
            logger.info(f"[Recovery] ✅ Job '{pipeline.job.name}' recovered and completed successfully")

        except Exception as e:
            logger.error(f"[Recovery] ❌ Job '{pipeline.job.name}' recovery failed: {e}", exc_info=True)
            await pipeline.job_repo.update_job_status(
                pipeline.job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )
        finally:
            logger.info(f"[Recovery] Shutting down pipeline for job: {pipeline.job.name}")
            await pipeline.shutdown()
            logger.info(f"[Recovery] Pipeline shutdown complete for job: {pipeline.job.name}")
            # Remove pipeline from tracking
            self._pipelines.pop(pipeline.job.id, None)

    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """
        Get a job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            The job if found, None otherwise
        """
        return await self.job_repo.get_job(job_id)

    async def stop_job(self, job_id: str) -> bool:
        """
        Stop a running job.
        
        This method stops the pipeline workers and marks the job as cancelled.
        
        Args:
            job_id: The job ID to stop
            
        Returns:
            True if job was stopped, False if job not found or not running
        """
        # Check if job exists
        job = await self.job_repo.get_job(job_id)
        if not job:
            logger.warning(f"Cannot stop job {job_id}: Job not found")
            return False

        # Only allow stopping running or pending jobs
        if job.status not in [JobStatus.RUNNING, JobStatus.PENDING]:
            logger.warning(
                f"Cannot stop job {job_id}: Job is in '{job.status.value}' state. "
                "Only running or pending jobs can be stopped."
            )
            return False

        # Check if pipeline is tracked
        pipeline = self._pipelines.get(job_id)
        if not pipeline:
            logger.warning(f"Cannot stop job {job_id}: Pipeline not found in active pipelines")
            # Update status anyway in case it's stuck
            await self.job_repo.update_job_status(
                job_id,
                JobStatus.CANCELLED,
                error_message="Job stopped by user (pipeline not found)"
            )
            return True

        logger.info(f"Stopping job '{job.name}' (ID: {job_id})")

        try:
            # Shutdown the pipeline (stops workers gracefully)
            await pipeline.shutdown()
            logger.info(f"Pipeline shutdown complete for job '{job.name}'")

            # Update job status to cancelled
            await self.job_repo.update_job_status(
                job_id,
                JobStatus.CANCELLED,
                error_message="Job stopped by user"
            )

            # Remove from tracking
            self._pipelines.pop(job_id, None)

            logger.info(f"✅ Job '{job.name}' stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
            # Still try to update status
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
        """
        Delete a job and optionally its documents.
        
        Args:
            job_id: The job ID to delete
            delete_documents: Whether to also delete associated documents
            
        Returns:
            True if deleted successfully, False if job not found
        """
        job = await self.job_repo.get_job(job_id)
        if not job:
            return False

        # Only allow deletion of completed, failed, or cancelled jobs
        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(
                f"Cannot delete job in '{job.status.value}' state. "
                "Stop the job first or wait for it to complete."
            )

        logger.info(f"Deleting job '{job.name}' (ID: {job_id})")

        # Delete associated documents if requested
        if delete_documents:
            # Get count for logging
            docs = await self.doc_repo.list_documents(job_id, status=None, limit=10000)
            doc_count = len(docs)

            # Delete all documents for this job
            for doc in docs:
                await self.doc_repo.delete_document(doc.id)

            logger.info(f"Deleted {doc_count} document(s) for job {job_id}")

        # Delete the job
        await self.job_repo.delete_job(job_id)
        logger.info(f"Job '{job.name}' deleted successfully")

        return True

    async def restart_job(
        self,
        job_id: str,
        parser_map: dict[str, DocumentParser],
        chunker: NodeParser,
        vector_index: VectorIndex[Node],
        doc_store: DocumentStore,
        index_id: str = "default",
    ) -> str:
        """
        Restart a completed or failed job.

        This creates a new job with the same configuration and processes
        the documents again.

        Args:
            job_id: The job ID to restart
            parser_map: Map of mimetype to parser
            chunker: Node parser for chunking
            vector_index: Vector index for storing chunks
            doc_store: Document store for full documents
            index_id: Identifier for the vector index

        Returns:
            New job ID
        """
        job = await self.job_repo.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Only allow restarting completed or failed jobs
        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(
                f"Cannot restart job in '{job.status.value}' state. "
                "Wait for it to complete or fail first."
            )

        logger.info(f"Restarting job '{job.name}' (ID: {job_id})")

        # Create a new job with the same configuration
        new_job_id = await self.create_job(
            name=f"{job.name} (restarted)",
            source_path=job.source_path,
            parser_map=parser_map,
            chunker=chunker,
            vector_index=vector_index,
            doc_store=doc_store,
            index_id=index_id,
        )

        logger.info(f"Job '{job.name}' restarted as new job {new_job_id}")
        return new_job_id

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionJob]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Offset for pagination

        Returns:
            List of jobs
        """
        return await self.job_repo.list_jobs(status, limit, offset)

    async def get_job_documents(
        self,
        job_id: str,
        status: Optional[DocumentStatus] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        Get documents for a job.

        Args:
            job_id: The job ID
            status: Filter by document status
            limit: Maximum number of documents to return
            offset: Offset for pagination

        Returns:
            List of documents
        """
        return await self.doc_repo.list_documents(job_id, status, limit, offset)

    async def retry_failed_documents(self, job_id: str) -> int:
        """
        Retry all failed documents for a job.

        Args:
            job_id: The job ID

        Returns:
            Number of documents retried
        """
        failed_docs = await self.doc_repo.get_documents_by_status(
            job_id,
            DocumentStatus.FAILED
        )

        count = 0
        for doc in failed_docs:
            # Reset document status
            await self.doc_repo.update_document_status(
                doc.id,
                DocumentStatus.PENDING,
                current_step=None,
                error_message=None,
                error_step=None
            )

            # Increment retry count
            await self.doc_repo.increment_retry_count(doc.id)

            # Re-enqueue document
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
        """
        Get all documents across all jobs with optional filtering.

        Args:
            status: Filter by document status
            limit: Maximum number of documents to return
            offset: Offset for pagination

        Returns:
            Tuple of (documents list, jobs dict keyed by job_id)
        """
        docs = await self.doc_repo.list_all_documents(status, limit, offset)

        # Get unique job IDs
        job_ids = set(doc.job_id for doc in docs)

        # Fetch all jobs
        jobs_dict = {}
        for job_id in job_ids:
            job = await self.job_repo.get_job(job_id)
            if job:
                jobs_dict[job_id] = job

        return docs, jobs_dict
