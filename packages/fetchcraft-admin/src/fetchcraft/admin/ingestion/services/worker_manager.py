"""
Worker Manager for managing ingestion pipeline workers.

This module provides a WorkerManager that:
- Starts automatically with the application
- Manages worker lifecycle for active jobs
- Recovers pending tasks on startup
- Does not block the main thread
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, List, TYPE_CHECKING

from fetchcraft.ingestion import (
    TrackedIngestionPipeline,
    QueueBackend,
    JobRepository,
    DocumentRepository,
    TaskRepository,
    JobStatus,
    DocumentStatus,
)

if TYPE_CHECKING:
    from fetchcraft.admin.ingestion.pipeline_factory import FetchcraftIngestionPipelineFactory

logger = logging.getLogger(__name__)


class WorkerManager:
    """
    Manages worker lifecycle for ingestion pipelines.
    
    The WorkerManager:
    - Starts automatically when the application starts
    - Maintains a registry of active pipelines by job_id
    - Recovers pending jobs and tasks on startup
    - Provides methods for starting/stopping workers for specific jobs
    - Does not block the main thread
    """
    
    def __init__(
        self,
        queue_backend: QueueBackend,
        job_repo: JobRepository,
        doc_repo: DocumentRepository,
        task_repo: TaskRepository,
        pipeline_factory: "FetchcraftIngestionPipelineFactory",
        callback_base_url: str = "",
    ):
        """
        Initialize the WorkerManager.
        
        Args:
            queue_backend: Queue backend for message passing
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            task_repo: Repository for task tracking
            pipeline_factory: Factory for creating pipelines
            callback_base_url: Base URL for async transformation callbacks
        """
        self.queue_backend = queue_backend
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.task_repo = task_repo
        self.pipeline_factory = pipeline_factory
        self.callback_base_url = callback_base_url
        
        self._pipelines: Dict[str, TrackedIngestionPipeline] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False
        
        logger.info("WorkerManager initialized")
    
    async def start(self) -> int:
        """
        Start the WorkerManager and recover any pending jobs.
        
        Returns:
            Number of jobs recovered
        """
        if self._running:
            logger.warning("WorkerManager already running")
            return 0
        
        self._running = True
        logger.info("🚀 Starting WorkerManager...")
        
        running_jobs = await self.job_repo.list_jobs(status=JobStatus.RUNNING, limit=1000)
        pending_jobs = await self.job_repo.list_jobs(status=JobStatus.PENDING, limit=1000)
        
        jobs_to_recover = running_jobs + pending_jobs
        
        if not jobs_to_recover:
            logger.info("✓ No jobs to recover")
            return 0
        
        logger.info(f"📋 Found {len(jobs_to_recover)} job(s) to recover:")
        for job in jobs_to_recover:
            logger.info(f"   - {job.name} (ID: {job.id}, Status: {job.status})")
        
        recovered = 0
        for job in jobs_to_recover:
            try:
                await self._recover_job(job)
                recovered += 1
            except Exception as e:
                logger.error(f"Failed to recover job '{job.name}': {e}", exc_info=True)
                await self.job_repo.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message=f"Recovery failed: {str(e)}"
                )
        
        logger.info(f"✅ WorkerManager started, recovered {recovered} job(s)")
        return recovered
    
    async def _recover_job(self, job):
        """Recover a single job by creating pipeline and starting workers."""
        logger.info(f"Recovering job '{job.name}' (ID: {job.id})")
        
        pipeline = await self.pipeline_factory.create_pipeline(
            job=job,
            include_source=False,
        )
        
        pipeline.task_repo = self.task_repo
        pipeline.callback_base_url = self.callback_base_url
        
        self._pipelines[job.id] = pipeline
        
        if job.status == JobStatus.PENDING:
            await self.job_repo.set_job_started(job.id)
        
        pending_docs = await self.doc_repo.get_documents_by_status(
            job.id,
            DocumentStatus.PENDING
        )
        
        if pending_docs:
            logger.info(f"[Recovery] Found {len(pending_docs)} PENDING documents to re-enqueue")
            
            pipeline_steps = job.pipeline_steps
            for doc in pending_docs:
                next_step_idx = self._get_next_step_index(doc, pipeline_steps)
                
                await self.queue_backend.enqueue(
                    "ingest.main",
                    body={
                        "type": "document",
                        "doc_id": doc.id,
                        "job_id": job.id,
                        "current_step_idx": next_step_idx,
                    }
                )
                logger.debug(f"[Recovery] Re-enqueued document {doc.id} at step index {next_step_idx}")
        
        task = asyncio.create_task(self._run_job_workers(pipeline))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info(f"Job '{job.name}' recovery initiated")
    
    def _get_next_step_index(self, doc, pipeline_steps: List[str]) -> int:
        """Determine the next step index for a document based on its step_statuses."""
        for idx, step_name in enumerate(pipeline_steps):
            status = doc.step_statuses.get(step_name, "pending")
            if status == "pending":
                return idx
        return 0
    
    async def _run_job_workers(self, pipeline: TrackedIngestionPipeline):
        """Run workers for a job until completion."""
        try:
            await pipeline._start_workers()
            logger.info(f"[Recovery] Workers started for job: {pipeline.job.name}")
            
            await pipeline._wait_until_idle()
            logger.info(f"[Recovery] All work completed for job: {pipeline.job.name}")
            
            await self.job_repo.set_job_completed(pipeline.job.id)
            logger.info(f"[Recovery] ✅ Job '{pipeline.job.name}' completed successfully")
            
        except Exception as e:
            logger.error(f"[Recovery] ❌ Job '{pipeline.job.name}' failed: {e}", exc_info=True)
            await self.job_repo.update_job_status(
                pipeline.job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )
        finally:
            await pipeline.shutdown()
            self._pipelines.pop(pipeline.job.id, None)
    
    async def stop(self):
        """Stop the WorkerManager and all active workers."""
        if not self._running:
            return
        
        logger.info("Stopping WorkerManager...")
        self._running = False
        
        for job_id, pipeline in list(self._pipelines.items()):
            try:
                await pipeline.shutdown()
            except Exception as e:
                logger.error(f"Error stopping pipeline for job {job_id}: {e}")
        
        self._pipelines.clear()
        
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("WorkerManager stopped")
    
    def get_pipeline(self, job_id: str) -> Optional[TrackedIngestionPipeline]:
        """Get the pipeline for a specific job."""
        return self._pipelines.get(job_id)
    
    def get_active_job_ids(self) -> List[str]:
        """Get list of active job IDs."""
        return list(self._pipelines.keys())
    
    def register_pipeline(self, job_id: str, pipeline: TrackedIngestionPipeline) -> None:
        """Register a pipeline for callback handling."""
        self._pipelines[job_id] = pipeline
        logger.debug(f"Registered pipeline for job {job_id}")
    
    def unregister_pipeline(self, job_id: str) -> None:
        """Unregister a pipeline."""
        if job_id in self._pipelines:
            del self._pipelines[job_id]
            logger.debug(f"Unregistered pipeline for job {job_id}")
    
    async def handle_task_callback(
        self,
        task_id: str,
        status: str,
        message: dict,
    ) -> bool:
        """
        Handle a callback for an async task.
        
        Args:
            task_id: The task ID from the callback
            status: Callback status ('PROCESSING', 'COMPLETED', 'FAILED')
            message: Callback payload
            
        Returns:
            True if callback was handled successfully
        """
        task = await self.task_repo.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        pipeline = self._pipelines.get(task.job_id)
        if not pipeline:
            logger.error(f"No active pipeline for job {task.job_id}")
            return False
        
        return await pipeline.handle_task_callback(task_id, status, message)
