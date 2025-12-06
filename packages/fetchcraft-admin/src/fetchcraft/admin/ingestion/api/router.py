"""
Ingestion API router factory.

This module provides a factory function to create the ingestion API router
with injected dependencies.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query

from fetchcraft.ingestion.models import DocumentStatus, JobStatus

from fetchcraft.admin.ingestion.api.schema import (
    CallbackMessage,
    CallbackResponse,
    CreateJobRequest,
    DirectoryItem,
    DirectoryListResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    IngestionStatusResponse,
    JobListResponse,
    JobResponse,
    MessageResponse,
    MessagesListResponse,
    QueueStatsResponse,
    RetryResponse,
)

if TYPE_CHECKING:
    from fetchcraft.admin.ingestion.services.ingestion_service import IngestionService
    from fetchcraft.admin.ingestion.services.worker_manager import WorkerManager
    from fetchcraft.admin.ingestion.config import IngestionConfig
    from fetchcraft.ingestion.interfaces import QueueBackend

logger = logging.getLogger(__name__)


def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime to ISO string."""
    if dt is None:
        return None
    return dt.isoformat()


def create_ingestion_router(
    get_ingestion_service: Callable[[], "IngestionService"],
    get_worker_manager: Callable[[], "WorkerManager"],
    get_queue_backend: Callable[[], "QueueBackend"],
    get_config: Callable[[], "IngestionConfig"],
    get_ingestion_dependencies: Callable[[], dict],
) -> APIRouter:
    """
    Create the ingestion API router with injected dependencies.
    
    Args:
        get_ingestion_service: Function to get the ingestion service
        get_worker_manager: Function to get the worker manager
        get_queue_backend: Function to get the queue backend
        get_config: Function to get the configuration
        get_ingestion_dependencies: Function to get ingestion dependencies (parser_map, chunker, etc.)
    
    Returns:
        Configured APIRouter
    """
    router = APIRouter(tags=["ingestion"])
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    @router.get("/health", response_model=HealthResponse, tags=["General"])
    async def health():
        """Health check endpoint."""
        config = get_config()
        queue_backend = get_queue_backend()
        
        db_healthy = False
        if queue_backend:
            try:
                db_healthy = await queue_backend.check_health()
            except Exception:
                pass
        
        status = "healthy" if db_healthy else "degraded"
        return HealthResponse(
            status=status,
            config=config.model_dump(),
            environment=dict(os.environ),
        )
    
    # =========================================================================
    # Directory Browsing
    # =========================================================================
    
    @router.get("/directories", response_model=DirectoryListResponse)
    async def list_directories(
        path: str = Query("", description="Relative path from document root")
    ):
        """List directories and files in the document root."""
        try:
            config = get_config()
            full_path = config.documents_path / path if path else config.documents_path
            full_path = full_path.resolve()
            
            if not str(full_path).startswith(str(config.documents_path.resolve())):
                raise HTTPException(status_code=400, detail="Path must be within document root")
            
            if not full_path.exists():
                raise HTTPException(status_code=404, detail="Path not found")
            
            if not full_path.is_dir():
                raise HTTPException(status_code=400, detail="Path is not a directory")
            
            items = []
            for item in sorted(full_path.iterdir()):
                try:
                    relative_path = item.relative_to(config.documents_path)
                    items.append(DirectoryItem(
                        name=item.name,
                        path=str(relative_path),
                        is_directory=item.is_dir()
                    ))
                except ValueError:
                    continue
            
            return DirectoryListResponse(
                items=items,
                current_path=str(full_path.relative_to(config.documents_path)) if path else ""
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing directories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Jobs
    # =========================================================================
    
    @router.post("/jobs", response_model=JobResponse)
    async def create_job(request: CreateJobRequest):
        """Create a new ingestion job."""
        ingestion_service = get_ingestion_service()
        config = get_config()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            deps = get_ingestion_dependencies()
            
            job_id = await ingestion_service.create_job(
                name=request.name,
                source_path=request.source_path,
                parser_map=deps["parser_map"],
                chunker=deps["chunker"],
                index_factory=deps["index_factory"],
                index_id=config.index_id,
            )
            
            job = await ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=500, detail="Job created but not found")
            
            return JobResponse(
                id=job.id,
                name=job.name,
                status=job.status.value,
                source_path=job.source_path,
                document_root=job.document_root,
                pipeline_steps=job.pipeline_steps,
                created_at=format_datetime(job.created_at),
                started_at=format_datetime(job.started_at),
                completed_at=format_datetime(job.completed_at),
                error_message=job.error_message,
            )
        
        except Exception as e:
            logger.error(f"Error creating job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/jobs", response_model=JobListResponse)
    async def list_jobs(
        status: Optional[str] = Query(None, description="Filter by status"),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
    ):
        """List all ingestion jobs."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            job_status = JobStatus(status) if status else None
            jobs = await ingestion_service.list_jobs(job_status, limit, offset)
            
            job_responses = [
                JobResponse(
                    id=job.id,
                    name=job.name,
                    status=job.status.value,
                    source_path=job.source_path,
                    document_root=job.document_root,
                    pipeline_steps=job.pipeline_steps,
                    created_at=format_datetime(job.created_at),
                    started_at=format_datetime(job.started_at),
                    completed_at=format_datetime(job.completed_at),
                    error_message=job.error_message,
                )
                for job in jobs
            ]
            
            return JobListResponse(jobs=job_responses, total=len(job_responses))
        
        except Exception as e:
            logger.error(f"Error listing jobs: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str):
        """Get details of a specific job."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            job = await ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return JobResponse(
                id=job.id,
                name=job.name,
                status=job.status.value,
                source_path=job.source_path,
                document_root=job.document_root,
                pipeline_steps=job.pipeline_steps,
                created_at=format_datetime(job.created_at),
                started_at=format_datetime(job.started_at),
                completed_at=format_datetime(job.completed_at),
                error_message=job.error_message,
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        delete_documents: bool = Query(True, description="Delete associated documents")
    ):
        """Delete a job and optionally its documents."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            deleted = await ingestion_service.delete_job(job_id, delete_documents)
            if not deleted:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return {"message": "Job deleted successfully", "job_id": job_id}
        
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/jobs/{job_id}/stop")
    async def stop_job(job_id: str):
        """Stop a running or pending job."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            stopped = await ingestion_service.stop_job(job_id)
            
            if not stopped:
                job = await ingestion_service.get_job(job_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Job cannot be stopped. Current status: {job.status.value}"
                    )
            
            job = await ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found after stopping")
            
            return {
                "message": f"Job '{job.name}' stopped successfully",
                "job_id": job_id,
                "status": job.status.value
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error stopping job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/jobs/{job_id}/restart", response_model=JobResponse)
    async def restart_job(job_id: str):
        """Restart a completed or failed job."""
        ingestion_service = get_ingestion_service()
        config = get_config()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            deps = get_ingestion_dependencies()
            
            new_job_id = await ingestion_service.restart_job(
                job_id=job_id,
                parser_map=deps["parser_map"],
                chunker=deps["chunker"],
                index_factory=deps["index_factory"],
                index_id=config.index_id,
            )
            
            new_job = await ingestion_service.get_job(new_job_id)
            if not new_job:
                raise HTTPException(status_code=500, detail="Job restarted but not found")
            
            return JobResponse(
                id=new_job.id,
                name=new_job.name,
                status=new_job.status.value,
                source_path=new_job.source_path,
                document_root=new_job.document_root,
                pipeline_steps=new_job.pipeline_steps,
                created_at=format_datetime(new_job.created_at),
                started_at=format_datetime(new_job.started_at),
                completed_at=format_datetime(new_job.completed_at),
                error_message=new_job.error_message,
            )
        
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error restarting job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Documents
    # =========================================================================
    
    @router.get("/jobs/{job_id}/documents", response_model=DocumentListResponse)
    async def list_job_documents(
        job_id: str,
        status: Optional[str] = Query(None, description="Filter by status"),
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0)
    ):
        """List documents for a job."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            doc_status = DocumentStatus(status) if status else None
            docs = await ingestion_service.get_job_documents(
                job_id, doc_status, limit, offset
            )
            
            doc_responses = [
                DocumentResponse(
                    id=doc.id,
                    job_id=doc.job_id,
                    source=doc.source,
                    status=doc.status.value,
                    current_step=doc.current_step,
                    step_statuses=doc.step_statuses,
                    created_at=format_datetime(doc.created_at),
                    started_at=format_datetime(doc.started_at),
                    completed_at=format_datetime(doc.completed_at),
                    error_message=doc.error_message,
                    error_step=doc.error_step,
                    retry_count=doc.retry_count,
                )
                for doc in docs
            ]
            
            return DocumentListResponse(documents=doc_responses, total=len(doc_responses))
        
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/jobs/{job_id}/retry", response_model=RetryResponse)
    async def retry_failed_documents(job_id: str):
        """Retry all failed documents for a job."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            count = await ingestion_service.retry_failed_documents(job_id)
            return RetryResponse(retried_count=count)
        
        except Exception as e:
            logger.error(f"Error retrying documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Queue Messages
    # =========================================================================
    
    @router.get("/messages", response_model=MessagesListResponse)
    async def list_messages(
        job_id: str = Query(..., description="Job ID (required)"),
        state: Optional[str] = Query(None, description="Filter by state"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0)
    ):
        """List documents with detailed pipeline information for a specific job."""
        ingestion_service = get_ingestion_service()
        
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            job = await ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            current_pipeline_steps = await ingestion_service.get_pipeline_steps(job_id)
            
            doc_status = None
            if state and state.lower() != "all":
                status_map = {
                    "pending": DocumentStatus.PENDING,
                    "processing": DocumentStatus.PROCESSING,
                    "completed": DocumentStatus.COMPLETED,
                    "failed": DocumentStatus.FAILED,
                }
                doc_status = status_map.get(state.lower())
            
            docs = await ingestion_service.get_job_documents(
                job_id=job_id,
                status=doc_status,
                limit=limit,
                offset=offset
            )
            
            messages = []
            for doc in docs:
                messages.append(
                    MessageResponse(
                        id=doc.id,
                        job_id=doc.job_id,
                        job_name=job.name,
                        source=doc.source,
                        status=doc.status.value,
                        current_step=doc.current_step,
                        step_statuses=doc.step_statuses,
                        pipeline_steps=current_pipeline_steps,
                        created_at=format_datetime(doc.created_at),
                        started_at=format_datetime(doc.started_at),
                        completed_at=format_datetime(doc.completed_at),
                        error_message=doc.error_message,
                        error_step=doc.error_step,
                        retry_count=doc.retry_count,
                    )
                )
            
            return MessagesListResponse(
                messages=messages,
                total=len(messages),
                limit=limit,
                offset=offset,
                has_more=False,
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing messages: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats", response_model=QueueStatsResponse)
    async def get_queue_stats():
        """Get statistics about the ingestion queues."""
        queue_backend = get_queue_backend()
        
        if not queue_backend:
            raise HTTPException(status_code=503, detail="Queue backend not initialized")
        
        try:
            stats = await queue_backend.get_stats()
            
            oldest_pending = None
            if stats.get("oldest_pending"):
                oldest_pending = datetime.fromtimestamp(stats["oldest_pending"]).strftime("%Y-%m-%d %H:%M:%S")
            
            return QueueStatsResponse(
                total_messages=stats.get("total_messages", 0),
                by_state=stats.get("by_state", {}),
                by_queue=stats.get("by_queue", {}),
                failed_messages=stats.get("failed_messages", 0),
                oldest_pending=oldest_pending,
            )
        
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Legacy Ingestion Control
    # =========================================================================
    
    @router.get("/ingestion/status", response_model=IngestionStatusResponse)
    async def get_ingestion_status():
        """Get legacy ingestion status."""
        ingestion_service = get_ingestion_service()
        
        try:
            running_jobs = await ingestion_service.list_jobs(
                status=JobStatus.RUNNING,
                limit=1,
                offset=0
            )
            
            if running_jobs:
                return IngestionStatusResponse(status="running", pid=None)
            else:
                return IngestionStatusResponse(status="stopped", pid=None)
        except Exception as e:
            logger.error(f"Error getting ingestion status: {e}", exc_info=True)
            return IngestionStatusResponse(status="error", pid=None)
    
    @router.post("/ingestion/start", response_model=IngestionStatusResponse)
    async def start_ingestion():
        """Legacy start ingestion endpoint (deprecated)."""
        raise HTTPException(
            status_code=400,
            detail="Legacy ingestion start is not supported. Please create jobs via the Jobs API."
        )
    
    @router.post("/ingestion/stop", response_model=IngestionStatusResponse)
    async def stop_ingestion():
        """Legacy stop ingestion endpoint (deprecated)."""
        raise HTTPException(
            status_code=400,
            detail="Legacy ingestion stop is not supported. Jobs run to completion or failure."
        )
    
    # =========================================================================
    # Task Callbacks
    # =========================================================================
    
    @router.post("/tasks/callback", response_model=CallbackResponse)
    async def task_callback(callback: CallbackMessage):
        """Generic callback endpoint for async transformations."""
        worker_manager = get_worker_manager()
        
        if not worker_manager:
            raise HTTPException(status_code=503, detail="Worker manager not initialized")
        
        logger.info(
            f"Received task callback: task_id={callback.task_id}, status={callback.status}"
        )
        
        try:
            success = await worker_manager.handle_task_callback(
                task_id=callback.task_id,
                status=callback.status,
                message=callback.message,
            )
            
            if success:
                return CallbackResponse(
                    success=True,
                    message="Callback processed successfully",
                    task_id=callback.task_id
                )
            else:
                return CallbackResponse(
                    success=False,
                    message="Failed to process callback",
                    task_id=callback.task_id
                )
        
        except Exception as e:
            logger.error(f"Error processing task callback: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return router
