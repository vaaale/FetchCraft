"""Main API router for jobs, documents, messages, and callbacks."""
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from fetchcraft.admin.api.schema import (
    CallbackMessage,
    CallbackResponse,
    CreateJobRequest,
    DirectoryItem,
    DirectoryListResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestionStatusResponse,
    JobListResponse,
    JobResponse,
    MessageResponse,
    MessagesListResponse,
    QueueStatsResponse,
    RetryResponse, HealthResponse,
)
from fetchcraft.admin.config import Settings
from fetchcraft.ingestion.models import DocumentStatus, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime to ISO string."""
    if dt is None:
        return None
    return dt.isoformat()


def get_app_state():
    """Get the application state. Must be set by server.py during startup."""
    from fetchcraft.admin.server import app_state
    return app_state


def get_settings() -> Settings:
    """Get application settings."""
    from fetchcraft.admin.config import settings
    return settings


def get_ingestion_dependencies():
    """Get ingestion dependencies. Must be set by server.py."""
    from fetchcraft.admin.server import _get_ingestion_dependencies
    return _get_ingestion_dependencies()


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """
    Health check endpoint.

    Returns the current status of the service and configuration information.
    """
    app_state = get_app_state()
    settings = get_settings()
    db_healthy = False
    if app_state.queue_backend:
        try:
            db_healthy = await app_state.queue_backend.check_health()
        except Exception:
            pass

    status = "healthy" if (app_state.initialized and db_healthy) else "initializing" if not app_state.initialized else "degraded" if not db_healthy else "unknown"
    health = HealthResponse(
        status=status,
        config=settings.model_dump(),
        environment=dict(os.environ),
    )
    logger.info(f"System health:\n\t{health.model_dump_json(indent=2)}")

    return health


# =============================================================================
# Directory Browsing
# =============================================================================

@router.get("/directories", response_model=DirectoryListResponse)
async def list_directories(
    path: str = Query("", description="Relative path from document root")
):
    """List directories and files in the document root."""
    try:
        settings = get_settings()
        full_path = settings.documents_path / path if path else settings.documents_path
        full_path = full_path.resolve()

        if not str(full_path).startswith(str(settings.documents_path.resolve())):
            raise HTTPException(status_code=400, detail="Path must be within document root")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        if not full_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        items = []
        for item in sorted(full_path.iterdir()):
            try:
                relative_path = item.relative_to(settings.documents_path)
                items.append(DirectoryItem(
                    name=item.name,
                    path=str(relative_path),
                    is_directory=item.is_dir()
                ))
            except ValueError:
                continue

        return DirectoryListResponse(
            items=items,
            current_path=str(full_path.relative_to(settings.documents_path)) if path else ""
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing directories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Jobs
# =============================================================================

@router.post("/jobs", response_model=JobResponse)
async def create_job(request: CreateJobRequest):
    """Create a new ingestion job."""
    app_state = get_app_state()
    settings = get_settings()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        deps = get_ingestion_dependencies()

        job_id = await app_state.ingestion_service.create_job(
            name=request.name,
            source_path=request.source_path,
            parser_map=deps["parser_map"],
            chunker=deps["chunker"],
            index_factory=deps["index_factory"],
            index_id=settings.index_id,
        )

        job = await app_state.ingestion_service.get_job(job_id)
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
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        job_status = JobStatus(status) if status else None
        jobs = await app_state.ingestion_service.list_jobs(job_status, limit, offset)

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
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        job = await app_state.ingestion_service.get_job(job_id)
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
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        deleted = await app_state.ingestion_service.delete_job(job_id, delete_documents)
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
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        stopped = await app_state.ingestion_service.stop_job(job_id)

        if not stopped:
            job = await app_state.ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job cannot be stopped. Current status: {job.status.value}"
                )

        job = await app_state.ingestion_service.get_job(job_id)
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
    app_state = get_app_state()
    settings = get_settings()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        deps = get_ingestion_dependencies()

        new_job_id = await app_state.ingestion_service.restart_job(
            job_id=job_id,
            parser_map=deps["parser_map"],
            chunker=deps["chunker"],
            index_factory=deps["index_factory"],
            index_id=settings.index_id,
        )

        new_job = await app_state.ingestion_service.get_job(new_job_id)
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


# =============================================================================
# Documents
# =============================================================================

@router.get("/jobs/{job_id}/documents", response_model=DocumentListResponse)
async def list_job_documents(
    job_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """List documents for a job."""
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        doc_status = DocumentStatus(status) if status else None
        docs = await app_state.ingestion_service.get_job_documents(
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
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        count = await app_state.ingestion_service.retry_failed_documents(job_id)
        return RetryResponse(retried_count=count)

    except Exception as e:
        logger.error(f"Error retrying documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Queue Messages
# =============================================================================

@router.get("/messages", response_model=MessagesListResponse)
async def list_messages(
    job_id: str = Query(..., description="Job ID (required)"),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List documents with detailed pipeline information for a specific job."""
    app_state = get_app_state()
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        job = await app_state.ingestion_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        current_pipeline_steps = await app_state.ingestion_service.get_pipeline_steps(job_id)

        doc_status = None
        if state and state.lower() != "all":
            status_map = {
                "pending": DocumentStatus.PENDING,
                "processing": DocumentStatus.PROCESSING,
                "completed": DocumentStatus.COMPLETED,
                "failed": DocumentStatus.FAILED,
            }
            doc_status = status_map.get(state.lower())

        docs = await app_state.ingestion_service.get_job_documents(
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
    app_state = get_app_state()

    if not app_state.queue_backend:
        raise HTTPException(status_code=503, detail="Queue backend not initialized")

    try:
        stats = await app_state.queue_backend.get_stats()

        # Format oldest_pending timestamp if present
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


# =============================================================================
# Legacy Ingestion Control
# =============================================================================

@router.get("/ingestion/status", response_model=IngestionStatusResponse)
async def get_ingestion_status():
    """Get legacy ingestion status."""
    app_state = get_app_state()
    try:
        running_jobs = await app_state.job_repo.list_jobs(
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
        detail="Legacy ingestion start is not supported in V2. Please create jobs via the Jobs tab."
    )


@router.post("/ingestion/stop", response_model=IngestionStatusResponse)
async def stop_ingestion():
    """Legacy stop ingestion endpoint (deprecated)."""
    raise HTTPException(
        status_code=400,
        detail="Legacy ingestion stop is not supported in V2. Jobs run to completion or failure."
    )


# =============================================================================
# Generic Task Callbacks
# =============================================================================

@router.post("/tasks/callback", response_model=CallbackResponse)
async def task_callback(callback: CallbackMessage):
    """
    Generic callback endpoint for async transformations.
    
    External services (e.g., docling parsing server) send callbacks here
    when async tasks complete or fail.
    """
    app_state = get_app_state()

    if not app_state.worker_manager:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")

    logger.info(
        f"Received task callback: task_id={callback.task_id}, status={callback.status}"
    )

    try:
        success = await app_state.worker_manager.handle_task_callback(
            task_id=callback.task_id,
            status=callback.status,
            message=callback.message,
        )

        if success:
            return CallbackResponse(
                success=True,
                message=f"Callback processed successfully",
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
