"""
Fetchcraft Admin Server V2 - Enhanced job and document tracking.

This FastAPI application provides a web interface for:
- Creating and managing ingestion jobs
- Tracking documents through pipeline steps
- Viewing job and document status
- Retrying failed documents
"""
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from fetchcraft.node import DocumentNode

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "asyncpg is required for PostgreSQL backend. "
        "Install it with: pip install asyncpg"
    )

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient

from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.ingestion.models import JobStatus, DocumentStatus
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
from fetchcraft.ingestion.repository import (
    PostgresJobRepository,
    PostgresDocumentRepository,
)
from fetchcraft.node_parser import HierarchicalNodeParser
from fetchcraft.parsing.docling.client.docling_parser import RemoteDoclingParser
from fetchcraft.parsing.text_file_parser import TextFileParser
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.admin.services.ingestion_service import IngestionService
from fetchcraft.admin.services.pipeline_factory import IngestionPipelineFactory
from fetchcraft.admin.services.callback_service import CallbackService
from fetchcraft.admin.api.models import DoclingNodeCallback, DoclingCompletionCallback
from fetchcraft.admin.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
# All configuration is now managed via pydantic-settings in config.py
# Access settings via the global 'settings' object imported from fetchcraft.admin.config

# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    pool: Optional[asyncpg.Pool] = None
    ingestion_service: Optional[IngestionService] = None
    job_repo: Optional[PostgresJobRepository] = None
    doc_repo: Optional[PostgresDocumentRepository] = None
    queue_backend: Optional[AsyncPostgresQueue] = None

app_state = AppState()

# ============================================================================
# Database Helper Functions
# ============================================================================

async def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool."""
    if app_state.pool is None:
        raise RuntimeError("Database pool not initialized")
    return app_state.pool

def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime to ISO string."""
    if dt is None:
        return None
    return dt.isoformat()

# ============================================================================
# API Models
# ============================================================================

class CreateJobRequest(BaseModel):
    name: str
    source_path: str  # Relative to document root

class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    source_path: str
    document_root: str
    pipeline_steps: List[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]

class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int

class DocumentResponse(BaseModel):
    id: str
    job_id: str
    source: str
    status: str
    current_step: Optional[str]
    step_statuses: Dict[str, str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    error_step: Optional[str]
    retry_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

class DirectoryItem(BaseModel):
    name: str
    path: str
    is_directory: bool

class DirectoryListResponse(BaseModel):
    items: List[DirectoryItem]
    current_path: str

class RetryResponse(BaseModel):
    retried_count: int

class MessageResponse(BaseModel):
    id: str
    job_id: str
    job_name: str
    source: str
    status: str
    current_step: Optional[str]
    step_statuses: Dict[str, str]
    pipeline_steps: List[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    error_step: Optional[str]
    retry_count: int

class MessagesListResponse(BaseModel):
    messages: List[MessageResponse]
    total: int
    limit: int
    offset: int
    has_more: bool

class QueueStatsResponse(BaseModel):
    total_messages: int
    by_state: Dict[str, int]
    by_queue: Dict[str, int]
    failed_messages: int
    oldest_pending: Optional[str]

class IngestionStatusResponse(BaseModel):
    status: str  # "running", "stopped", or "error"
    pid: Optional[int]

class DoclingNodeCallback(BaseModel):
    """Callback payload from docling server for a single node."""
    job_id: str
    filename: str
    node_index: int
    total_nodes: int
    node: Dict[str, Any]

class DoclingCompletionCallback(BaseModel):
    """Callback payload from docling server for completion."""
    job_id: str
    filename: str
    status: str  # "completed" or "failed"
    total_nodes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the application."""
    logger.info("=" * 70)
    logger.info("🚀 Fetchcraft Admin Server V2 - Initializing")
    logger.info("=" * 70)
    
    logger.info(f"🗄️  Database: {settings.database_display}")
    logger.info(f"🌐 Server: http://{settings.host}:{settings.port}")
    logger.info(f"📁 Documents: {settings.documents_path}")
    
    try:
        # Calculate pool size based on workers
        # Formula: base connections + (workers * 2) for main+remote queues + overhead
        # Each worker needs connections for: lease_next, ack/nack, enqueue operations
        min_pool_size = max(settings.pool_min_size, 5)
        max_pool_size = max(settings.pool_max_size, settings.num_workers * 3 + 10)
        
        logger.info(
            f"📊 Calculated connection pool size: {min_pool_size}-{max_pool_size} "
            f"(based on {settings.num_workers} workers)"
        )
        
        # Initialize database pool
        app_state.pool = await asyncpg.create_pool(
            settings.postgres_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            command_timeout=60,
        )
        logger.info(f"✓ Connected to PostgreSQL (pool: {min_pool_size}-{max_pool_size})")
        
        # Initialize repositories
        app_state.job_repo = PostgresJobRepository(app_state.pool)
        app_state.doc_repo = PostgresDocumentRepository(app_state.pool)
        logger.info("✓ Repositories initialized")
        
        # Initialize database schemas
        await app_state.job_repo._ensure_schema()
        await app_state.doc_repo._ensure_schema()
        logger.info("✓ Database schemas initialized")
        
        # Initialize queue backend - IMPORTANT: Pass the shared pool
        app_state.queue_backend = AsyncPostgresQueue(pool=app_state.pool)
        # Initialize the messages table (required when using external pool)
        await app_state.queue_backend._init_db()
        logger.info("✓ Queue backend initialized (using shared connection pool)")
        
        # Initialize pipeline factory
        pipeline_factory = IngestionPipelineFactory(
            queue_backend=app_state.queue_backend,
            job_repo=app_state.job_repo,
            doc_repo=app_state.doc_repo,
            document_root=settings.documents_path,
            num_workers=settings.num_workers,
        )
        logger.info(f"✓ Pipeline factory initialized with {settings.num_workers} worker(s)")
        
        # Initialize ingestion service with factory
        app_state.ingestion_service = IngestionService(
            job_repo=app_state.job_repo,
            doc_repo=app_state.doc_repo,
            pipeline_factory=pipeline_factory,
            document_root=settings.documents_path,
        )
        logger.info("✓ Ingestion service initialized")
        
        # Recover any jobs that were running when server crashed/restarted
        logger.info("🔄 Checking for jobs to recover...")
        deps = _get_ingestion_dependencies()
        recovered_count = await app_state.ingestion_service.recover_running_jobs(
            parser_map=deps["parser_map"],
            chunker=deps["chunker"],
            vector_index=deps["vector_index"],
            doc_store=deps["doc_store"],
            index_id=settings.index_id,
        )
        if recovered_count > 0:
            logger.info(f"✓ Recovered {recovered_count} job(s)")
        
        logger.info("✅ Server ready!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"⚠️  Error during initialization: {e}")
        raise
    
    yield
    
    # Cleanup
    if app_state.pool:
        await app_state.pool.close()
        logger.info("✓ Database pool closed")
    
    logger.info("👋 Server shutdown complete")

# Create FastAPI api
app = FastAPI(
    title="Fetchcraft Admin V2",
    description="Enhanced administration interface with job and document tracking",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helper Functions
# ============================================================================

def _get_ingestion_dependencies():
    """Get common dependencies needed for ingestion jobs."""
    doc_store = MongoDBDocumentStore(
        connection_string=settings.mongo_uri,
        database_name="fetchcraft",
        collection_name=settings.collection_name,
    )
    
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url
    )
    
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=settings.enable_hybrid,
        fusion_method=settings.fusion_method
    )
    
    chunker = HierarchicalNodeParser(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        child_sizes=settings.child_chunks,
        child_overlap=50
    )
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        doc_store=doc_store,
        index_id=settings.index_id
    )
    
    # Build callback URL for docling async parsing
    callback_url = f"{settings.callback_base_url}/api/callbacks/parsing"
    
    parser_map = {
        "default": TextFileParser(),
        "application/pdf": RemoteDoclingParser(
            docling_url=settings.docling_server,
            callback_url=callback_url
        )
    }
    
    return {
        "doc_store": doc_store,
        "vector_index": vector_index,
        "chunker": chunker,
        "parser_map": parser_map,
    }


# ============================================================================
# API Routes - Jobs
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    db_healthy = False
    if app_state.pool:
        try:
            async with app_state.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                db_healthy = True
        except Exception:
            pass
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database_connected": db_healthy
    }

@app.get("/api/directories", response_model=DirectoryListResponse)
async def list_directories(
    path: str = Query("", description="Relative path from document root")
):
    """
    List directories and files in the document root.
    
    This endpoint allows browsing the document directory structure
    to select a folder for ingestion.
    """
    try:
        # Resolve path relative to document root
        full_path = settings.documents_path / path if path else settings.documents_path
        
        # Security check: ensure path is within document root
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(settings.documents_path.resolve())):
            raise HTTPException(
                status_code=400,
                detail="Path must be within document root"
            )
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        
        if not full_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        # List items
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
                # Skip items outside document root
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

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: CreateJobRequest):
    """
    Create a new ingestion job.
    
    This starts processing documents from the specified path.
    """
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get ingestion dependencies
        deps = _get_ingestion_dependencies()
        
        # Create job
        job_id = await app_state.ingestion_service.create_job(
            name=request.name,
            source_path=request.source_path,
            parser_map=deps["parser_map"],
            chunker=deps["chunker"],
            vector_index=deps["vector_index"],
            doc_store=deps["doc_store"],
            index_id=settings.index_id,
        )
        
        # Get created job
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

@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all ingestion jobs."""
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

@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get details of a specific job."""
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

@app.delete("/api/jobs/{job_id}")
async def delete_job(
    job_id: str,
    delete_documents: bool = Query(True, description="Delete associated documents")
):
    """
    Delete a job and optionally its documents.
    
    Only completed or failed jobs can be deleted.
    """
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

@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """
    Stop a running or pending job.
    
    This stops the pipeline workers and marks the job as cancelled.
    """
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Stop the job
        stopped = await app_state.ingestion_service.stop_job(job_id)
        
        if not stopped:
            # Get job to provide better error message
            job = await app_state.ingestion_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job cannot be stopped. Current status: {job.status.value}"
                )
        
        # Get the updated job
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

@app.post("/api/jobs/{job_id}/restart", response_model=JobResponse)
async def restart_job(job_id: str):
    """
    Restart a completed or failed job.
    
    This creates a new job with the same configuration.
    """
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get dependencies
        deps = _get_ingestion_dependencies()
        
        # Restart the job
        new_job_id = await app_state.ingestion_service.restart_job(
            job_id=job_id,
            parser_map=deps["parser_map"],
            chunker=deps["chunker"],
            vector_index=deps["vector_index"],
            doc_store=deps["doc_store"],
            index_id=settings.index_id,
        )
        
        # Get the new job
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

# ============================================================================
# API Routes - Documents
# ============================================================================

@app.get("/api/jobs/{job_id}/documents", response_model=DocumentListResponse)
async def list_job_documents(
    job_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """List documents for a job."""
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

@app.post("/api/jobs/{job_id}/retry", response_model=RetryResponse)
async def retry_failed_documents(job_id: str):
    """Retry all failed documents for a job."""
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        count = await app_state.ingestion_service.retry_failed_documents(job_id)
        return RetryResponse(retried_count=count)
        
    except Exception as e:
        logger.error(f"Error retrying documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API Routes - Queue Messages (for Queue tab)
# ============================================================================

@app.get("/api/messages", response_model=MessagesListResponse)
async def list_messages(
    job_id: str = Query(..., description="Job ID (required)"),
    state: Optional[str] = Query(None, description="Filter by state (maps to document status)"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List documents with detailed pipeline information for a specific job."""
    if not app_state.ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get the job to verify it exists
        job = await app_state.ingestion_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get pipeline steps for this specific job
        current_pipeline_steps = await app_state.ingestion_service.get_pipeline_steps(job_id)
        
        # Map frontend state filter to DocumentStatus
        doc_status = None
        if state and state.lower() != "all":
            status_map = {
                "pending": DocumentStatus.PENDING,
                "processing": DocumentStatus.PROCESSING,
                "completed": DocumentStatus.COMPLETED,
                "failed": DocumentStatus.FAILED,
            }
            doc_status = status_map.get(state.lower())
        
        # Get documents for this specific job
        docs = await app_state.ingestion_service.get_job_documents(
            job_id=job_id,
            status=doc_status,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
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
                    pipeline_steps=current_pipeline_steps,  # Job-specific pipeline steps
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
            has_more=False,  # We're getting all matching docs for now
        )
        
    except Exception as e:
        logger.error(f"Error listing messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """Get statistics about the ingestion queues."""
    try:
        async with app_state.pool.acquire() as conn:
            # Total messages
            total = await conn.fetchval("SELECT COUNT(*) FROM messages")

            # Messages by state
            state_rows = await conn.fetch(
                "SELECT state, COUNT(*) as count FROM messages GROUP BY state"
            )
            by_state = {row["state"]: row["count"] for row in state_rows}

            # Messages by queue
            queue_rows = await conn.fetch(
                "SELECT queue, COUNT(*) as count FROM messages GROUP BY queue"
            )
            by_queue = {row["queue"]: row["count"] for row in queue_rows}

            # Failed messages (state = 'done' and attempts > 3)
            failed = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE state = 'done' AND attempts > 3"
            )

            # Oldest pending message
            oldest_row = await conn.fetchrow(
                "SELECT MIN(available_at) as oldest FROM messages WHERE state = 'ready'"
            )
            oldest_pending = None
            if oldest_row and oldest_row["oldest"]:
                oldest_pending = datetime.fromtimestamp(oldest_row["oldest"]).strftime("%Y-%m-%d %H:%M:%S")

            return QueueStatsResponse(
                total_messages=total or 0,
                by_state=by_state,
                by_queue=by_queue,
                failed_messages=failed or 0,
                oldest_pending=oldest_pending,
            )

    except Exception as e:
        logger.error(f"Error getting queue stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API Routes - Legacy Ingestion Control (for Ingestion Control tab)
# ============================================================================
# Note: The V2 system uses job-based ingestion instead of a single background process.
# These endpoints are provided for backward compatibility with the UI.

@app.get("/api/ingestion/status", response_model=IngestionStatusResponse)
async def get_ingestion_status():
    """
    Get legacy ingestion status.
    
    NOTE: V2 uses job-based ingestion. Check running jobs via /api/jobs instead.
    This endpoint reports based on whether any jobs are currently running.
    """
    try:
        # Check if any jobs are currently running
        running_jobs = await app_state.job_repo.list_jobs(
            status=JobStatus.RUNNING,
            limit=1,
            offset=0
        )
        
        if running_jobs:
            return IngestionStatusResponse(
                status="running",
                pid=None  # No single PID in job-based system
            )
        else:
            return IngestionStatusResponse(
                status="stopped",
                pid=None
            )
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}", exc_info=True)
        return IngestionStatusResponse(
            status="error",
            pid=None
        )

@app.post("/api/ingestion/start", response_model=IngestionStatusResponse)
async def start_ingestion():
    """
    Legacy start ingestion endpoint.
    
    NOTE: V2 uses job-based ingestion. Create jobs via /api/jobs instead.
    This endpoint is deprecated and returns a message directing users to the Jobs tab.
    """
    raise HTTPException(
        status_code=400,
        detail="Legacy ingestion start is not supported in V2. Please create jobs via the Jobs tab."
    )

@app.post("/api/ingestion/stop", response_model=IngestionStatusResponse)
async def stop_ingestion():
    """
    Legacy stop ingestion endpoint.
    
    NOTE: V2 uses job-based ingestion. Individual jobs cannot be stopped via this endpoint.
    """
    raise HTTPException(
        status_code=400,
        detail="Legacy ingestion stop is not supported in V2. Jobs run to completion or failure."
    )

# ============================================================================
# Docling Parsing Callbacks
# ============================================================================

@app.post("/api/callbacks/parsing")
async def docling_parsing_callback(callback: Dict[str, Any]):
    """
    Receive callbacks from the docling parsing server.
    
    This endpoint handles two types of callbacks:
    1. Node callbacks - Each parsed node is sent individually
    2. Completion callbacks - Final status when all nodes are processed
    
    The callback uses job_id as the correlation ID to track which document
    the nodes belong to.
    """
    # Validate required dependencies
    if not app_state.doc_repo or not app_state.job_repo or not app_state.queue_backend:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Create callback service instance
    callback_service = CallbackService(
        doc_repo=app_state.doc_repo,
        job_repo=app_state.job_repo,
        queue_backend=app_state.queue_backend
    )
    
    try:
        # Determine callback type based on presence of 'node' field
        if "node" in callback:
            # Node callback - process and enqueue the node
            node_callback = DoclingNodeCallback(**callback)
            
            logger.info(
                f"Received docling node callback: job_id={node_callback.job_id}, "
                f"filename={node_callback.filename}, node={node_callback.node_index+1}/{node_callback.total_nodes}"
            )
            
            # Delegate to service
            doc_id = await callback_service.handle_node_callback(
                job_id=node_callback.job_id,
                filename=node_callback.filename,
                node_index=node_callback.node_index,
                total_nodes=node_callback.total_nodes,
                node=node_callback.node
            )
            
            if not doc_id:
                logger.error(f"Parent document not found for docling_job_id={node_callback.job_id}")
                raise HTTPException(status_code=404, detail="Parent document not found")
            
            return {"status": "success", "document_id": doc_id}
        
        elif "status" in callback:
            # Completion callback - mark parsing as complete
            completion_callback = DoclingCompletionCallback(**callback)
            
            logger.info(
                f"Received docling completion callback: job_id={completion_callback.job_id}, "
                f"status={completion_callback.status}, total_nodes={completion_callback.total_nodes}"
            )
            
            # Delegate to service
            success = await callback_service.handle_completion_callback(
                job_id=completion_callback.job_id,
                filename=completion_callback.filename,
                status=completion_callback.status,
                total_nodes=completion_callback.total_nodes,
                error=completion_callback.error
            )
            
            if not success:
                logger.error(f"Parent document not found for docling_job_id={completion_callback.job_id}")
                raise HTTPException(status_code=404, detail="Parent document not found")
            
            return {"status": "success"}
        
        else:
            logger.warning(f"Unknown callback type: {callback}")
            raise HTTPException(status_code=400, detail="Unknown callback type")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing docling callback: {e}", exc_info=True)
        raise

# ============================================================================
# Static File Serving
# ============================================================================

# Navigate from src/fetchcraft/admin to package root, then to frontend/dist
# Path: packages/fetchcraft-admin/src/fetchcraft/admin/server_v2.py -> packages/fetchcraft-admin/frontend/dist
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    
    @app.get("/")
    async def serve_index():
        """Serve the main index.html file."""
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            {"message": "Frontend not built"},
            status_code=404,
        )
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - return index.html for all non-API routes."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        
        return JSONResponse(
            {"message": "Frontend not built"},
            status_code=404,
        )
else:
    @app.get("/")
    async def no_frontend():
        """Fallback when frontend is not built."""
        return JSONResponse({
            "message": "Fetchcraft Admin API V2 is running",
            "note": "Frontend not built",
            "api_docs": "/docs",
        })

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the admin server."""
    import uvicorn
    
    logger.info("=" * 70)
    logger.info("🚀 Starting Fetchcraft Admin Server V2")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  • Database: {settings.database_display}")
    logger.info(f"  • Host: {settings.host}")
    logger.info(f"  • Port: {settings.port}")
    logger.info(f"  • Documents: {settings.documents_path}")
    logger.info(f"  • Workers: {settings.num_workers}")
    logger.info("=" * 70)
    
    uvicorn.run(
        "fetchcraft.admin.server:api",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )

if __name__ == "__main__":
    main()
