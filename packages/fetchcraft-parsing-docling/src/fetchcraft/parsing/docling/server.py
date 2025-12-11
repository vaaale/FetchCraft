"""
FastAPI server for document parsing using Docling.

This server provides endpoints for parsing documents into DocumentNodes
with configurable concurrency for requests and file processing.

Features:
- Multipart file upload support for multiple files
- Concurrent file processing with configurable limits
- Request-level concurrency control
- Health check endpoint
- Detailed error handling and logging

Environment Variables:
- HOST: Server host (default: 0.0.0.0)
- PORT: Server port (default: 8080)
- MAX_CONCURRENT_REQUESTS: Maximum concurrent requests (default: 10)
- MAX_CONCURRENT_FILES: Maximum concurrent file processing (default: 4)
- MAX_FILE_SIZE_MB: Maximum file size in MB (default: 100)
- PAGE_CHUNKS: Split documents into pages (default: true)
- DO_OCR: Enable OCR for scanned documents (default: true)
- DO_TABLE_STRUCTURE: Extract table structure (default: true)

Usage:
    # Run the server
    python -m fetchcraft.parsing.docling.server
    
    # Or use uvicorn directly
    uvicorn fetchcraft.parsing.docling.server:api --host 0.0.0.0 --port 8080
    
    # Test with curl
    curl -X POST http://localhost:8080/parse \
      -F "files=@document1.pdf" \
      -F "files=@document2.docx"
"""

import asyncio
import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fetchcraft.parsing.docling.docling_parser import DOCLING_SUPPORTED_EXTENSIONS
from fetchcraft.parsing.docling.models import (
    ParseResponse,
    BatchParseResponse,
    HealthResponse,
    JobStatusEnum,
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse
)
from fetchcraft.parsing.docling.services.parsing_service import ParsingService

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Server configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    max_concurrent_files: int = Field(default=4, description="Maximum concurrent file processing")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")

    # Docling parser configuration
    page_chunks: bool = Field(default=True, description="Split documents into pages")
    do_ocr: bool = Field(default=True, description="Enable OCR for scanned documents")
    do_table_structure: bool = Field(default=True, description="Extract table structure")

    # Version
    version: str = Field(default="1.0.0", description="API version")

    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()


# ============================================================================
# Global State
# ============================================================================

@dataclass
class Job:
    """Represents a parsing job."""
    job_id: str
    files: List[tuple] = field(default_factory=list)  # List of (filename, content) tuples
    status: JobStatusEnum = JobStatusEnum.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Optional[BatchParseResponse] = None
    error: Optional[str] = None
    task_id: Optional[str] = None  # Task ID for callback correlation
    callback_url: Optional[str] = None  # URL to send callbacks to
    metadata: Optional[Dict] = None  # Metadata to include in parsed documents


class AppState:
    """Global application state."""
    request_semaphore: Optional[asyncio.Semaphore] = None
    file_semaphore: Optional[asyncio.Semaphore] = None
    initialized: bool = False
    jobs: Dict[str, Job] = {}
    job_queue: Optional[asyncio.Queue] = None
    background_task: Optional[asyncio.Task] = None
    executor: Optional[ThreadPoolExecutor] = None
    parsing_service: Optional[ParsingService] = None


app_state = AppState()


# ============================================================================
# FastAPI Lifespan
# ============================================================================

async def send_callback(url: str, payload: dict) -> bool:
    """Send a callback to the specified URL."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            return 200 <= response.status_code < 300
    except Exception as e:
        logger.error(f"⚠️  Callback failed: {url} - {e}")
        return False


async def process_jobs():
    """Background task to process jobs from the queue."""
    logger.info("📋 Job processor started")
    while True:
        try:
            # Get next job from queue
            job_id = await app_state.job_queue.get()
            job = app_state.jobs.get(job_id)
            logger.info(f"Processing job {job_id}")

            if not job:
                continue

            # Update job status
            job.status = JobStatusEnum.PROCESSING
            job.started_at = time.time()

            logger.info(f"⚙️  Processing job {job_id} with {len(job.files)} files")

            # Process the job
            try:
                batch_start_time = time.time()

                # Create temporary directory for files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Save all files first
                    file_paths = []
                    for filename, content in job.files:
                        file_path = temp_path / filename
                        with open(file_path, "wb") as f:
                            f.write(content)
                        file_paths.append(file_path)

                    # Process files using thread pool to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    results = []
                    total_nodes_sent = 0

                    for file_path in file_paths:
                        # Run the parsing in a thread pool with metadata
                        async with app_state.file_semaphore:
                            import functools
                            parse_func = functools.partial(
                                app_state.parsing_service.parse_file_sync,
                                file_path,
                                metadata=job.metadata
                            )
                            result = await loop.run_in_executor(
                                app_state.executor,
                                parse_func
                            )
                            results.append(result)

                            # Send node callbacks if callback_url is set
                            if job.callback_url and result.success:
                                for node_idx, node_data in enumerate(result.nodes):
                                    # Note: metadata is already included in node_data by parse_file_sync
                                    callback_payload = {
                                        "task_id": job.task_id,
                                        "status": "PROCESSING",
                                        "message": {
                                            "type": "node",
                                            "node": node_data,
                                            "node_index": total_nodes_sent + node_idx,
                                            "filename": result.filename,
                                        }
                                    }
                                    await send_callback(job.callback_url, callback_payload)
                                total_nodes_sent += result.num_nodes

                            # Yield control back to event loop periodically
                            await asyncio.sleep(0)

                batch_processing_time = (time.time() - batch_start_time) * 1000

                # Aggregate statistics
                total_files = len(results)
                successful = sum(1 for r in results if r.success)
                failed = total_files - successful
                total_nodes = sum(r.num_nodes for r in results)

                job.results = BatchParseResponse(
                    results=results,
                    total_files=total_files,
                    successful=successful,
                    failed=failed,
                    total_nodes=total_nodes,
                    total_processing_time_ms=round(batch_processing_time, 2)
                )
                job.status = JobStatusEnum.COMPLETED
                print(f"✅ Job {job_id} completed: {successful}/{total_files} files successful, {total_nodes} nodes")

                # Send completion callback
                if job.callback_url:
                    callback_payload = {
                        "task_id": job.task_id,
                        "status": "COMPLETED",
                        "message": {
                            "type": "completion",
                            "total_nodes": total_nodes,
                            "total_files": total_files,
                            "successful_files": successful,
                            "failed_files": failed,
                            "processing_time_ms": round(batch_processing_time, 2),
                        }
                    }
                    await send_callback(job.callback_url, callback_payload)

            except Exception as e:
                job.status = JobStatusEnum.FAILED
                job.error = str(e)
                print(f"❌ Job {job_id} failed: {e}")

                # Send failure callback
                if job.callback_url:
                    processing_time = (time.time() - batch_start_time) * 1000 if 'batch_start_time' in dir() else 0
                    callback_payload = {
                        "task_id": job.task_id,
                        "status": "FAILED",
                        "message": {
                            "type": "failure",
                            "processing_time_ms": round(processing_time, 2),
                        },
                        "error": str(e)
                    }
                    await send_callback(job.callback_url, callback_payload)

            finally:
                job.completed_at = time.time()
                app_state.job_queue.task_done()

        except Exception as e:
            print(f"⚠️  Error in job processor: {e}")
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    print("\n" + "=" * 70)
    print("🚀 Starting Docling Parsing Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Host: {settings.host}")
    print(f"  • Port: {settings.port}")
    print(f"  • Max Concurrent Requests: {settings.max_concurrent_requests}")
    print(f"  • Max Concurrent Files: {settings.max_concurrent_files}")
    print(f"  • Max File Size: {settings.max_file_size_mb} MB")
    print(f"  • Page Chunks: {settings.page_chunks}")
    print(f"  • OCR Enabled: {settings.do_ocr}")
    print(f"  • Table Structure: {settings.do_table_structure}")
    print("=" * 70)

    # Initialize semaphores for concurrency control
    app_state.request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    app_state.file_semaphore = asyncio.Semaphore(settings.max_concurrent_files)
    app_state.job_queue = asyncio.Queue()
    app_state.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_files)

    # Initialize parsing service
    app_state.parsing_service = ParsingService(
        page_chunks=settings.page_chunks,
        do_ocr=settings.do_ocr,
        do_table_structure=settings.do_table_structure
    )

    app_state.initialized = True

    # Start background job processor
    app_state.background_task = asyncio.create_task(process_jobs())

    print(f"\n✅ Server ready at http://{settings.host}:{settings.port}")
    print(f"   API docs: http://{settings.host}:{settings.port}/docs")
    print(f"   OpenAPI schema: http://{settings.host}:{settings.port}/openapi.json")
    print("=" * 70 + "\n")

    yield

    print("\n👋 Shutting down server...")
    # Cancel background task
    if app_state.background_task:
        app_state.background_task.cancel()
        try:
            await app_state.background_task
        except asyncio.CancelledError:
            pass

    # Shutdown thread pool executor
    if app_state.executor:
        app_state.executor.shutdown(wait=True)
        print("   ThreadPoolExecutor shut down")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Docling Document Parsing API",
    description=(
        "Parse documents into structured DocumentNodes using Docling. "
        "Supports PDF, DOCX, PPTX, XLSX, HTML, Images, AsciiDoc, and Markdown."
    ),
    version=settings.version,
    lifespan=lifespan,
    contact={
        "name": "Docling Parsing API",
        "url": "https://github.com/vaaale/fetchcraft",
    },
    license_info={
        "name": "MIT",
    },
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

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """
    Save an uploaded file to disk.
    
    Args:
        upload_file: The uploaded file
        destination: Path to save the file
    """
    with open(destination, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)


async def parse_single_file(
    file: UploadFile,
    temp_dir: Path
) -> ParseResponse:
    """
    Parse a single file and return DocumentNodes.
    
    Args:
        file: Uploaded file
        temp_dir: Temporary directory for file storage
        
    Returns:
        ParseResponse with parsing results
    """
    filename = file.filename or "unknown"

    try:
        # Check file size
        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size_bytes:
            return ParseResponse(
                filename=filename,
                success=False,
                nodes=[],
                error=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum ({settings.max_file_size_mb} MB)",
                num_nodes=0,
                processing_time_ms=0
            )

        # Save file temporarily
        temp_path = temp_dir / filename
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        # Use file semaphore to limit concurrent file processing
        async with app_state.file_semaphore:
            # Run parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                app_state.executor,
                app_state.parsing_service.parse_file_sync,
                temp_path
            )
            return result

    except Exception as e:
        return ParseResponse(
            filename=filename,
            success=False,
            nodes=[],
            error=str(e),
            num_nodes=0,
            processing_time_ms=0
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API and available endpoints.
    """
    return {
        "name": "Docling Document Parsing API",
        "version": settings.version,
        "description": "Parse documents into DocumentNodes using Docling",
        "supported_formats": [
            "PDF (.pdf)",
            "Microsoft Word (.docx)",
            "Microsoft PowerPoint (.pptx)",
            "Microsoft Excel (.xlsx)",
            "HTML (.html, .htm)",
            "Images (.png, .jpg, .jpeg, .tiff, .bmp)",
            "AsciiDoc (.asciidoc, .adoc)",
            "Markdown (.md)"
        ],
        "endpoints": {
            "parse": "/parse (blocking)",
            "submit": "/submit (async job submission)",
            "job_status": "/jobs/{job_id} (check job status)",
            "job_results": "/jobs/{job_id}/results (get job results)",
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """
    Health check endpoint.
    
    Returns the current status of the service and configuration information.
    """

    return HealthResponse(
        status="healthy" if app_state.initialized else "initializing",
        version=settings.version,
        config={
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_concurrent_files": settings.max_concurrent_files,
            "max_file_size_mb": settings.max_file_size_mb,
            "page_chunks": settings.page_chunks,
            "do_ocr": settings.do_ocr,
            "do_table_structure": settings.do_table_structure
        }
    )


@app.post("/parse", response_model=BatchParseResponse, tags=["Parsing"])
async def parse_documents(
    files: List[UploadFile] = File(..., description="One or more files to parse")
):
    """
    Parse one or more documents into DocumentNodes.
    
    This endpoint accepts multiple files via multipart/form-data and processes
    them concurrently with configurable limits. Each file is parsed into one or
    more DocumentNode objects depending on the configuration.
    
    Args:
        files: List of files to parse
        
    Returns:
        BatchParseResponse with parsing results for each file
        
    Raises:
        HTTPException: If the service is not initialized or no files provided
    """
    if not app_state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again in a moment."
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    # Use request semaphore to limit concurrent requests
    async with app_state.request_semaphore:
        batch_start_time = time.time()

        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Process all files concurrently
            tasks = [
                parse_single_file(file, temp_path)
                for file in files
            ]
            results = await asyncio.gather(*tasks)

        batch_processing_time = (time.time() - batch_start_time) * 1000

        # Aggregate statistics
        total_files = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_files - successful
        total_nodes = sum(r.num_nodes for r in results)

        return BatchParseResponse(
            results=results,
            total_files=total_files,
            successful=successful,
            failed=failed,
            total_nodes=total_nodes,
            total_processing_time_ms=round(batch_processing_time, 2)
        )


@app.post("/submit", response_model=JobSubmitResponse, tags=["Async Parsing"])
async def submit_job(
    files: List[UploadFile] = File(..., description="One or more files to parse"),
    task_id: Optional[str] = Form(None, description="Task ID for callback correlation"),
    callback_url: Optional[str] = Form(None, description="URL to send callbacks to"),
    metadata: Optional[str] = Form(None, description="JSON metadata to include in parsed documents"),
):
    """
    Submit a parsing job and return immediately with a job ID.
    
    This endpoint accepts files and queues them for processing. It returns
    a job ID immediately without waiting for the parsing to complete.
    Use the /jobs/{job_id} endpoint to check the status and 
    /jobs/{job_id}/results to retrieve results when complete.
    
    If callback_url is provided, callbacks will be sent in CallbackMessage format:
    - task_id: The provided task_id (or job_id if not provided)
    - status: "PROCESSING" for each node, "COMPLETED" or "FAILED" at the end
    - message: Contains node data or completion info
    
    Args:
        files: List of files to parse
        task_id: Optional task ID for callback correlation
        callback_url: Optional URL to send callbacks to
        
    Returns:
        JobSubmitResponse with job_id for tracking
        
    Raises:
        HTTPException: If the service is not initialized or no files provided
    """
    if not app_state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again in a moment."
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Read file contents and store
    file_data = []
    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"

        # Check file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in DOCLING_SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file format: {file.filename}")
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported file format: {file.filename}"
            )

        # Check file size
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File {filename} size ({len(content) / 1024 / 1024:.2f} MB) exceeds maximum ({settings.max_file_size_mb} MB)"
            )

        file_data.append((filename, content))

    # Parse metadata JSON if provided
    parsed_metadata = None
    if metadata:
        import json
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in metadata field"
            )
    
    # Create job with callback info and metadata
    job = Job(
        job_id=job_id,
        files=file_data,
        status=JobStatusEnum.PENDING,
        task_id=task_id or job_id,  # Use job_id as task_id if not provided
        callback_url=callback_url,
        metadata=parsed_metadata
    )

    # Store job
    app_state.jobs[job_id] = job

    # Queue job for processing
    await app_state.job_queue.put(job_id)

    callback_info = f" (callback: {callback_url})" if callback_url else ""
    print(f"📥 Job {job_id} submitted with {len(files)} files{callback_info}")

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        message=f"Job submitted successfully with {len(files)} file(s)"
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Async Parsing"])
async def get_job_status(job_id: str):
    """
    Get the status of a parsing job.
    
    Returns the current status of the job including timestamps and any error
    information if the job failed.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        JobStatusResponse with current job status
        
    Raises:
        HTTPException: If job_id is not found
    """
    job = app_state.jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error
    )


@app.get("/jobs/{job_id}/results", response_model=JobResultResponse, tags=["Async Parsing"])
async def get_job_results(job_id: str):
    """
    Get the results of a completed parsing job.
    
    Returns the parsing results if the job is completed, or the current status
    if it's still processing or failed.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        JobResultResponse with results if completed
        
    Raises:
        HTTPException: If job_id is not found
    """
    job = app_state.jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        results=job.results,
        error=job.error
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "fetchcraft.parsing.docling.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
