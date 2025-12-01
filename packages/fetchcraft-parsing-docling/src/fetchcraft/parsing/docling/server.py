"""
FastAPI server for document parsing using Docling.

This server provides endpoints for parsing documents into DocumentNodes
with configurable concurrency for requests and file processing.

Architecture:
- Presentation Layer: FastAPI endpoints (this file)
- Business Logic Layer: Services (job_service.py, parsing_service.py)
- Data Access Layer: Repositories (job_repository.py)

Features:
- Multipart file upload support for multiple files
- Concurrent file processing with configurable limits
- Request-level concurrency control
- Job persistence and resumption
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
- DATA_DIR: Directory for storing jobs and files (default: ./data)
- JOBS_FILE: Name of the jobs JSON file (default: jobs.json)

Usage:
    # Run the server
    python -m fetchcraft.parsing.docling.server
    
    # Or use uvicorn directly
    uvicorn fetchcraft.parsing.docling.server:app --host 0.0.0.0 --port 8080
    
    # Test with curl
    curl -X POST http://localhost:8080/parse \
      -F "files=@document1.pdf" \
      -F "files=@document2.docx"
"""

import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict

from fetchcraft.parsing.docling.models import (
    ParseResponse,
    BatchParseResponse,
    HealthResponse,
    JobStatusEnum,
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse
)
from fetchcraft.parsing.docling.repositories.job_repository import FileSystemJobRepository
from fetchcraft.parsing.docling.services.job_service import JobService
from fetchcraft.parsing.docling.services.parsing_service import ParsingService
from fetchcraft.parsing.docling.services.callback_service import CallbackService

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Concurrency configuration
    max_concurrent_requests: int = 10
    max_concurrent_files: int = 4
    max_file_size_mb: int = 100
    
    # Docling parser configuration
    page_chunks: bool = True
    do_ocr: bool = True
    do_table_structure: bool = True
    
    # Persistence configuration
    data_dir: str = "./data"
    jobs_file: str = "jobs.json"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def max_file_size_bytes(self) -> int:
        """Calculate maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


# Initialize settings
settings = Settings()

VERSION = "1.0.0"

# ============================================================================
# Global State (Dependency Container)
# ============================================================================

class AppState:
    """Global application state and dependency container."""
    request_semaphore: Optional[asyncio.Semaphore] = None
    file_semaphore: Optional[asyncio.Semaphore] = None
    initialized: bool = False
    job_queue: Optional[asyncio.Queue] = None
    background_task: Optional[asyncio.Task] = None
    executor: Optional[ThreadPoolExecutor] = None
    
    # Services and repositories (dependency injection)
    job_repository: Optional[FileSystemJobRepository] = None
    parsing_service: Optional[ParsingService] = None
    callback_service: Optional[CallbackService] = None
    job_service: Optional[JobService] = None


app_state = AppState()


# ============================================================================
# Background Job Processor
# ============================================================================

async def process_jobs():
    """Background task to process jobs from the queue."""
    print("📋 Job processor started")
    while True:
        try:
            # Get next job from queue
            job_id = await app_state.job_queue.get()
            
            # Process the job using the service layer
            await app_state.job_service.process_job(
                job_id,
                app_state.executor,
                app_state.file_semaphore
            )
            
            app_state.job_queue.task_done()
                
        except Exception as e:
            print(f"⚠️  Error in job processor: {e}")
            await asyncio.sleep(1)


# ============================================================================
# FastAPI Lifespan
# ============================================================================

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
    print(f"  • Data Directory: {Path(settings.data_dir).absolute()}")
    print("=" * 70)
    
    # Initialize semaphores for concurrency control
    app_state.request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    app_state.file_semaphore = asyncio.Semaphore(settings.max_concurrent_files)
    app_state.job_queue = asyncio.Queue()
    app_state.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_files)
    
    # Initialize repository layer
    app_state.job_repository = FileSystemJobRepository(
        data_dir=settings.data_dir,
        jobs_file=settings.jobs_file
    )
    
    # Initialize service layer
    app_state.parsing_service = ParsingService(
        page_chunks=settings.page_chunks,
        do_ocr=settings.do_ocr,
        do_table_structure=settings.do_table_structure
    )
    
    # Initialize callback service
    try:
        app_state.callback_service = CallbackService()
        print("  • Callback service: Enabled")
    except ImportError:
        app_state.callback_service = None
        print("  • Callback service: Disabled (httpx not installed)")
    
    app_state.job_service = JobService(
        job_repository=app_state.job_repository,
        parsing_service=app_state.parsing_service,
        max_file_size_bytes=settings.max_file_size_bytes,
        callback_service=app_state.callback_service
    )
    
    # Load existing jobs from disk
    app_state.job_service.load_jobs()
    
    app_state.initialized = True
    
    # Start background job processor
    app_state.background_task = asyncio.create_task(process_jobs())
    
    # Resume processing of unfinished jobs
    unfinished_job_ids = await app_state.job_service.resume_unfinished_jobs()
    for job_id in unfinished_job_ids:
        await app_state.job_queue.put(job_id)
    
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
    
    # Close callback service
    if app_state.callback_service:
        await app_state.callback_service.close()
        print("   Callback service closed")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Docling Document Parsing API",
    description=(
        "Parse documents into structured DocumentNodes using Docling. "
        "Supports PDF, DOCX, PPTX, XLSX, HTML, Images, AsciiDoc, and Markdown."
    ),
    version=VERSION,
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
# Helper Functions (Presentation Layer)
# ============================================================================

async def parse_single_file(
    file: UploadFile,
    temp_dir: Path
) -> ParseResponse:
    """
    Parse a single file and return DocumentNodes.
    
    This function runs the CPU-intensive parsing in a thread pool
    to avoid blocking the event loop.
    
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
        # Use parsing service to parse the file
        async with app_state.file_semaphore:
            result = await app_state.parsing_service.parse_file_async(
                temp_path,
                app_state.executor
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
# API Endpoints (Presentation Layer)
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API and available endpoints.
    """
    return {
        "name": "Docling Document Parsing API",
        "version": VERSION,
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
        version=VERSION,
        config={
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_concurrent_files": settings.max_concurrent_files,
            "max_file_size_mb": settings.max_file_size_mb,
            "page_chunks": settings.page_chunks,
            "do_ocr": settings.do_ocr,
            "do_table_structure": settings.do_table_structure
        },
        environment=dict(os.environ)
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
    callback_url: Optional[str] = Form(None, description="Optional callback URL to receive nodes as they are parsed")
):
    """
    Submit a parsing job and return immediately with a job ID.
    
    This endpoint accepts files and queues them for processing. It returns
    a job ID immediately without waiting for the parsing to complete.
    Use the /jobs/{job_id} endpoint to check the status and 
    /jobs/{job_id}/results to retrieve results when complete.
    
    If callback_url is provided, nodes will be sent to this URL as they become
    available during parsing. Each callback includes job metadata for identification.
    
    Args:
        files: List of files to parse
        callback_url: Optional URL to receive parsing callbacks
        
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
    
    # Validate callback_url if provided
    if callback_url and not app_state.callback_service:
        raise HTTPException(
            status_code=400,
            detail="Callback functionality not available (httpx not installed)"
        )
    
    # Read file contents
    files_data = []
    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"
        files_data.append((filename, content))
    
    # Use service layer to create job
    try:
        job_id, filenames = app_state.job_service.create_job(files_data, callback_url)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    
    # Queue job for processing
    await app_state.job_queue.put(job_id)
    
    callback_msg = f" with callback to {callback_url}" if callback_url else ""
    print(f"📥 Job {job_id} submitted with {len(files)} files{callback_msg}")
    
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
    job = app_state.job_service.get_job(job_id)
    
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
    job = app_state.job_service.get_job(job_id)
    
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

    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"  • {key}: {value}")

    uvicorn.run(
        "fetchcraft.parsing.docling.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
