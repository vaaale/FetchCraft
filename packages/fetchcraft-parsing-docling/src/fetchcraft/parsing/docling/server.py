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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fetchcraft.parsing.docling.docling_parser import DoclingDocumentParser
from fetchcraft.parsing.docling.models import ParseResponse, BatchParseResponse, HealthResponse

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
MAX_CONCURRENT_FILES = int(os.getenv("MAX_CONCURRENT_FILES", "4"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Docling parser configuration
PAGE_CHUNKS = os.getenv("PAGE_CHUNKS", "true").lower() == "true"
DO_OCR = os.getenv("DO_OCR", "true").lower() == "true"
DO_TABLE_STRUCTURE = os.getenv("DO_TABLE_STRUCTURE", "true").lower() == "true"

VERSION = "1.0.0"

# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    request_semaphore: Optional[asyncio.Semaphore] = None
    file_semaphore: Optional[asyncio.Semaphore] = None
    initialized: bool = False


app_state = AppState()


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
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print(f"  • Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"  • Max Concurrent Files: {MAX_CONCURRENT_FILES}")
    print(f"  • Max File Size: {MAX_FILE_SIZE_MB} MB")
    print(f"  • Page Chunks: {PAGE_CHUNKS}")
    print(f"  • OCR Enabled: {DO_OCR}")
    print(f"  • Table Structure: {DO_TABLE_STRUCTURE}")
    print("=" * 70)
    
    # Initialize semaphores for concurrency control
    app_state.request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    app_state.file_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    app_state.initialized = True
    
    print(f"\n✅ Server ready at http://{HOST}:{PORT}")
    print(f"   API docs: http://{HOST}:{PORT}/docs")
    print(f"   OpenAPI schema: http://{HOST}:{PORT}/openapi.json")
    print("=" * 70 + "\n")
    
    yield
    
    print("\n👋 Shutting down server...")


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
    start_time = time.time()
    filename = file.filename or "unknown"
    
    try:
        # Check file size
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            return ParseResponse(
                filename=filename,
                success=False,
                nodes=[],
                error=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum ({MAX_FILE_SIZE_MB} MB)",
                num_nodes=0,
                processing_time_ms=0
            )
        
        # Save file temporarily
        temp_path = temp_dir / filename
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # Use file semaphore to limit concurrent file processing
        async with app_state.file_semaphore:
            # Create parser for this file
            parser = DoclingDocumentParser.from_file(
                file_path=temp_path,
                page_chunks=PAGE_CHUNKS,
                do_ocr=DO_OCR,
                do_table_structure=DO_TABLE_STRUCTURE
            )
            
            # Parse document and collect nodes
            nodes = []
            async for node in parser.get_documents():
                # Convert node to dictionary for JSON serialization
                nodes.append(node.model_dump())
            
            processing_time = (time.time() - start_time) * 1000
            
            return ParseResponse(
                filename=filename,
                success=True,
                nodes=nodes,
                error=None,
                num_nodes=len(nodes),
                processing_time_ms=round(processing_time, 2)
            )
            
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return ParseResponse(
            filename=filename,
            success=False,
            nodes=[],
            error=str(e),
            num_nodes=0,
            processing_time_ms=round(processing_time, 2)
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
            "parse": "/parse",
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
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "max_concurrent_files": MAX_CONCURRENT_FILES,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "page_chunks": PAGE_CHUNKS,
            "do_ocr": DO_OCR,
            "do_table_structure": DO_TABLE_STRUCTURE
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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "fetchcraft.parsing.docling.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
