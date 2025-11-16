"""
Fetchcraft Admin Server - Web-based administration interface for document ingestion.

This FastAPI application provides a web interface for:
- Viewing and filtering ingestion queue messages
- Starting and stopping ingestion jobs
- Monitoring ingestion statistics
"""
import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "asyncpg is required for PostgreSQL backend. "
        "Install it with: pip install asyncpg"
    )

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

POSTGRES_URL = os.getenv("POSTGRES_URL","postgresql://postgres:password@localhost:5432/ingestion")
POOL_MIN_SIZE = int(os.getenv("POOL_MIN_SIZE", "5"))
POOL_MAX_SIZE = int(os.getenv("POOL_MAX_SIZE", "10"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# ============================================================================
# Global State
# ============================================================================


class AppState:
    """Global application state."""

    pool: Optional[asyncpg.Pool] = None
    ingestion_process: Optional[Process] = None
    ingestion_status: str = "stopped"  # stopped, running, error


app_state = AppState()

# ============================================================================
# Database Helper Functions
# ============================================================================


async def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool."""
    if app_state.pool is None:
        raise RuntimeError("Database pool not initialized")
    return app_state.pool


def format_timestamp(timestamp: Optional[int]) -> Optional[str]:
    """Format Unix timestamp to human-readable string."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def parse_message_body(body: Any) -> Dict[str, Any]:
    """Parse message body (JSONB from PostgreSQL)."""
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}
    return {"raw": str(body)}


# ============================================================================
# Ingestion Worker (Module-level for multiprocessing)
# ============================================================================


def _ingestion_worker():
    """
    Wrapper to run the async ingestion function in a separate process.
    Must be at module level for multiprocessing to pickle it.
    """
    from .ingestion import run_ingestion
    asyncio.run(run_ingestion())


# ============================================================================
# API Models
# ============================================================================


class MessageResponse(BaseModel):
    id: str
    queue: str
    state: str
    attempts: int
    available_at: Optional[str]
    lease_until: Optional[str]
    body_preview: str


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
    status: str
    pid: Optional[int]


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the application."""
    print("\n" + "=" * 70)
    print("🚀 Fetchcraft Admin Server - Initializing")
    print("=" * 70)
    db_display = POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL
    print(f"\n🗄️  Database: {db_display}")
    print(f"🌐 Server: http://{HOST}:{PORT}")

    # Initialize database pool
    try:
        app_state.pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            command_timeout=60,
        )
        print(f"✓ Connected to PostgreSQL (pool: {POOL_MIN_SIZE}-{POOL_MAX_SIZE})")
        
        # Initialize schema if it doesn't exist
        async with app_state.pool.acquire() as conn:
            # Create messages table if it doesn't exist
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages
                (
                    id           TEXT PRIMARY KEY,
                    queue        TEXT    NOT NULL,
                    body         JSONB   NOT NULL,
                    available_at BIGINT  NOT NULL,
                    lease_until  BIGINT,
                    attempts     INTEGER NOT NULL DEFAULT 0,
                    state        TEXT    NOT NULL DEFAULT 'ready',
                    created_at   BIGINT  NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
                )
                """
            )
            
            # Create indexes for efficient queries
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_queue_avail 
                ON messages(queue, available_at) WHERE state = 'ready'
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_state 
                ON messages(state)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_lease 
                ON messages(lease_until) WHERE state = 'leased'
                """
            )
            
            print("✓ Database schema initialized")
            
            # Get message count
            count = await conn.fetchval("SELECT COUNT(*) FROM messages")
            print(f"✓ {count} messages in queue")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to database: {e}")
        print("   Server will start but API will fail until database is available")

    print("\n✅ Server ready!")
    print("=" * 70 + "\n")
    yield

    # Cleanup: Stop ingestion job if running
    if app_state.ingestion_process and app_state.ingestion_process.is_alive():
        print("\n🛑 Stopping ingestion job...")
        app_state.ingestion_process.terminate()
        app_state.ingestion_process.join(timeout=5)
        if app_state.ingestion_process.is_alive():
            app_state.ingestion_process.kill()

    # Close database pool
    if app_state.pool:
        await app_state.pool.close()
        print("✓ Database pool closed")

    print("\n👋 Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Fetchcraft Admin",
    description="Web-based administration interface for Fetchcraft document ingestion",
    version="0.1.0",
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
# API Routes
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
    return {"status": "healthy" if db_healthy else "degraded", "database_connected": db_healthy}


@app.get("/api/messages", response_model=MessagesListResponse)
async def list_messages(
    state: Optional[str] = Query(None, description="Filter by state (done, ready, processing, failed)"),
    queue: Optional[str] = Query(None, description="Filter by queue (ingest.main, ingest.deferred, ingest.error)"),
    limit: int = Query(50, ge=1, le=1000, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    List messages in the ingestion queue with optional filtering.
    
    Args:
        state: Filter by message state. Use 'done' or 'ready' or leave empty for all
        queue: Filter by queue name. Leave empty for all queues
        limit: Maximum number of messages to return
        offset: Offset for pagination
    
    Returns:
        Paginated list of messages
    """
    try:
        pool = await get_db_pool()

        # Build query with filters
        where_clauses = []
        params = []
        param_index = 1

        if state and state.lower() != "all":
            # Map "processing" to "leased" for database query
            db_state = "leased" if state.lower() == "processing" else state
            where_clauses.append(f"state = ${param_index}")
            params.append(db_state)
            param_index += 1

        if queue and queue.lower() != "all":
            where_clauses.append(f"queue = ${param_index}")
            params.append(queue)
            param_index += 1

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        async with pool.acquire() as conn:
            # Get total count
            count_query = f"SELECT COUNT(*) FROM messages{where_sql}"
            total = await conn.fetchval(count_query, *params)

            # Get paginated results
            query = f"""
                SELECT id, queue, body, available_at, lease_until, attempts, state
                FROM messages
                {where_sql}
                ORDER BY available_at DESC
                LIMIT ${param_index} OFFSET ${param_index + 1}
            """
            rows = await conn.fetch(query, *params, limit, offset)

            messages = []
            for row in rows:
                body_data = parse_message_body(row["body"])
                messages.append(
                    MessageResponse(
                        id=row["id"],
                        queue=row["queue"],
                        state=row["state"],
                        attempts=row["attempts"],
                        available_at=format_timestamp(row["available_at"]),
                        lease_until=format_timestamp(row["lease_until"]),
                        body_preview=(
                            str(body_data)[:100] + "..."
                            if len(str(body_data)) > 100
                            else str(body_data)
                        ),
                    )
                )

            return MessagesListResponse(
                messages=messages,
                total=total,
                limit=limit,
                offset=offset,
                has_more=offset + limit < total,
            )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing messages: {str(e)}")


@app.get("/api/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """
    Get statistics about the ingestion queues.
    
    Returns:
        Queue statistics including message counts by state and queue
    """
    try:
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            # Total messages
            total = await conn.fetchval("SELECT COUNT(*) FROM messages")

            # Messages by state
            state_rows = await conn.fetch(
                """
                SELECT state, COUNT(*) as count
                FROM messages
                GROUP BY state
            """
            )
            by_state = {row["state"]: row["count"] for row in state_rows}

            # Messages by queue
            queue_rows = await conn.fetch(
                """
                SELECT queue, COUNT(*) as count
                FROM messages
                GROUP BY queue
            """
            )
            by_queue = {row["queue"]: row["count"] for row in queue_rows}

            # Failed messages (high retry count)
            failed = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE attempts >= 3
            """
            )

            # Oldest pending message
            oldest = await conn.fetchval(
                """
                SELECT MIN(available_at)
                FROM messages
                WHERE state = 'ready'
            """
            )

            return QueueStatsResponse(
                total_messages=total,
                by_state=by_state,
                by_queue=by_queue,
                failed_messages=failed,
                oldest_pending=format_timestamp(oldest) if oldest else None,
            )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/api/ingestion/start", response_model=IngestionStatusResponse)
async def start_ingestion():
    """
    Start the ingestion job.
    
    Returns:
        Status of the ingestion job
    """
    if app_state.ingestion_process and app_state.ingestion_process.is_alive():
        raise HTTPException(status_code=400, detail="Ingestion job is already running")

    try:
        print("Starting ingestion in a separate process...")
        app_state.ingestion_process = Process(
            target=_ingestion_worker, name="ingestion-process"
        )
        app_state.ingestion_process.start()
        app_state.ingestion_status = "running"

        return IngestionStatusResponse(
            status="running", pid=app_state.ingestion_process.pid
        )

    except Exception as e:
        traceback.print_exc()
        app_state.ingestion_status = "error"
        raise HTTPException(
            status_code=500, detail=f"Error starting ingestion: {str(e)}"
        )


@app.post("/api/ingestion/stop", response_model=IngestionStatusResponse)
async def stop_ingestion():
    """
    Stop the ingestion job.
    
    Returns:
        Status of the ingestion job
    """
    if not app_state.ingestion_process or not app_state.ingestion_process.is_alive():
        raise HTTPException(status_code=400, detail="No ingestion job is running")

    try:
        print("Stopping ingestion process...")
        app_state.ingestion_process.terminate()
        app_state.ingestion_process.join(timeout=5)

        if app_state.ingestion_process.is_alive():
            print("Force killing ingestion process...")
            app_state.ingestion_process.kill()
            app_state.ingestion_process.join()

        app_state.ingestion_status = "stopped"
        app_state.ingestion_process = None

        return IngestionStatusResponse(status="stopped", pid=None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping ingestion: {str(e)}")


@app.get("/api/ingestion/status", response_model=IngestionStatusResponse)
async def get_ingestion_status():
    """
    Get the current status of the ingestion job.
    
    Returns:
        Status of the ingestion job (running, stopped, or error)
    """
    is_running = app_state.ingestion_process and app_state.ingestion_process.is_alive()

    if is_running:
        status = "running"
        pid = app_state.ingestion_process.pid
    else:
        # If process exists but is not alive, it finished or crashed
        if app_state.ingestion_process is not None:
            app_state.ingestion_status = "stopped"
            app_state.ingestion_process = None
        status = app_state.ingestion_status
        pid = None

    return IngestionStatusResponse(status=status, pid=pid)


# ============================================================================
# Static File Serving (for production)
# ============================================================================

# Get the directory where this file is located
PACKAGE_DIR = Path(__file__).parent
FRONTEND_DIST = PACKAGE_DIR / "frontend" / "dist"

# Serve static files if the frontend build exists
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        """Serve the main index.html file."""
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            {"message": "Frontend not built. Please run 'npm run build' in the frontend directory."},
            status_code=404,
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - return index.html for all non-API routes."""
        # Don't serve index for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Try to serve the specific file first
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise return index.html for SPA routing
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        return JSONResponse(
            {"message": "Frontend not built. Please run 'npm run build' in the frontend directory."},
            status_code=404,
        )
else:
    @app.get("/")
    async def no_frontend():
        """Fallback when frontend is not built."""
        return JSONResponse(
            {
                "message": "Fetchcraft Admin API is running",
                "note": "Frontend not built. Please run 'npm run build' in the frontend directory.",
                "api_docs": "/docs",
            }
        )


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run the admin server."""
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 Starting Fetchcraft Admin Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    db_display = POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL
    print(f"  • Database: {db_display}")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print(f"  • Documents: {DOCUMENTS_PATH}")
    print("=" * 70 + "\n")

    uvicorn.run(
        "fetchcraft.admin.server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
