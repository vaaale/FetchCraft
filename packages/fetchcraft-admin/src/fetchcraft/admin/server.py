"""
Fetchcraft Admin Server V2 - Enhanced job and document tracking.

This FastAPI application provides a web interface for:
- Creating and managing ingestion jobs
- Tracking documents through pipeline steps
- Viewing job and document status
- Retrying failed documents
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "asyncpg is required for PostgreSQL backend. "
        "Install it with: pip install asyncpg"
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
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
from fetchcraft.admin.services.worker_manager import WorkerManager
from fetchcraft.admin.api.endpoints import main_router, ui_router
from fetchcraft.admin.config import settings
from fetchcraft.ingestion.repository import PostgresTaskRepository

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
    task_repo: Optional[PostgresTaskRepository] = None
    queue_backend: Optional[AsyncPostgresQueue] = None
    worker_manager: Optional[WorkerManager] = None

app_state = AppState()

# =============================================================================
# Application Lifespan
# =============================================================================

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
        app_state.task_repo = PostgresTaskRepository(app_state.pool)
        logger.info("✓ Repositories initialized")
        
        # Initialize database schemas
        await app_state.job_repo._ensure_schema()
        await app_state.doc_repo._ensure_schema()
        await app_state.task_repo._ensure_schema()
        logger.info("✓ Database schemas initialized")
        
        # Initialize queue backend - IMPORTANT: Pass the shared pool
        app_state.queue_backend = AsyncPostgresQueue(pool=app_state.pool)
        # Initialize the messages table (required when using external pool)
        await app_state.queue_backend._init_db()
        logger.info("✓ Queue backend initialized (using shared connection pool)")
        
        # Build callback base URL
        callback_base_url = f"http://{settings.host}:{settings.port}"
        
        # Initialize pipeline factory
        pipeline_factory = IngestionPipelineFactory(
            queue_backend=app_state.queue_backend,
            job_repo=app_state.job_repo,
            doc_repo=app_state.doc_repo,
            document_root=settings.documents_path,
            num_workers=settings.num_workers,
            task_repo=app_state.task_repo,
            callback_base_url=callback_base_url,
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
        
        # Initialize worker manager
        app_state.worker_manager = WorkerManager(
            queue_backend=app_state.queue_backend,
            job_repo=app_state.job_repo,
            doc_repo=app_state.doc_repo,
            task_repo=app_state.task_repo,
            pipeline_factory=pipeline_factory,
            callback_base_url=callback_base_url,
        )
        logger.info("✓ Worker manager initialized")
        
        # Link ingestion service to worker manager for callback handling
        app_state.ingestion_service.worker_manager = app_state.worker_manager
        
        # Start worker manager and recover any pending jobs
        logger.info("🔄 Starting worker manager and recovering jobs...")
        deps = _get_ingestion_dependencies()
        recovered_count = await app_state.worker_manager.start(
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
    if app_state.worker_manager:
        await app_state.worker_manager.stop()
        logger.info("✓ Worker manager stopped")
    
    if app_state.pool:
        await app_state.pool.close()
        logger.info("✓ Database pool closed")
    
    logger.info("👋 Server shutdown complete")


# =============================================================================
# Helper Functions
# =============================================================================

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
    callback_url = f"{settings.callback_base_url}/api/tasks/callback"
    
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


# =============================================================================
# FastAPI Application
# =============================================================================

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

# Include routers
app.include_router(main_router)

# Static file serving for frontend
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

# Include UI router last (catch-all routes)
app.include_router(ui_router)

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
        "fetchcraft.admin.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )

if __name__ == "__main__":
    main()
