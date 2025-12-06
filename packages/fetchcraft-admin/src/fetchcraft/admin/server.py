"""
Fetchcraft Admin Server - Main server class.

This module provides the FetchcraftAdminServer class that manages
the FastAPI application, handlers, and shared resources.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence

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

from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
from fetchcraft.ingestion.repository import (
    PostgresJobRepository,
    PostgresDocumentRepository,
    PostgresTaskRepository,
)

from fetchcraft.admin.config import FetchcraftAdminConfig
from fetchcraft.admin.context import HandlerContext
from fetchcraft.admin.handler import FetchcraftAdminHandler


class FetchcraftAdminServer:
    """
    Main server class for Fetchcraft Admin.
    
    This class manages:
    - FastAPI application lifecycle
    - Shared resources (database pool, queue backend, repositories)
    - Handler registration and lifecycle
    - Static file serving for frontend
    
    Example:
        config = MyConfig()
        handler = MyIngestionHandler(pipeline_factory=MyPipelineFactory())
        
        server = FetchcraftAdminServer(
            handlers=[handler],
            config=config,
        )
        server.run()
    """
    
    def __init__(
        self,
        handlers: Sequence[FetchcraftAdminHandler],
        config: FetchcraftAdminConfig,
        title: str = "Fetchcraft Admin",
        description: str = "Fetchcraft Administration Interface",
        version: str = "2.0.0",
        frontend_dist: Optional[Path] = None,
    ):
        """
        Initialize the admin server.
        
        Args:
            handlers: List of handler modules to register
            config: Server configuration
            title: API title for OpenAPI docs
            description: API description for OpenAPI docs
            version: API version
            frontend_dist: Path to frontend dist directory (optional)
        """
        self._handlers = list(handlers)
        self._config = config
        self._title = title
        self._description = description
        self._version = version
        self._frontend_dist = frontend_dist
        
        # Shared resources (initialized during startup)
        self._pool: Optional[asyncpg.Pool] = None
        self._queue_backend: Optional[AsyncPostgresQueue] = None
        self._job_repo: Optional[PostgresJobRepository] = None
        self._doc_repo: Optional[PostgresDocumentRepository] = None
        self._task_repo: Optional[PostgresTaskRepository] = None
        self._context: Optional[HandlerContext] = None
        
        # Setup logging
        self._logger = self._setup_logging()
        
        # Create FastAPI app
        self._app = self._create_app()
    
    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self._app
    
    @property
    def config(self) -> FetchcraftAdminConfig:
        """Get the server configuration."""
        return self._config
    
    @property
    def context(self) -> Optional[HandlerContext]:
        """Get the handler context (available after startup)."""
        return self._context
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging to output to both console and file."""
        self._config.logdir.mkdir(parents=True, exist_ok=True)
        
        log_file = self._config.logdir / f"fetchcraft-admin-{datetime.now().strftime('%Y%m%d')}.log"
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Logging configured: level={self._config.log_level}, file={log_file}")
        
        return module_logger
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title=self._title,
            description=self._description,
            version=self._version,
            lifespan=self._lifespan,
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Application lifespan manager."""
        self._logger.info("=" * 70)
        self._logger.info("🚀 Fetchcraft Admin Server - Initializing")
        self._logger.info("=" * 70)
        
        self._logger.info(f"🗄️  Database: {self._config.database_display}")
        self._logger.info(f"🌐 Server: http://{self._config.host}:{self._config.port}")
        self._logger.info(f"📝 Logs: {self._config.logdir.resolve()}")
        self._logger.info(f"📦 Handlers: {[h.get_name() for h in self._handlers]}")
        
        try:
            await self._initialize_shared_resources()
            await self._initialize_handlers()
            self._register_routes()
            self._setup_static_files()
            
            self._logger.info("✅ Server ready!")
            self._logger.info("=" * 70)
            
        except Exception as e:
            self._logger.error(f"⚠️  Error during initialization: {e}", exc_info=True)
            raise
        
        yield
        
        await self._shutdown_handlers()
        await self._cleanup_shared_resources()
        
        self._logger.info("👋 Server shutdown complete")
    
    async def _initialize_shared_resources(self) -> None:
        """Initialize shared resources (pool, queue, repositories)."""
        min_pool_size = max(self._config.pool_min_size, 5)
        max_pool_size = max(self._config.pool_max_size, self._config.num_workers * 3 + 10)
        
        self._logger.info(
            f"📊 Connection pool size: {min_pool_size}-{max_pool_size} "
            f"(based on {self._config.num_workers} workers)"
        )
        
        self._pool = await asyncpg.create_pool(
            self._config.postgres_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            command_timeout=60,
        )
        self._logger.info(f"✓ Connected to PostgreSQL")
        
        self._job_repo = PostgresJobRepository(self._pool)
        self._doc_repo = PostgresDocumentRepository(self._pool)
        self._task_repo = PostgresTaskRepository(self._pool)
        self._logger.info("✓ Repositories initialized")
        
        await self._job_repo._ensure_schema()
        await self._doc_repo._ensure_schema()
        await self._task_repo._ensure_schema()
        self._logger.info("✓ Database schemas initialized")
        
        self._queue_backend = AsyncPostgresQueue(pool=self._pool)
        await self._queue_backend._init_db()
        self._logger.info("✓ Queue backend initialized")
        
        self._context = HandlerContext(
            config=self._config,
            pool=self._pool,
            queue_backend=self._queue_backend,
            job_repo=self._job_repo,
            doc_repo=self._doc_repo,
            task_repo=self._task_repo,
        )
    
    async def _initialize_handlers(self) -> None:
        """Initialize all registered handlers."""
        for handler in self._handlers:
            self._logger.info(f"🔧 Initializing handler: {handler.get_name()}")
            await handler.on_startup(self._context)
            self._logger.info(f"✓ Handler '{handler.get_name()}' initialized")
    
    def _register_routes(self) -> None:
        """Register routes from all handlers."""
        # First register API routes with their prefixes
        for handler in self._handlers:
            router = handler.get_router()
            prefix = handler.get_router_prefix()
            self._app.include_router(router, prefix=prefix)
            self._logger.info(f"✓ Registered API routes for '{handler.get_name()}' at {prefix}")
        
        # Then register UI routes at root (must be last to avoid catching API routes)
        for handler in self._handlers:
            ui_router = handler.get_ui_router()
            if ui_router is not None:
                self._app.include_router(ui_router)
                self._logger.info(f"✓ Registered UI routes for '{handler.get_name()}' at /")
    
    def _setup_static_files(self) -> None:
        """Setup static file serving for frontend."""
        if self._frontend_dist and self._frontend_dist.exists():
            assets_dir = self._frontend_dist / "assets"
            if assets_dir.exists():
                self._app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
                self._logger.info(f"✓ Static files mounted from {self._frontend_dist}")
    
    async def _shutdown_handlers(self) -> None:
        """Shutdown all handlers."""
        for handler in reversed(self._handlers):
            try:
                self._logger.info(f"🔧 Shutting down handler: {handler.get_name()}")
                await handler.on_shutdown()
                self._logger.info(f"✓ Handler '{handler.get_name()}' shutdown complete")
            except Exception as e:
                self._logger.error(f"Error shutting down handler '{handler.get_name()}': {e}")
    
    async def _cleanup_shared_resources(self) -> None:
        """Cleanup shared resources."""
        if self._pool:
            await self._pool.close()
            self._logger.info("✓ Database pool closed")
    
    def run(self) -> None:
        """Run the server using uvicorn."""
        import uvicorn
        
        self._logger.info("=" * 70)
        self._logger.info("🚀 Starting Fetchcraft Admin Server")
        self._logger.info("=" * 70)
        self._logger.info(f"Configuration:")
        self._logger.info(f"  • Database: {self._config.database_display}")
        self._logger.info(f"  • Host: {self._config.host}")
        self._logger.info(f"  • Port: {self._config.port}")
        self._logger.info(f"  • Workers: {self._config.num_workers}")
        self._logger.info(f"  • Callback Base URL: {self._config.callback_base_url}")
        self._logger.info(f"  • Logs: {self._config.logdir.resolve()}")
        self._logger.info("=" * 70)
        
        uvicorn.run(
            self._app,
            host=self._config.host,
            port=self._config.port,
            log_level="info",
        )
