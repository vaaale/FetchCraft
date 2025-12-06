"""
Fetchcraft Ingestion Admin Handler.

This module provides the main handler class for ingestion functionality
that integrates with the FetchcraftAdminServer framework.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from fetchcraft.admin.handler import FetchcraftAdminHandler
from fetchcraft.admin.ingestion.api.router import create_ingestion_router
from fetchcraft.admin.ingestion.services.ingestion_service import IngestionService
from fetchcraft.admin.ingestion.services.worker_manager import WorkerManager

if TYPE_CHECKING:
    from fetchcraft.admin.context import HandlerContext
    from fetchcraft.admin.ingestion.pipeline_factory import FetchcraftIngestionPipelineFactory
    from fetchcraft.admin.ingestion.config import IngestionConfig

logger = logging.getLogger(__name__)


class FetchcraftIngestionAdminHandler(FetchcraftAdminHandler):
    """
    Handler for ingestion functionality in the Fetchcraft Admin Server.
    
    This handler provides:
    - Job management (create, list, delete, restart)
    - Document tracking
    - Pipeline execution
    - Worker management
    - Callback handling for async transformations
    
    Example:
        class MyPipelineFactory(FetchcraftIngestionPipelineFactory):
            def configure_pipeline(self, pipeline):
                pipeline.add_transformation(...)
                pipeline.add_sink(...)
        
        handler = FetchcraftIngestionAdminHandler(
            pipeline_factory=MyPipelineFactory(),
            ingestion_dependencies_factory=my_deps_factory,
        )
        
        server = FetchcraftAdminServer(handlers=[handler], config=config)
        server.run()
    """
    
    def __init__(
        self,
        pipeline_factory: "FetchcraftIngestionPipelineFactory",
        ingestion_dependencies_factory: Optional[Callable[["IngestionConfig"], dict]] = None,
        frontend_dist: Optional[Path] = None,
    ):
        """
        Initialize the ingestion handler.
        
        Args:
            pipeline_factory: Factory for creating ingestion pipelines
            ingestion_dependencies_factory: Optional factory function that creates
                ingestion dependencies (parser_map, chunker, index_factory).
                If not provided, a default factory will be used.
                Signature: (config: IngestionConfig) -> dict
            frontend_dist: Path to frontend dist directory for UI serving
        """
        self._pipeline_factory = pipeline_factory
        self._ingestion_dependencies_factory = ingestion_dependencies_factory
        self._frontend_dist = frontend_dist
        
        # These will be initialized during on_startup
        self._context: Optional[HandlerContext] = None
        self._ingestion_service: Optional[IngestionService] = None
        self._worker_manager: Optional[WorkerManager] = None
        self._router: Optional[APIRouter] = None
        self._ui_router: Optional[APIRouter] = None
        self._ingestion_deps: Optional[dict] = None
    
    def get_name(self) -> str:
        """Get the handler name."""
        return "ingestion"
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router for this handler."""
        if self._router is None:
            raise RuntimeError("Handler not initialized. Call on_startup first.")
        return self._router
    
    def get_router_prefix(self) -> str:
        """Get the URL prefix for this handler's router."""
        return "/api"
    
    def get_ui_router(self) -> Optional[APIRouter]:
        """Get the UI router for serving frontend (registered at root)."""
        return self._ui_router
    
    async def on_startup(self, context: "HandlerContext") -> None:
        """Initialize the handler with shared resources."""
        self._context = context
        config = self._get_config()
        
        # Initialize pipeline factory with context
        self._pipeline_factory.initialize(
            context=context,
            document_root=config.documents_path,
        )
        
        # Create ingestion dependencies
        if self._ingestion_dependencies_factory:
            self._ingestion_deps = self._ingestion_dependencies_factory(config)
        else:
            self._ingestion_deps = self._create_default_ingestion_dependencies(config)
        
        # Initialize ingestion service
        self._ingestion_service = IngestionService(
            job_repo=context.job_repo,
            doc_repo=context.doc_repo,
            pipeline_factory=self._pipeline_factory,
            document_root=config.documents_path,
            queue_backend=context.queue_backend,
        )
        
        # Initialize worker manager
        self._worker_manager = WorkerManager(
            queue_backend=context.queue_backend,
            job_repo=context.job_repo,
            doc_repo=context.doc_repo,
            task_repo=context.task_repo,
            pipeline_factory=self._pipeline_factory,
            callback_base_url=config.callback_base_url,
        )
        
        # Link services
        self._ingestion_service.worker_manager = self._worker_manager
        
        # Start worker manager and recover jobs
        logger.info("🔄 Starting worker manager and recovering jobs...")
        recovered_count = await self._worker_manager.start(
            parser_map=self._ingestion_deps["parser_map"],
            chunker=self._ingestion_deps["chunker"],
            index_factory=self._ingestion_deps["index_factory"],
            index_id=config.index_id,
        )
        if recovered_count > 0:
            logger.info(f"✓ Recovered {recovered_count} job(s)")
        
        # Create router with dependency injection
        self._router = self._create_router()
        
        # Create UI router (served at root, not under /api)
        self._ui_router = self._create_ui_router()
        
        logger.info(f"✓ Ingestion handler initialized (documents: {config.documents_path})")
    
    async def on_shutdown(self) -> None:
        """Cleanup handler resources."""
        if self._worker_manager:
            await self._worker_manager.stop()
            logger.info("✓ Worker manager stopped")
    
    def _get_config(self) -> "IngestionConfig":
        """Get the configuration cast to IngestionConfig."""
        from fetchcraft.admin.ingestion.config import IngestionConfig
        
        config = self._context.config
        if isinstance(config, IngestionConfig):
            return config
        
        # If base config, create IngestionConfig with same values
        # This allows using IngestionConfig features even if base config was passed
        return IngestionConfig(**config.model_dump())
    
    def _create_router(self) -> APIRouter:
        """Create the API router with all endpoints."""
        # Create main API router
        api_router = create_ingestion_router(
            get_ingestion_service=lambda: self._ingestion_service,
            get_worker_manager=lambda: self._worker_manager,
            get_queue_backend=lambda: self._context.queue_backend,
            get_config=self._get_config,
            get_ingestion_dependencies=lambda: self._ingestion_deps,
        )
        
        # Create combined router
        router = APIRouter()
        router.include_router(api_router)
        
        return router
    
    def _create_ui_router(self) -> APIRouter:
        """Create the UI router for serving frontend."""
        router = APIRouter()
        frontend_dist = self._frontend_dist
        
        if frontend_dist is None:
            # Try default location
            package_root = Path(__file__).parent.parent.parent.parent
            frontend_dist = package_root / "frontend" / "dist"
        
        @router.get("/", include_in_schema=False)
        async def serve_index():
            """Serve the main index.html file."""
            if not frontend_dist.exists():
                return JSONResponse({
                    "message": "Fetchcraft Admin API is running",
                    "note": "Frontend not built",
                    "api_docs": "/docs",
                })
            
            index_file = frontend_dist / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return JSONResponse(
                {"message": "Frontend not built"},
                status_code=404,
            )
        
        @router.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            """Serve SPA - return index.html for all non-API routes."""
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            
            if not frontend_dist.exists():
                return JSONResponse({
                    "message": "Fetchcraft Admin API is running",
                    "note": "Frontend not built",
                    "api_docs": "/docs",
                })
            
            file_path = frontend_dist / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            
            index_file = frontend_dist / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            
            return JSONResponse(
                {"message": "Frontend not built"},
                status_code=404,
            )
        
        return router
    
    def _create_default_ingestion_dependencies(self, config: "IngestionConfig") -> dict:
        """Create default ingestion dependencies from config."""
        from qdrant_client import QdrantClient
        
        from fetchcraft.document_store import MongoDBDocumentStore
        from fetchcraft.embeddings import OpenAIEmbeddings
        from fetchcraft.node_parser import HierarchicalNodeParser
        from fetchcraft.parsing.docling.client.docling_parser import RemoteDoclingParser
        from fetchcraft.parsing.text_file_parser import TextFileParser
        from fetchcraft.vector_store import QdrantVectorStore
        
        from fetchcraft.admin.ingestion.pipeline_factory import DefaultIndexFactory
        
        doc_store = MongoDBDocumentStore(
            connection_string=config.mongo_uri,
            database_name="fetchcraft",
            collection_name=config.collection_name,
        )
        
        embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url
        )
        
        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embeddings=embeddings,
            distance="Cosine",
            enable_hybrid=config.enable_hybrid,
            fusion_method=config.fusion_method
        )
        
        chunker = HierarchicalNodeParser(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            child_sizes=config.child_chunks,
            child_overlap=50
        )
        
        index_factory = DefaultIndexFactory(
            vector_store=vector_store,
            doc_store=doc_store,
            index_id=config.index_id
        )
        
        callback_url = f"{config.callback_base_url}/api/tasks/callback"
        
        parser_map = {
            "default": TextFileParser(),
            "application/pdf": RemoteDoclingParser(
                docling_url=config.docling_server,
                callback_url=callback_url
            )
        }
        
        return {
            "index_factory": index_factory,
            "chunker": chunker,
            "parser_map": parser_map,
        }
