"""
Fetchcraft MCP Server - Main server class.

This module provides the FetchcraftMCPServer class that manages
the FastAPI application with MCP integration and service injection.

Example:
    from fetchcraft.mcp import FetchcraftMCPServer, FetchcraftMCPConfig
    from fetchcraft.mcp.services import DefaultFindFilesService, DefaultQueryService
    
    config = FetchcraftMCPConfig()
    find_files = DefaultFindFilesService.create(config)
    query_service = DefaultQueryService.create(config)
    
    server = FetchcraftMCPServer(
        find_files_service=find_files,
        query_service=query_service,
        config=config,
    )
    server.run()
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from fetchcraft.mcp.api import create_api_router
from fetchcraft.mcp.config import FetchcraftMCPConfig
from fetchcraft.mcp.frontend_router import create_frontend_router, mount_static_assets
from fetchcraft.mcp.iframe_middleware import IframeHeadersMiddleware
from fetchcraft.mcp.interface import FindFilesService, QueryService
from fetchcraft.mcp.mcp_api import add_tools


class FetchcraftMCPServer:
    """
    Main server class for Fetchcraft MCP Server.
    
    This class manages:
    - FastAPI application with MCP integration
    - Service injection for find_files and query functionality
    - Static file serving for frontend
    - CORS and iframe middleware configuration
    
    Example:
        config = FetchcraftMCPConfig()
        find_files = MyFindFilesService(...)
        query_service = MyQueryService(...)
        
        server = FetchcraftMCPServer(
            find_files_service=find_files,
            query_service=query_service,
            config=config,
        )
        server.run()
    """

    def __init__(
        self,
        find_files_service: FindFilesService,
        query_service: QueryService,
        config: Optional[FetchcraftMCPConfig] = None,
        title: str = "Fetchcraft MCP Server",
        description: str = "MCP Server with file search and RAG query capabilities",
        version: str = "1.0.0",
    ):
        """
        Initialize the MCP server.
        
        Args:
            find_files_service: Service for file search functionality
            query_service: Service for RAG query functionality
            config: Server configuration (uses defaults if not provided)
            title: API title for OpenAPI docs
            description: API description for OpenAPI docs
            version: API version
        """
        self._find_files_service = find_files_service
        self._query_service = query_service
        self._config = config or FetchcraftMCPConfig()
        self._title = title
        self._description = description
        self._version = version

        # Setup logging
        self._logger = self._setup_logging()

        # Create FastAPI app with MCP integration
        self._app = self._create_app()

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self._app

    @property
    def config(self) -> FetchcraftMCPConfig:
        """Get the server configuration."""
        return self._config

    def _setup_logging(self) -> logging.Logger:
        """Configure logging to output to both console and file."""
        self._config.logdir.mkdir(parents=True, exist_ok=True)

        log_file = self._config.logdir / f"fetchcraft-mcp-{datetime.now().strftime('%Y%m%d')}.log"
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

    def _get_frontend_dist(self) -> Path:
        """Get the frontend dist directory path."""
        if self._config.frontend_dist:
            return self._config.frontend_dist
        # Default: look relative to package
        return Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application with MCP integration."""
        # Create base FastAPI app
        base_app = FastAPI(
            title=self._title,
            description=self._description,
            version=self._version,
        )

        # Add API router
        api_router = create_api_router(
            find_files_service=self._find_files_service,
            query_service=self._query_service
        )
        base_app.include_router(api_router)

        # Create frontend router
        frontend_dir = self._get_frontend_dist()
        frontend_router = create_frontend_router(frontend_dir)

        # Create MCP from FastAPI
        mcp = FastMCP.from_fastapi(app=base_app, name=self._config.mcp_server_name)

        # Add MCP tools
        add_tools(
            mcp=mcp,
            find_files_service=self._find_files_service,
            server_url=self._config.frontend_base_url,
            mcp_server_name=self._config.mcp_server_name
        )

        # Create MCP HTTP app
        mcp_app = mcp.http_app(path=self._config.mcp_path)

        # Combine all routes into final app
        combined_app = FastAPI(
            title=self._title,
            description=self._description,
            version=self._version,
            routes=[
                *mcp_app.routes,  # MCP routes
                *base_app.routes,  # API routes
                *frontend_router.routes  # Frontend routes
            ],
            lifespan=mcp_app.lifespan,
        )

        # Add CORS middleware for iframe compatibility
        combined_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        combined_app.add_middleware(IframeHeadersMiddleware)

        # Mount static assets
        mount_static_assets(combined_app, frontend_dir)

        # Add health endpoint
        @combined_app.get("/health")
        async def health():
            return {"status": "ok"}

        self._logger.info(f"Created MCP server: {self._config.mcp_server_name}")

        return combined_app

    def run(self) -> None:
        """Run the server using uvicorn."""
        import uvicorn

        self._logger.info("=" * 70)
        self._logger.info("🚀 Starting Fetchcraft MCP Server")
        self._logger.info("=" * 70)
        self._logger.info(f"Configuration:")
        self._logger.info(f"  • Host: {self._config.host}")
        self._logger.info(f"  • Port: {self._config.port}")
        self._logger.info(f"  • MCP Server Name: {self._config.mcp_server_name}")
        self._logger.info(f"  • MCP Path: {self._config.mcp_path}")
        self._logger.info(f"  • Frontend URL: {self._config.frontend_base_url}")
        self._logger.info(f"  • Logs: {self._config.logdir.resolve()}")
        self._logger.info("=" * 70)

        uvicorn.run(
            self._app,
            host=self._config.host,
            port=self._config.port,
            log_level="info",
        )
