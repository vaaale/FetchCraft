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
from fetchcraft.mcp import FetchcraftMCPServer
from fetchcraft.mcp.config import FetchcraftMCPConfig


def init_server(*args):
    """
    Factory function for creating the server app.
    
    This is used by uvicorn when running with --factory flag.
    For new code, prefer using FetchcraftMCPServer directly.
    """
    from fetchcraft.mcp.services import DefaultFindFilesService, DefaultQueryService, DefaultDocumentPreviewService
    
    config = FetchcraftMCPConfig()
    
    find_files_service = DefaultFindFilesService.create(config)
    query_service = DefaultQueryService.create(config)
    document_preview_service = DefaultDocumentPreviewService.create(config)
    
    server = FetchcraftMCPServer(
        find_files_service=find_files_service,
        query_service=query_service,
        document_preview_service=document_preview_service,
        config=config,
        title="Fetchcraft Files MCP Server",
    )
    server.run()


if __name__ == "__main__":
    init_server()

