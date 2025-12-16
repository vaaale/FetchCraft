"""
Fetchcraft MCP Server - A framework for building MCP servers with file search and RAG capabilities.

This package provides:
- FetchcraftMCPServer: Main server class with dependency injection
- FetchcraftMCPConfig: Configuration class for server settings
- Service interfaces: FindFilesService, QueryService
- Default implementations: DefaultFindFilesService, DefaultQueryService

Example usage:
    from fetchcraft.mcp import FetchcraftMCPServer, FetchcraftMCPConfig
    from fetchcraft.mcp.services import DefaultFindFilesService, DefaultQueryService
    
    # Create config
    config = FetchcraftMCPConfig()
    
    # Create services
    find_files = DefaultFindFilesService.create(config)
    query_service = DefaultQueryService.create(config)
    
    # Create and run server
    server = FetchcraftMCPServer(
        find_files_service=find_files,
        query_service=query_service,
        config=config,
    )
    server.run()
"""
from fetchcraft.mcp.server import FetchcraftMCPServer
from fetchcraft.mcp.config import FetchcraftMCPConfig
from fetchcraft.mcp.interface import FindFilesService, QueryService, QueryResponse

__all__ = [
    "FetchcraftMCPServer",
    "FetchcraftMCPConfig",
    "FindFilesService",
    "QueryService",
    "QueryResponse",
]
