"""
Default service implementations for the Fetchcraft MCP Server.

This module provides ready-to-use implementations of the service interfaces
using Qdrant vector store and MongoDB document store.
"""
from fetchcraft.mcp.services.find_files import DefaultFindFilesService
from fetchcraft.mcp.services.query import DefaultQueryService

__all__ = [
    "DefaultFindFilesService",
    "DefaultQueryService",
]
