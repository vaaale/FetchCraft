"""
Service interfaces for the Fetchcraft MCP Server.

This module defines abstract interfaces that can be implemented
by custom service classes for dependency injection.
"""
from fetchcraft.mcp.interface.services import FindFilesService, QueryService, QueryResponse

__all__ = [
    "FindFilesService",
    "QueryService",
    "QueryResponse",
]
