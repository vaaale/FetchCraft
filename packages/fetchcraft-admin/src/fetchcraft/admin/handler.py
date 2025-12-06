"""
Base handler interface for Fetchcraft Admin Server.

This module provides the abstract base class that all handler modules
must implement to integrate with the admin server.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fetchcraft.admin.context import HandlerContext


class FetchcraftAdminHandler(ABC):
    """
    Abstract base class for admin server handler modules.
    
    Handlers are self-contained modules that provide specific functionality
    to the admin server. Each handler:
    - Provides a FastAPI router with its endpoints
    - Has lifecycle hooks for startup and shutdown
    - Receives shared resources via HandlerContext
    
    Example:
        class MyHandler(FetchcraftAdminHandler):
            def get_name(self) -> str:
                return "my-handler"
            
            def get_router(self) -> APIRouter:
                return self._router
            
            async def on_startup(self, context: HandlerContext) -> None:
                # Initialize handler-specific resources
                pass
            
            async def on_shutdown(self) -> None:
                # Cleanup resources
                pass
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the unique name of this handler.
        
        This name is used for logging and identification purposes.
        
        Returns:
            Handler name (e.g., "ingestion", "search", "admin")
        """
        pass
    
    @abstractmethod
    def get_router(self) -> "APIRouter":
        """
        Get the FastAPI router for this handler.
        
        The router should contain all endpoints provided by this handler.
        The server will include this router in the main application.
        
        Returns:
            FastAPI APIRouter instance
        """
        pass
    
    @abstractmethod
    async def on_startup(self, context: "HandlerContext") -> None:
        """
        Called when the server starts up.
        
        This method is called after the server has initialized shared
        resources (database pool, queue backend, etc.). Handlers should
        use this to initialize their own resources using the provided context.
        
        Args:
            context: HandlerContext with shared resources
        """
        pass
    
    @abstractmethod
    async def on_shutdown(self) -> None:
        """
        Called when the server shuts down.
        
        Handlers should use this to cleanup any resources they created
        during startup or operation.
        """
        pass
    
    def get_router_prefix(self) -> str:
        """
        Get the URL prefix for this handler's router.
        
        Override this method to customize the URL prefix.
        Default is "/api".
        
        Returns:
            URL prefix string
        """
        return "/api"
    
    def get_ui_router(self) -> Optional["APIRouter"]:
        """
        Get an optional UI router for serving frontend assets.
        
        This router is registered at the root level (no prefix) and should
        be used for serving static files and SPA routes.
        
        Override this method to provide UI routes.
        Default returns None (no UI routes).
        
        Returns:
            Optional APIRouter for UI routes, or None
        """
        return None
