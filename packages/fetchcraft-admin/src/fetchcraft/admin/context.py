"""
Handler context for Fetchcraft Admin Server.

This module provides the HandlerContext class that holds shared resources
passed to handlers during their lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg
    from fetchcraft.ingestion.interfaces import QueueBackend
    from fetchcraft.ingestion.repository import (
        JobRepository,
        DocumentRepository,
        TaskRepository,
    )
    from fetchcraft.admin.config import FetchcraftAdminConfig


@dataclass
class HandlerContext:
    """
    Context passed to handlers during lifecycle events.
    
    This class holds shared resources that are initialized by the server
    and made available to all handlers. Handlers should use these resources
    rather than creating their own connections.
    
    Attributes:
        config: The server configuration
        pool: PostgreSQL connection pool
        queue_backend: Queue backend for message passing
        job_repo: Repository for job persistence
        doc_repo: Repository for document tracking
        task_repo: Repository for task tracking
        extras: Additional handler-specific data
    """
    config: "FetchcraftAdminConfig"
    pool: "asyncpg.Pool"
    queue_backend: "QueueBackend"
    job_repo: "JobRepository"
    doc_repo: "DocumentRepository"
    task_repo: "TaskRepository"
    extras: dict[str, Any] = field(default_factory=dict)
    
    def get_extra(self, key: str, default: Any = None) -> Any:
        """Get an extra value by key."""
        return self.extras.get(key, default)
    
    def set_extra(self, key: str, value: Any) -> None:
        """Set an extra value."""
        self.extras[key] = value
