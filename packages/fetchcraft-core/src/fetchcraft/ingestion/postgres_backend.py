"""
Async PostgreSQL Queue Backend for Ingestion Pipeline

Uses asyncpg for native async PostgreSQL operations.
Provides durable, lease-based message queue with automatic lease expiry.

Schema:
  messages(id TEXT PK, queue TEXT, body JSONB, available_at BIGINT, 
           lease_until BIGINT, attempts INTEGER, state TEXT)
  
States: 'ready', 'leased', 'done'

Usage:
    backend = AsyncPostgresQueue(
        connection_string="postgresql://user:pass@localhost:5432/dbname"
    )
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "asyncpg is required for PostgreSQL backend. "
        "Install it with: pip install asyncpg"
    )

from fetchcraft.ingestion.base import AsyncQueueBackend, QueueMessage, to_json, from_json


def _sanitize_for_postgres(obj):
    """
    Recursively sanitize data to remove null bytes which PostgreSQL cannot store.
    
    PostgreSQL TEXT and JSONB string values cannot contain \\u0000 (null bytes).
    This function removes them from all strings in the data structure.
    
    Args:
        obj: Any Python object (dict, list, str, etc.)
        
    Returns:
        Sanitized version of the object
    """
    if isinstance(obj, str):
        # Remove null bytes from strings
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {k: _sanitize_for_postgres(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_postgres(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_postgres(item) for item in obj)
    else:
        # For other types (int, float, bool, None, etc.), return as-is
        return obj


class AsyncPostgresQueue(AsyncQueueBackend):
    """
    PostgreSQL-backed async queue implementation.
    
    Uses PostgreSQL's FOR UPDATE SKIP LOCKED for efficient message leasing
    and JSONB for flexible message body storage.
    
    Args:
        connection_string: PostgreSQL connection string (optional if pool provided)
        pool_min_size: Minimum number of connections in the pool (ignored if pool provided)
        pool_max_size: Maximum number of connections in the pool (ignored if pool provided)
        pool: Existing asyncpg Pool to use (recommended for sharing pools)
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool_min_size: int = 10,
        pool_max_size: int = 20,
        pool: Optional[asyncpg.Pool] = None,
    ):
        if pool is None and connection_string is None:
            raise ValueError("Either connection_string or pool must be provided")
        
        self.connection_string = connection_string
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: Optional[asyncpg.Pool] = pool
        self._owns_pool = pool is None  # Track if we created the pool
    
    async def _ensure_pool(self) -> asyncpg.Pool:
        """Lazily create connection pool on first use (if not provided)."""
        if self._pool is None:
            if self.connection_string is None:
                raise ValueError("Cannot create pool without connection_string")
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60,
            )
            await self._init_db()
        return self._pool
    
    async def _init_db(self):
        """Create messages table and indexes if they don't exist."""
        if self._pool is None:
            return
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages
                    (
                        id           TEXT PRIMARY KEY,
                        queue        TEXT    NOT NULL,
                        body         JSONB   NOT NULL,
                        available_at BIGINT  NOT NULL,
                        lease_until  BIGINT,
                        attempts     INTEGER NOT NULL DEFAULT 0,
                        state        TEXT    NOT NULL DEFAULT 'ready',
                        created_at   BIGINT  NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
                    )
                    """
                )
            except asyncpg.exceptions.UniqueViolationError as e:
                # Table and its composite type already exist, this is fine
                if "pg_type_typname_nsp_index" in str(e):
                    pass
                else:
                    raise
            
            # Create indexes for efficient queries
            # Wrap each in try-except to handle concurrent creation
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_queue_avail "
                    "ON messages(queue, available_at) WHERE state = 'ready'"
                )
            except asyncpg.exceptions.UniqueViolationError:
                pass  # Index already exists from concurrent creation
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_state "
                    "ON messages(state)"
                )
            except asyncpg.exceptions.UniqueViolationError:
                pass  # Index already exists from concurrent creation
            
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_lease "
                    "ON messages(lease_until) WHERE state = 'leased'"
                )
            except asyncpg.exceptions.UniqueViolationError:
                pass  # Index already exists from concurrent creation
    
    async def enqueue(self, queue_name: str, body: dict, delay_seconds: int = 0) -> str:
        """
        Add a message to the queue.
        
        Args:
            queue_name: Name of the queue
            body: Message body (will be stored as JSONB)
            delay_seconds: Delay before message becomes available
            
        Returns:
            Message ID (UUID)
        """
        pool = await self._ensure_pool()
        now = int(time.time())
        available_at = now + max(0, int(delay_seconds))
        message_id = str(uuid.uuid4())
        
        # Sanitize body to remove null bytes that PostgreSQL cannot store
        sanitized_body = _sanitize_for_postgres(body)
        body_json = to_json(sanitized_body)
        
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, queue, body, available_at, attempts, state, created_at)
                VALUES ($1, $2, $3::jsonb, $4, 0, 'ready', $5)
                """,
                message_id,
                queue_name,
                body_json,
                available_at,
                now,
            )
        
        return message_id
    
    async def lease_next(self, queue_name: str, lease_seconds: int = 30) -> Optional[QueueMessage]:
        """
        Lease the next available message from the queue.
        
        Uses PostgreSQL's FOR UPDATE SKIP LOCKED for efficient, lock-free leasing.
        Automatically reaps expired leases before attempting to lease.
        
        Args:
            queue_name: Name of the queue
            lease_seconds: Duration of the lease in seconds
            
        Returns:
            QueueMessage if available, None otherwise
        """
        pool = await self._ensure_pool()
        
        # First, reap any expired leases
        await self._reap_expired()
        
        now = int(time.time())
        lease_until = now + max(1, int(lease_seconds))
        
        async with pool.acquire() as conn:
            # Use FOR UPDATE SKIP LOCKED for efficient message leasing
            # This prevents lock contention between multiple workers
            row = await conn.fetchrow(
                """
                UPDATE messages
                SET state = 'leased',
                    lease_until = $1,
                    attempts = attempts + 1
                WHERE id = (
                    SELECT id
                    FROM messages
                    WHERE queue = $2
                      AND state = 'ready'
                      AND available_at <= $3
                    ORDER BY available_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, body
                """,
                lease_until,
                queue_name,
                now,
            )
            
            if row:
                return QueueMessage(
                    id=row["id"],
                    body=from_json(row["body"]),
                )
            
            return None
    
    async def ack(self, queue_name: str, message_id: str) -> None:
        """
        Acknowledge successful processing of a message.
        
        Marks the message as 'done', removing it from active processing.
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to acknowledge
        """
        pool = await self._ensure_pool()
        
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE messages
                SET state = 'done'
                WHERE id = $1 AND queue = $2
                """,
                message_id,
                queue_name,
            )
    
    async def nack(self, queue_name: str, message_id: str, requeue_delay_seconds: int = 0) -> None:
        """
        Negative acknowledge - return message to queue for retry.
        
        Resets the message state to 'ready' with optional delay.
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to requeue
            requeue_delay_seconds: Delay before message becomes available again
        """
        pool = await self._ensure_pool()
        available_at = int(time.time()) + max(0, int(requeue_delay_seconds))
        
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE messages
                SET state = 'ready',
                    lease_until = NULL,
                    available_at = $1
                WHERE id = $2 AND queue = $3
                """,
                available_at,
                message_id,
                queue_name,
            )
    
    async def has_pending(self, *queue_names: str) -> bool:
        """
        Check if any of the specified queues have pending messages.
        
        Considers both 'ready' and 'leased' messages as pending.
        Automatically reaps expired leases before checking.
        
        Args:
            queue_names: Names of queues to check
            
        Returns:
            True if any queue has pending messages
        """
        pool = await self._ensure_pool()
        
        # Reap expired leases first
        await self._reap_expired()
        
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE queue = ANY($1::text[])
                  AND state IN ('ready', 'leased')
                """,
                list(queue_names),
            )
            
            return count > 0
    
    async def _reap_expired(self) -> int:
        """
        Reap expired leases, returning them to 'ready' state.
        
        This is called automatically by lease_next and has_pending.
        
        Returns:
            Number of messages reaped
        """
        if self._pool is None:
            return 0
        
        now = int(time.time())
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE messages
                SET state = 'ready',
                    lease_until = NULL
                WHERE state = 'leased'
                  AND lease_until <= $1
                """,
                now,
            )
            
            # Extract count from result string like "UPDATE 5"
            return int(result.split()[-1]) if result else 0
    
    async def get_stats(self) -> dict:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics including counts by state and queue
        """
        pool = await self._ensure_pool()
        
        async with pool.acquire() as conn:
            # Total messages
            total = await conn.fetchval("SELECT COUNT(*) FROM messages")
            
            # By state
            by_state_rows = await conn.fetch(
                "SELECT state, COUNT(*) as count FROM messages GROUP BY state"
            )
            by_state = {row["state"]: row["count"] for row in by_state_rows}
            
            # By queue
            by_queue_rows = await conn.fetch(
                "SELECT queue, COUNT(*) as count FROM messages GROUP BY queue"
            )
            by_queue = {row["queue"]: row["count"] for row in by_queue_rows}
            
            # Failed messages (high retry count)
            failed = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE attempts >= 3"
            )
            
            # Oldest pending message
            oldest_row = await conn.fetchrow(
                "SELECT MIN(available_at) as oldest FROM messages WHERE state = 'ready'"
            )
            oldest_pending = oldest_row["oldest"] if oldest_row and oldest_row["oldest"] else None
            
            return {
                "total_messages": total or 0,
                "by_state": by_state,
                "by_queue": by_queue,
                "failed_messages": failed or 0,
                "oldest_pending": oldest_pending,
            }
    
    async def check_health(self) -> bool:
        """
        Check if the queue backend is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception:
            return False
    
    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def clear_queue(self, queue_name: str, state: Optional[str] = None) -> int:
        """
        Clear messages from a queue.
        
        Args:
            queue_name: Name of the queue to clear
            state: Optional state filter (only delete messages in this state)
            
        Returns:
            Number of messages deleted
        """
        pool = await self._ensure_pool()
        
        async with pool.acquire() as conn:
            if state:
                result = await conn.execute(
                    "DELETE FROM messages WHERE queue = $1 AND state = $2",
                    queue_name,
                    state,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM messages WHERE queue = $1",
                    queue_name,
                )
            
            return int(result.split()[-1]) if result else 0
    
    async def delete_message(self, message_id: str) -> bool:
        """
        Delete a specific message.
        
        Args:
            message_id: ID of the message to delete
            
        Returns:
            True if message was deleted, False if not found
        """
        pool = await self._ensure_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM messages WHERE id = $1",
                message_id,
            )
            
            return int(result.split()[-1]) > 0 if result else False
    
    async def close(self):
        """Close the connection pool if we own it."""
        if self._pool and self._owns_pool:
            await self._pool.close()
            self._pool = None
