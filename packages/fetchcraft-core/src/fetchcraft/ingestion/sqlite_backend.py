from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from typing import Optional

from fetchcraft.ingestion import AsyncQueueBackend
from fetchcraft.ingestion.base import to_json, QueueMessage, from_json


# -----------------------------
# Durable SQLite async queue (no extra deps)
# -----------------------------

class _SQLiteQueueSync:
    """Synchronous implementation; wrapped by AsyncSQLiteQueue using asyncio.to_thread.

    Schema:
      messages(id TEXT PK, queue TEXT, body TEXT, available_at INTEGER, lease_until INTEGER,
               attempts INTEGER, state TEXT)
    States: 'ready', 'leased', 'done'
    """

    def __init__(self, path: str = "pipeline_queue.db"):
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages
                (
                    id           TEXT PRIMARY KEY,
                    queue        TEXT    NOT NULL,
                    body         TEXT    NOT NULL,
                    available_at INTEGER NOT NULL,
                    lease_until  INTEGER,
                    attempts     INTEGER NOT NULL DEFAULT 0,
                    state        TEXT    NOT NULL DEFAULT 'ready'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_q_avail ON messages(queue, available_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON messages(state);")

    def enqueue(self, queue_name: str, body: dict, delay_seconds: int = 0) -> str:
        now = int(time.time())
        available_at = now + max(0, int(delay_seconds))
        mid = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(id, queue, body, available_at, attempts, state) VALUES(?,?,?,?,0,'ready')",
                (mid, queue_name, to_json(body), available_at),
            )
        return mid

    def lease_next(self, queue_name: str, lease_seconds: int = 30) -> Optional[QueueMessage]:
        deadline = int(time.time())
        lease_until = deadline + max(1, int(lease_seconds))
        with self._connect() as conn:
            for _ in range(5):
                row = conn.execute(
                    """
                    SELECT id, body
                    FROM messages
                    WHERE queue = ?
                      AND state = 'ready'
                      AND available_at <= ?
                    ORDER BY available_at ASC
                    LIMIT 1
                    """,
                    (queue_name, deadline),
                ).fetchone()
                if not row:
                    return None
                res = conn.execute(
                    "UPDATE messages SET state='leased', lease_until=?, attempts=attempts+1 WHERE id=? AND state='ready'",
                    (lease_until, row["id"]),
                )
                if res.rowcount:
                    return QueueMessage(id=row["id"], body=from_json(row["body"]))
        return None

    def ack(self, queue_name: str, message_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE messages SET state='done' WHERE id=? AND queue=?", (message_id, queue_name))

    def nack(self, queue_name: str, message_id: str, requeue_delay_seconds: int = 0) -> None:
        avail = int(time.time()) + max(0, int(requeue_delay_seconds))
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET state='ready', lease_until=NULL, available_at=? WHERE id=? AND queue=?",
                (avail, message_id, queue_name),
            )

    def reap_expired(self):
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET state='ready', lease_until=NULL WHERE state='leased' AND lease_until<=?",
                (now,),
            )

    def count_pending(self, queues: list[str]) -> int:
        qmarks = ",".join("?" for _ in queues)
        sql = f"""
        SELECT COUNT(*) AS c
        FROM messages
        WHERE queue IN ({qmarks})
          AND state IN ('ready', 'leased')
        """
        with self._connect() as conn:
            (c,) = conn.execute(sql, queues).fetchone()
        return int(c)


class AsyncSQLiteQueue(AsyncQueueBackend):
    def __init__(self, path: str = "pipeline_queue.db"):
        self._sync = _SQLiteQueueSync(path)

    async def enqueue(self, queue_name: str, body: dict, delay_seconds: int = 0) -> str:
        return await asyncio.to_thread(self._sync.enqueue, queue_name, body, delay_seconds)

    async def lease_next(self, queue_name: str, lease_seconds: int = 30) -> Optional[QueueMessage]:
        await asyncio.to_thread(self._sync.reap_expired)
        return await asyncio.to_thread(self._sync.lease_next, queue_name, lease_seconds)

    async def ack(self, queue_name: str, message_id: str) -> None:
        await asyncio.to_thread(self._sync.ack, queue_name, message_id)

    async def nack(self, queue_name: str, message_id: str, requeue_delay_seconds: int = 0) -> None:
        await asyncio.to_thread(self._sync.nack, queue_name, message_id, requeue_delay_seconds)

    async def has_pending(self, *queue_names: str) -> bool:
        # treat expired leases as pending by reaping first
        await asyncio.to_thread(self._sync.reap_expired)
        count = await asyncio.to_thread(self._sync.count_pending, list(queue_names))
        return count > 0
