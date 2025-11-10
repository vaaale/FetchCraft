"""
Fetchcraft Ingestion Administration MCP Server

This MCP server provides tools for administering ingestion pipelines that use
SQLite queue backend.

Features:
- List queued messages/records
- View message details
- Clear queue
- Retry failed messages
- Get queue statistics
- Delete specific messages

Usage:
    python -m fetchcraft.ingestion.admin.server
"""

import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = os.getenv("DB_PATH", "demo_queue.db")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

# ============================================================================
# Global State
# ============================================================================


class AppState:
    """Global application state."""

    db_path: Path = Path(DB_PATH)
    initialized: bool = False


app_state = AppState()


# ============================================================================
# Helper Functions
# ============================================================================


def get_db_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    if not app_state.db_path.exists():
        raise FileNotFoundError(f"Database not found: {app_state.db_path}")
    conn = sqlite3.connect(str(app_state.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def format_timestamp(timestamp: Optional[int]) -> Optional[str]:
    """Format Unix timestamp to human-readable string."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def parse_message_body(body: str) -> Dict[str, Any]:
    """Parse message body JSON."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


# ============================================================================
# MCP Server Setup
# ============================================================================


@asynccontextmanager
async def app_lifespan(mcp_app: FastMCP):
    """Initialize the server on startup."""
    try:
        print("\n" + "=" * 70)
        print("🚀 Fetchcraft Ingestion Admin MCP Server - Initializing")
        print("=" * 70)
        print(f"\n📂 Database path: {app_state.db_path.absolute()}")

        # Check if database exists
        if not app_state.db_path.exists():
            print(f"⚠️  Warning: Database not found at {app_state.db_path}")
            print("   Server will start but tools will fail until database is available")
        else:
            # Test connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            count = cursor.fetchone()["count"]
            conn.close()
            print(f"✓ Connected to database ({count} messages in queue)")

        app_state.initialized = True
        print("\n✅ MCP Server ready!")
        print("=" * 70 + "\n")
        yield
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        print("\n👋 Shutting down MCP server...")


# Create MCP server
mcp = FastMCP(
    name="Fetchcraft Ingestion Admin Server",
    instructions="This server provides tools for administering ingestion pipeline queues.",
    lifespan=app_lifespan,
    host=HOST,
    port=PORT,
)


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def list_messages(
    queue: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    List messages in the ingestion queue.

    This tool retrieves messages from the SQLite queue database with optional
    filtering by queue name and state.

    Args:
        queue: Filter by queue name (e.g., 'ingest.main', 'ingest.deferred')
        state: Filter by message state (e.g., 'ready', 'processing', 'done', 'failed')
        limit: Maximum number of messages to return (1-1000, default: 50)
        offset: Offset for pagination (default: 0)

    Returns:
        Dictionary containing messages, total count, and pagination info
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    # Validate limit
    limit = max(1, min(limit, 1000))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Build query with filters
        where_clauses = []
        params = []

        if queue:
            where_clauses.append("queue = ?")
            params.append(queue)

        if state:
            where_clauses.append("state = ?")
            params.append(state)

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM messages{where_sql}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()["count"]

        # Get paginated results
        query = f"""
            SELECT id, queue, body, available_at, lease_until, attempts, state
            FROM messages
            {where_sql}
            ORDER BY available_at DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        messages = []
        for row in rows:
            body_data = parse_message_body(row["body"])

            messages.append(
                {
                    "id": row["id"],
                    "queue": row["queue"],
                    "state": row["state"],
                    "attempts": row["attempts"],
                    "available_at": format_timestamp(row["available_at"]),
                    "lease_until": format_timestamp(row["lease_until"]),
                    "body_preview": (
                        str(body_data)[:100] + "..."
                        if len(str(body_data)) > 100
                        else str(body_data)
                    ),
                }
            )

        conn.close()

        return {
            "messages": messages,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error listing messages: {str(e)}")


@mcp.tool()
async def get_message(message_id: str) -> dict:
    """
    Get detailed information about a specific message.

    This tool retrieves the full details of a message including its complete body.

    Args:
        message_id: The ID of the message to retrieve

    Returns:
        Dictionary containing the complete message details
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, queue, body, available_at, lease_until, attempts, state
            FROM messages
            WHERE id = ?
        """,
            (message_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise RuntimeError(f"Message not found: {message_id}")

        body_data = parse_message_body(row["body"])

        return {
            "id": row["id"],
            "queue": row["queue"],
            "state": row["state"],
            "attempts": row["attempts"],
            "available_at": format_timestamp(row["available_at"]),
            "lease_until": format_timestamp(row["lease_until"]),
            "body": body_data,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error getting message: {str(e)}")


@mcp.tool()
async def get_queue_stats() -> dict:
    """
    Get statistics about the ingestion queues.

    This tool provides an overview of queue status including message counts
    by state and queue.

    Returns:
        Dictionary containing queue statistics
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total messages
        cursor.execute("SELECT COUNT(*) as count FROM messages")
        total = cursor.fetchone()["count"]

        # Messages by state
        cursor.execute(
            """
            SELECT state, COUNT(*) as count
            FROM messages
            GROUP BY state
        """
        )
        by_state = {row["state"]: row["count"] for row in cursor.fetchall()}

        # Messages by queue
        cursor.execute(
            """
            SELECT queue, COUNT(*) as count
            FROM messages
            GROUP BY queue
        """
        )
        by_queue = {row["queue"]: row["count"] for row in cursor.fetchall()}

        # Failed messages (high retry count)
        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM messages
            WHERE attempts >= 3
        """
        )
        failed = cursor.fetchone()["count"]

        # Oldest pending message
        cursor.execute(
            """
            SELECT MIN(available_at) as oldest
            FROM messages
            WHERE state = 'ready'
        """
        )
        oldest = cursor.fetchone()["oldest"]

        conn.close()

        return {
            "total_messages": total,
            "by_state": by_state,
            "by_queue": by_queue,
            "failed_messages": failed,
            "oldest_pending": format_timestamp(oldest) if oldest else None,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error getting queue stats: {str(e)}")


@mcp.tool()
async def delete_message(message_id: str) -> dict:
    """
    Delete a specific message from the queue.

    This tool removes a message permanently from the database.

    Args:
        message_id: The ID of the message to delete

    Returns:
        Dictionary confirming deletion
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if message exists
        cursor.execute("SELECT id FROM messages WHERE id = ?", (message_id,))
        if not cursor.fetchone():
            conn.close()
            raise RuntimeError(f"Message not found: {message_id}")

        # Delete message
        cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": f"Message {message_id} deleted successfully",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error deleting message: {str(e)}")


@mcp.tool()
async def clear_queue(
    queue: Optional[str] = None, state: Optional[str] = None, confirm: bool = False
) -> dict:
    """
    Clear messages from the queue.

    This tool deletes multiple messages based on filters. Requires confirmation.

    Args:
        queue: Filter by queue name (if None, affects all queues)
        state: Filter by message state (if None, affects all states)
        confirm: Must be set to true to execute the deletion

    Returns:
        Dictionary confirming the number of messages deleted
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    if not confirm:
        raise RuntimeError(
            "Clear operation requires confirmation. Set confirm=true to proceed."
        )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Build delete query with filters
        where_clauses = []
        params = []

        if queue:
            where_clauses.append("queue = ?")
            params.append(queue)

        if state:
            where_clauses.append("state = ?")
            params.append(state)

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Get count before deletion
        count_query = f"SELECT COUNT(*) as count FROM messages{where_sql}"
        cursor.execute(count_query, params)
        count = cursor.fetchone()["count"]

        # Delete messages
        delete_query = f"DELETE FROM messages{where_sql}"
        cursor.execute(delete_query, params)
        conn.commit()
        conn.close()

        return {
            "success": True,
            "deleted_count": count,
            "message": f"Deleted {count} messages",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error clearing queue: {str(e)}")


@mcp.tool()
async def retry_message(message_id: str) -> dict:
    """
    Retry a failed or stuck message.

    This tool resets a message's state to 'ready' and clears its lease,
    allowing it to be processed again.

    Args:
        message_id: The ID of the message to retry

    Returns:
        Dictionary confirming the retry operation
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if message exists
        cursor.execute("SELECT id, state FROM messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise RuntimeError(f"Message not found: {message_id}")

        old_state = row["state"]

        # Reset message for retry
        cursor.execute(
            """
            UPDATE messages
            SET state = 'ready',
                lease_until = NULL,
                available_at = ?
            WHERE id = ?
        """,
            (int(time.time()), message_id),
        )
        conn.commit()
        conn.close()

        return {
            "success": True,
            "message_id": message_id,
            "old_state": old_state,
            "new_state": "ready",
            "message": f"Message {message_id} reset for retry",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error retrying message: {str(e)}")


@mcp.tool()
async def list_queues() -> dict:
    """
    List all available queues.

    This tool shows all distinct queue names in the database.

    Returns:
        Dictionary containing list of queue names with their message counts
    """
    if not app_state.initialized:
        raise RuntimeError("Server not initialized. Please try again in a moment.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT queue, COUNT(*) as count
            FROM messages
            GROUP BY queue
            ORDER BY queue
        """
        )

        queues = [
            {"name": row["queue"], "message_count": row["count"]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {"queues": queues, "total_queues": len(queues)}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Error listing queues: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    print("\n" + "=" * 70)
    print("🚀 Starting Fetchcraft Ingestion Admin MCP Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Database: {DB_PATH}")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print("=" * 70 + "\n")

    # Run the MCP server using streamable-http transport
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
