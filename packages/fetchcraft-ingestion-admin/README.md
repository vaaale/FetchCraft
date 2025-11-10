# Fetchcraft Ingestion Admin MCP Server

An MCP (Model Context Protocol) server for administering Fetchcraft ingestion pipelines that use SQLite queue backend.

## Features

This MCP server exposes tools for managing ingestion pipeline queues:

### 1. `list_messages` - List Queue Messages
List messages in the ingestion queue with optional filtering.

**Parameters:**
- `queue` (str, optional): Filter by queue name (e.g., 'ingest.main', 'ingest.deferred')
- `state` (str, optional): Filter by message state ('ready', 'processing', 'done', 'failed')
- `limit` (int, optional): Maximum number of messages to return (1-1000, default: 50)
- `offset` (int, optional): Offset for pagination (default: 0)

**Returns:**
- `messages`: List of messages with metadata
- `total`: Total number of matching messages
- `limit`: Limit used for this request
- `offset`: Offset used for this request
- `has_more`: Whether there are more results

### 2. `get_message` - Get Message Details
Get detailed information about a specific message including its full body.

**Parameters:**
- `message_id` (str): The ID of the message to retrieve

**Returns:**
- `id`: Message ID
- `queue`: Queue name
- `state`: Message state
- `attempts`: Number of processing attempts
- `available_at`: Timestamp when message became available
- `lease_until`: Timestamp until which message is leased
- `body`: Complete message body (parsed JSON)

### 3. `get_queue_stats` - Queue Statistics
Get comprehensive statistics about the ingestion queues.

**Returns:**
- `total_messages`: Total number of messages across all queues
- `by_state`: Message count grouped by state
- `by_queue`: Message count grouped by queue name
- `failed_messages`: Count of messages with 3+ retry attempts
- `oldest_pending`: Timestamp of oldest ready message

### 4. `delete_message` - Delete Message
Delete a specific message from the queue permanently.

**Parameters:**
- `message_id` (str): The ID of the message to delete

**Returns:**
- `success`: Boolean indicating success
- `message`: Confirmation message

### 5. `clear_queue` - Bulk Delete Messages
Clear multiple messages from the queue based on filters. **Requires confirmation**.

**Parameters:**
- `queue` (str, optional): Filter by queue name (if None, affects all queues)
- `state` (str, optional): Filter by message state (if None, affects all states)
- `confirm` (bool): Must be set to true to execute the deletion

**Returns:**
- `success`: Boolean indicating success
- `deleted_count`: Number of messages deleted
- `message`: Confirmation message

### 6. `retry_message` - Retry Failed Message
Reset a failed or stuck message to retry processing.

**Parameters:**
- `message_id` (str): The ID of the message to retry

**Returns:**
- `success`: Boolean indicating success
- `message_id`: ID of the retried message
- `old_state`: Previous state
- `new_state`: New state (always 'ready')
- `message`: Confirmation message

### 7. `list_queues` - List All Queues
List all available queue names with message counts.

**Returns:**
- `queues`: List of queues with names and message counts
- `total_queues`: Total number of distinct queues

## Installation

```bash
cd packages/fetchcraft-ingestion-admin
pip install -e .
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

### Required Configuration

- **Database Path**: Path to the SQLite database
  - `DB_PATH`: Path to queue database (default: demo_queue.db)

### Optional Configuration

- **Server**: HTTP server configuration
  - `HOST`: Server host (default: 0.0.0.0)
  - `PORT`: Server port (default: 8004)

## Usage

### Running the Server

```bash
fetchcraft-ingestion-admin
```

Or directly with Python:

```bash
python -m fetchcraft.ingestion.admin.server
```

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "fetchcraft-admin": {
      "command": "fetchcraft-ingestion-admin",
      "env": {
        "DB_PATH": "/path/to/your/demo_queue.db"
      }
    }
  }
}
```

### Using with Other MCP Clients

The server uses the standard MCP protocol over stdio/HTTP, so it can be integrated with any MCP-compatible client.

## Requirements

- Python 3.12+
- SQLite database (created by Fetchcraft ingestion pipelines)

## Example Workflow

### 1. Check Queue Status

```python
# Get queue statistics
stats = await get_queue_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"By state: {stats['by_state']}")
```

### 2. List Pending Messages

```python
# List messages in 'ready' state
messages = await list_messages(state='ready', limit=10)
for msg in messages['messages']:
    print(f"{msg['id']}: {msg['queue']} - {msg['state']}")
```

### 3. Inspect Message Details

```python
# Get full message details
details = await get_message(message_id='some-message-id')
print(f"Body: {details['body']}")
```

### 4. Retry Failed Message

```python
# Retry a stuck message
result = await retry_message(message_id='failed-message-id')
print(f"Retried: {result['message']}")
```

### 5. Clear Completed Messages

```python
# Clear all 'done' messages
result = await clear_queue(state='done', confirm=True)
print(f"Deleted {result['deleted_count']} messages")
```

## Database Schema

The server works with SQLite databases containing a `messages` table:

```sql
CREATE TABLE messages (
    id           TEXT PRIMARY KEY,
    queue        TEXT    NOT NULL,
    body         TEXT    NOT NULL,
    available_at INTEGER NOT NULL,
    lease_until  INTEGER,
    attempts     INTEGER NOT NULL DEFAULT 0,
    state        TEXT    NOT NULL DEFAULT 'ready'
)
```

## Common States

- **ready**: Message is ready to be processed
- **processing**: Message is currently being processed
- **done**: Message processing completed successfully
- **failed**: Message processing failed

## Architecture

The server is built on:
- **MCP (Model Context Protocol)**: Standard protocol for AI tool integration
- **SQLite**: Queue database backend
- **FastMCP**: Python MCP server framework

## License

MIT
