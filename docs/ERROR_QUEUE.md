# Error Queue Implementation

## Overview

The ingestion pipeline now includes an error queue (`ingest.error`) that captures all failed messages instead of dropping them. This allows for error analysis, debugging, and potential retry/recovery mechanisms.

## What Changed

### 1. New Error Queue Constant

```python
ERROR_QUEUE = "ingest.error"  # alongside MAIN_QUEUE and DEFER_QUEUE
```

### 2. Worker Error Handling

**Before:**
- After max retries, messages were simply dropped with a print statement

**After:**
- Failed messages are sent to the error queue with full context:
  ```python
  {
    "type": "error",
    "original_queue": "ingest.main",  # or "ingest.deferred"
    "message_id": "...",
    "attempts": 5,
    "error": "Error message",
    "error_type": "ValueError",
    "original_body": {...}  # Full original message
  }
  ```

### 3. Sink Error Handling

**Before:**
- Sink errors were caught and only printed

**After:**
- Sink errors are sent to the error queue with context:
  ```python
  {
    "type": "sink_error",
    "sink": "VectorIndexSink",
    "error": "Error message",
    "error_type": "ConnectionError",
    "record": {...}  # Full record data
  }
  ```

### 4. Worker Constructor Update

Workers now accept an optional `error_queue` parameter:
```python
Worker(
    name="main",
    backend=backend,
    handler=handler,
    config=WorkerConfig(...),
    error_queue=ERROR_QUEUE  # New parameter
)
```

## Error Queue Message Types

### Type: "error"
Worker-level errors (transformation failures, general processing errors)

**Fields:**
- `type`: "error"
- `original_queue`: Queue where the message was being processed
- `message_id`: Original message ID
- `attempts`: Number of retry attempts before failure
- `error`: Error message string
- `error_type`: Exception class name
- `original_body`: Complete original message body

### Type: "sink_error"
Errors that occur when writing to sinks

**Fields:**
- `type`: "sink_error"
- `sink`: Sink class name
- `error`: Error message string
- `error_type`: Exception class name
- `record`: Complete record data that failed to write

## Usage

### Accessing Error Messages

You can query the error queue using the admin tools:

```python
from fetchcraft.ingestion.sqlite_backend import AsyncSQLiteQueue

backend = AsyncSQLiteQueue("ingestion_queue.db")

# Get error messages
errors = await backend.list_messages(queue_name="ingest.error")
```

### Via Admin API

Using the fetchcraft-admin web interface:

```bash
curl "http://localhost:8080/api/messages?queue=ingest.error"
```

### Monitoring Errors

```python
# Get count of error messages
conn = sqlite3.connect("ingestion_queue.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM messages WHERE queue = 'ingest.error'")
error_count = cursor.fetchone()[0]
```

## Error Analysis

Error messages contain full context for debugging:

1. **Worker Errors**: See which transformation step failed, how many retries occurred, and the exact error
2. **Sink Errors**: See which sink failed, what record was being written, and the error details

## Recovery Options

### Option 1: Manual Inspection
1. Query error queue via admin interface
2. Analyze error patterns
3. Fix underlying issues
4. Clear error queue

### Option 2: Retry Failed Messages
```python
async def retry_errors():
    backend = AsyncSQLiteQueue("ingestion_queue.db")
    
    # Get error messages
    conn = sqlite3.connect("ingestion_queue.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, body FROM messages 
        WHERE queue = 'ingest.error' AND state = 'ready'
    """)
    
    for msg_id, body_json in cursor.fetchall():
        error_msg = json.loads(body_json)
        
        if error_msg["type"] == "error":
            # Re-enqueue to original queue
            original_body = error_msg["original_body"]
            # Reset attempts
            original_body["__attempts__"] = 0
            await backend.enqueue(
                error_msg["original_queue"],
                body=original_body
            )
            # Remove from error queue
            await backend.ack("ingest.error", msg_id)
```

### Option 3: Clear Error Queue
```python
# After fixing issues, clear the error queue
cursor.execute("DELETE FROM messages WHERE queue = 'ingest.error'")
conn.commit()
```

## Benefits

1. **No Data Loss**: Failed messages are preserved for analysis
2. **Better Debugging**: Full error context available
3. **Recovery Options**: Failed messages can be retried after fixes
4. **Monitoring**: Track error rates and patterns
5. **Auditability**: Complete record of what went wrong

## Considerations

1. **Storage**: Error queue will grow over time - implement cleanup strategy
2. **Monitoring**: Set up alerts for error queue size
3. **Investigation**: Regularly review error queue to identify systemic issues
4. **Cleanup**: Periodically archive or delete old error messages

## Example: Error Alert

```python
async def check_error_queue():
    conn = sqlite3.connect("ingestion_queue.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as count FROM messages 
        WHERE queue = 'ingest.error' 
        AND available_at > ?
    """, (time.time() - 3600,))  # Last hour
    
    recent_errors = cursor.fetchone()[0]
    
    if recent_errors > 10:
        # Send alert
        print(f"WARNING: {recent_errors} errors in the last hour!")
```

## Migration Notes

- **Backward Compatible**: If `error_queue` is not provided to Worker, it falls back to the old behavior (dropping messages)
- **Existing Pipelines**: Will automatically use error queue if they use the standard pipeline setup
- **Custom Workers**: Need to pass `error_queue=ERROR_QUEUE` parameter

## Future Enhancements

Potential improvements:
1. Automatic retry with exponential backoff
2. Dead letter queue for permanently failed messages
3. Error categorization and routing
4. Automated error notifications (email, Slack, etc.)
5. Error rate limiting
6. Batch error processing
