# PostgreSQL Backend for Ingestion Queue

A high-performance, production-ready PostgreSQL backend for the Fetchcraft ingestion pipeline.

## Features

- **Async Native**: Uses `asyncpg` for true async PostgreSQL operations
- **Efficient Leasing**: Leverages PostgreSQL's `FOR UPDATE SKIP LOCKED` for lock-free message leasing
- **Automatic Lease Expiry**: Automatically reaps expired leases
- **JSONB Storage**: Stores message bodies as JSONB for flexibility and queryability
- **Connection Pooling**: Built-in connection pool management
- **Production Ready**: Includes proper indexing, transaction handling, and error management

## Installation

Install the required dependency:

```bash
pip install asyncpg
# or
uv pip install asyncpg
```

## Usage

### Basic Setup

```python
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
from fetchcraft.ingestion.base import IngestionPipeline

# Create PostgreSQL backend
backend = AsyncPostgresQueue(
    connection_string="postgresql://user:password@localhost:5432/ingestion_db",
    pool_min_size=10,
    pool_max_size=20,
)

# Use with pipeline
pipeline = IngestionPipeline(backend=backend)
# ... configure pipeline ...
await pipeline.run_job()

# Clean up when done
await backend.close()
```

### Connection String Formats

```python
# Basic
"postgresql://user:password@localhost:5432/dbname"

# With SSL
"postgresql://user:password@host:5432/dbname?sslmode=require"

# With connection pooling options
"postgresql://user:password@host:5432/dbname?min_size=10&max_size=20"

# Using environment variable
import os
connection_string = os.getenv("DATABASE_URL")
backend = AsyncPostgresQueue(connection_string)
```

### Connection Pool Configuration

```python
backend = AsyncPostgresQueue(
    connection_string="postgresql://...",
    pool_min_size=10,   # Minimum connections to keep open
    pool_max_size=20,   # Maximum connections in pool
)
```

**Guidelines:**
- **Min Size**: Set to number of concurrent workers
- **Max Size**: 2-3x the number of concurrent workers
- For high throughput: Increase max_size
- For many databases: Keep pool sizes lower

## Database Schema

The backend automatically creates the following schema:

```sql
CREATE TABLE messages (
    id           TEXT PRIMARY KEY,
    queue        TEXT    NOT NULL,
    body         JSONB   NOT NULL,
    available_at BIGINT  NOT NULL,
    lease_until  BIGINT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    state        TEXT    NOT NULL DEFAULT 'ready',
    created_at   BIGINT  NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_messages_queue_avail 
    ON messages(queue, available_at) WHERE state = 'ready';
CREATE INDEX idx_messages_state 
    ON messages(state);
CREATE INDEX idx_messages_lease 
    ON messages(lease_until) WHERE state = 'leased';
```

### States

Messages can be in three states:
- `ready`: Available for processing
- `leased`: Currently being processed by a worker
- `done`: Processing completed

## Advanced Usage

### Get Queue Statistics

```python
stats = await backend.get_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"By state: {stats['by_state']}")
print(f"By queue: {stats['by_queue']}")
print(f"Failed: {stats['failed_messages']}")
```

### Clear a Queue

```python
# Clear all messages from a queue
count = await backend.clear_queue("ingest.main")
print(f"Deleted {count} messages")

# Clear only messages in a specific state
count = await backend.clear_queue("ingest.error", state="done")
print(f"Deleted {count} done error messages")
```

### Delete Specific Message

```python
deleted = await backend.delete_message("message-id-123")
if deleted:
    print("Message deleted")
```

### Manual Lease Management

```python
# Lease a message
msg = await backend.lease_next("ingest.main", lease_seconds=60)
if msg:
    try:
        # Process message
        result = process(msg.body)
        
        # Acknowledge success
        await backend.ack("ingest.main", msg.id)
    except Exception as e:
        # Return to queue for retry (with 5 second delay)
        await backend.nack("ingest.main", msg.id, requeue_delay_seconds=5)
```

## Performance Characteristics

### Message Leasing

The PostgreSQL backend uses `FOR UPDATE SKIP LOCKED` which provides:
- **Lock-free operation**: Workers don't block each other
- **High concurrency**: Multiple workers can lease messages simultaneously
- **No deadlocks**: Skip locked rows instead of waiting

### Indexing Strategy

- **Partial indexes**: Only index relevant rows (e.g., ready messages)
- **Compound indexes**: Optimize common query patterns
- **Covering indexes**: Reduce table lookups

### Connection Pooling

- **Reuses connections**: Avoids connection overhead
- **Configurable size**: Tune for your workload
- **Automatic recovery**: Handles connection failures gracefully

## Monitoring and Maintenance

### Monitor Queue Depth

```sql
-- Check queue depths
SELECT queue, state, COUNT(*) 
FROM messages 
GROUP BY queue, state;

-- Check for stuck messages (leased for too long)
SELECT queue, COUNT(*) 
FROM messages 
WHERE state = 'leased' 
  AND lease_until < EXTRACT(EPOCH FROM NOW())::BIGINT
GROUP BY queue;
```

### Cleanup Old Messages

```sql
-- Delete done messages older than 7 days
DELETE FROM messages 
WHERE state = 'done' 
  AND created_at < EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days')::BIGINT;

-- Archive old error messages
INSERT INTO messages_archive 
SELECT * FROM messages 
WHERE queue = 'ingest.error' 
  AND created_at < EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')::BIGINT;

DELETE FROM messages 
WHERE queue = 'ingest.error' 
  AND created_at < EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')::BIGINT;
```

### Performance Monitoring

```sql
-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'messages';

-- Check table statistics
SELECT * FROM pg_stat_user_tables WHERE tablename = 'messages';
```

## Comparison with SQLite Backend

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Concurrency | Limited (file-based locks) | Excellent (row-level locks) |
| Performance | Good for single worker | Scales with workers |
| Deployment | Single file | Separate database server |
| Backup | Copy file | pg_dump / WAL archiving |
| Monitoring | Basic | Rich ecosystem |
| HA/Replication | No | Yes (streaming, logical) |
| JSONB Queries | No | Yes |
| Best For | Dev/Small deployments | Production/Scale |

## Production Deployment

### High Availability

```python
# Use connection string with multiple hosts
backend = AsyncPostgresQueue(
    connection_string="postgresql://user:pass@host1:5432,host2:5432/db?target_session_attrs=read-write"
)
```

### SSL Configuration

```python
backend = AsyncPostgresQueue(
    connection_string="postgresql://user:pass@host:5432/db?sslmode=require&sslcert=/path/to/cert&sslkey=/path/to/key"
)
```

### Environment-Based Configuration

```python
import os

backend = AsyncPostgresQueue(
    connection_string=os.environ["DATABASE_URL"],
    pool_min_size=int(os.environ.get("DB_POOL_MIN_SIZE", "10")),
    pool_max_size=int(os.environ.get("DB_POOL_MAX_SIZE", "20")),
)
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: ingestion
      POSTGRES_USER: ingestion_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready -U ingestion_user" ]
      interval: 10s
      timeout: 5s
      retries: 5

  ingestion:
    build: ../packages/fetchcraft-core
    environment:
      DATABASE_URL: postgresql://ingestion_user:secure_password@postgres:5432/ingestion
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

## Migration from SQLite

### Export SQLite Data

```python
import sqlite3
import json
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue

# Read from SQLite
sqlite_conn = sqlite3.connect("ingestion_queue.db")
sqlite_conn.row_factory = sqlite3.Row
cursor = sqlite_conn.cursor()

# Setup PostgreSQL
pg_backend = AsyncPostgresQueue("postgresql://...")
await pg_backend._ensure_pool()

# Migrate messages
cursor.execute("SELECT * FROM messages")
for row in cursor.fetchall():
    await pg_backend.enqueue(
        queue_name=row["queue"],
        body=json.loads(row["body"]),
        delay_seconds=max(0, row["available_at"] - int(time.time()))
    )

sqlite_conn.close()
```

### Update Configuration

```python
# Old
from fetchcraft.ingestion.sqlite_backend import AsyncSQLiteQueue
backend = AsyncSQLiteQueue("ingestion_queue.db")

# New
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
backend = AsyncPostgresQueue("postgresql://...")
```

## Troubleshooting

### Connection Issues

```python
# Test connection
try:
    backend = AsyncPostgresQueue("postgresql://...")
    pool = await backend._ensure_pool()
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT version()")
        print(f"Connected: {version}")
except Exception as e:
    print(f"Connection failed: {e}")
```

### High Connection Count

If you see "too many connections" errors:
1. Reduce `pool_max_size`
2. Increase PostgreSQL `max_connections`
3. Use PgBouncer for connection pooling

### Slow Queries

```sql
-- Enable slow query logging in postgresql.conf
log_min_duration_statement = 1000  # Log queries > 1s

-- Check for missing indexes
SELECT * FROM pg_stat_user_tables WHERE tablename = 'messages';
```

### Memory Issues

```python
# Use smaller connection pools
backend = AsyncPostgresQueue(
    connection_string="...",
    pool_min_size=5,
    pool_max_size=10,
)
```

## Best Practices

1. **Connection Pooling**: Always use connection pools in production
2. **Cleanup Strategy**: Regularly archive or delete old messages
3. **Monitoring**: Set up alerts for queue depth and error rates
4. **Indexing**: Keep indexes up to date with VACUUM and ANALYZE
5. **Backups**: Regular pg_dump or continuous archiving
6. **SSL**: Always use SSL in production
7. **Resource Limits**: Set appropriate pool sizes for your workload

## Future Enhancements

Potential improvements:
- Partitioning for large message volumes
- Custom serialization formats
- Dead letter queue integration
- Message priority support
- Batch operations
- Streaming replication support
