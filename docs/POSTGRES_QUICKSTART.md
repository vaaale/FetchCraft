# PostgreSQL Backend - Quick Start

Get started with the PostgreSQL backend in 5 minutes.

## Installation

```bash
# Install PostgreSQL backend support
uv pip install fetchcraft-core[postgres]

# Or with pip
pip install fetchcraft-core[postgres]
```

This installs `asyncpg`, the async PostgreSQL driver.

## Setup PostgreSQL

### Option 1: Docker (Recommended for Development)

```bash
docker run --name fetchcraft-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=ingestion \
  -p 5432:5432 \
  -d postgres:16

# Verify it's running
docker ps
```

### Option 2: Existing PostgreSQL

Just ensure you have:
- PostgreSQL 12+ running
- A database created for ingestion
- Connection credentials

## Basic Usage

```python
import asyncio
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
from fetchcraft.ingestion.base import IngestionPipeline

async def main():
    # Create backend
    backend = AsyncPostgresQueue(
        connection_string="postgresql://postgres:yourpassword@localhost:5432/ingestion"
    )
    
    # Use with pipeline
    pipeline = IngestionPipeline(backend=backend)
    # ... configure your pipeline ...
    
    await pipeline.run_job()
    
    # Clean up
    await backend.close()

asyncio.run(main())
```

That's it! The backend automatically creates tables and indexes on first use.

## Environment Variable Configuration

```bash
# Set in .env file
POSTGRES_URL=postgresql://user:password@localhost:5432/ingestion

# Or export in shell
export POSTGRES_URL="postgresql://user:password@localhost:5432/ingestion"
```

```python
import os
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue

backend = AsyncPostgresQueue(
    connection_string=os.getenv("POSTGRES_URL")
)
```

## Connection String Formats

```python
# Basic
"postgresql://user:password@localhost:5432/dbname"

# Custom port
"postgresql://user:password@localhost:5433/dbname"

# With SSL
"postgresql://user:password@host:5432/db?sslmode=require"

# Cloud provider (example: Heroku)
"postgresql://user:pass@host.compute.amazonaws.com:5432/db"

# Multiple hosts (HA)
"postgresql://user:pass@host1:5432,host2:5432/db"
```

## Running the Example

```bash
# Basic example
cd packages/fetchcraft-core
python examples/postgres_queue_example.py

# Queue operations example
python examples/postgres_queue_example.py operations

# Full RAG stack example (requires MongoDB, Qdrant, OpenAI)
python examples/postgres_queue_example.py full
```

## Quick Comparison: SQLite vs PostgreSQL

| Use SQLite when... | Use PostgreSQL when... |
|-------------------|----------------------|
| Single machine | Multiple workers/machines |
| Development/testing | Production deployment |
| Low volume (<1M messages/day) | High volume |
| Simple deployment | Need HA/replication |
| No external dependencies | OK with database server |

## Next Steps

1. **Read the full documentation**: [POSTGRES_BACKEND.md](POSTGRES_BACKEND.md)
2. **Optimize connection pooling** for your workload
3. **Set up monitoring** for queue depth and performance
4. **Configure backups** for production
5. **Review indexing strategy** for large-scale deployments

## Common Issues

### "Could not connect to server"
- Ensure PostgreSQL is running: `docker ps` or `pg_isready`
- Check connection string is correct
- Verify firewall/network access

### "asyncpg not installed"
- Run: `uv pip install asyncpg`
- Or: `pip install fetchcraft-core[postgres]`

### "Database does not exist"
- Create it: `createdb ingestion`
- Or in Docker: `-e POSTGRES_DB=ingestion`

### "Too many connections"
- Reduce pool sizes in AsyncPostgresQueue
- Increase `max_connections` in postgresql.conf
- Use PgBouncer for connection pooling

## Production Checklist

- [ ] Use SSL connections (`sslmode=require`)
- [ ] Set up connection pooling
- [ ] Configure regular backups
- [ ] Monitor queue depth
- [ ] Set up cleanup jobs for old messages
- [ ] Review and optimize indexes
- [ ] Enable query logging for slow queries
- [ ] Configure replication for HA
- [ ] Set up alerting for errors

## Support

For issues or questions:
1. Check [POSTGRES_BACKEND.md](POSTGRES_BACKEND.md) for detailed docs
2. Review [ERROR_QUEUE.md](ERROR_QUEUE.md) for error handling
3. Look at example code in `examples/postgres_queue_example.py`
4. Check PostgreSQL logs for database-specific issues
