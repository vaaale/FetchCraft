"""
Enhanced ingestion pipeline with job and document tracking.

This module provides a production-ready ingestion system with:
- Job-level tracking and management
- Document-level progress monitoring through pipeline steps
- Support for remote/async transformations with callbacks
- Comprehensive error handling and retry mechanisms
- Repository pattern for flexible persistence
- Interface-based design for extensibility

Quick Start:
-----------

```python
from fetchcraft.ingestion import (
    TrackedIngestionPipeline,
    IngestionJob,
    ConnectorSource,
    ChunkingTransformation,
    VectorIndexSink,
    DocumentStoreSink,
    PostgresJobRepository,
    PostgresDocumentRepository,
    AsyncPostgresQueue,
)

# Initialize repositories and queue
pool = await asyncpg.create_pool(postgres_url)
job_repo = PostgresJobRepository(pool)
doc_repo = PostgresDocumentRepository(pool)
queue_backend = AsyncPostgresQueue(postgres_url)

# Create job
job = IngestionJob(name="My Job", source_path="documents/")

# Build pipeline
pipeline = TrackedIngestionPipeline(job, queue_backend, job_repo, doc_repo)
pipeline.source(ConnectorSource(...))
pipeline.add_transformation(ChunkingTransformation(...))
pipeline.add_sink(VectorIndexSink(...))

# Run job
await pipeline.run_job()
```

Components:
----------
- **Pipeline**: TrackedIngestionPipeline - Main pipeline with tracking
- **Models**: IngestionJob, DocumentRecord - Data models
- **Interfaces**: ISource, ITransformation, ISink, etc. - Component interfaces
- **Sources**: ConnectorSource - File reading
- **Transformations**: ParsingTransformation, ChunkingTransformation, DocumentSummarization, etc.
- **Sinks**: VectorIndexSink, DocumentStoreSink - Output destinations
- **Repositories**: Job and Document repositories for persistence
- **Queue Backends**: AsyncPostgresQueue, AsyncSQLiteQueue
"""

# Core pipeline
from fetchcraft.ingestion.pipeline import TrackedIngestionPipeline, PipelineStep, Worker

# Data models
from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    TaskRecord,
    JobStatus,
    DocumentStatus,
    TaskStatus,
    utcnow,
)

# Interfaces (new naming without I-prefix)
from fetchcraft.ingestion.interfaces import (
    Source,
    Transformation,
    AsyncTransformation,
    Sink,
    Connector,
    QueueBackend,
    # Backwards compatibility aliases
    ISource,
    ITransformation,
    IRemoteTransformation,
    ISink,
    IConnector,
    IQueueBackend,
)

# Concrete implementations
from fetchcraft.ingestion.sources import ConnectorSource
from fetchcraft.ingestion.transformations import (
    ParsingTransformation,
    AsyncParsingTransformation,
    ChunkingTransformation,
    DocumentSummarization,
    ExtractKeywords,
)
from fetchcraft.ingestion.sinks import (
    VectorIndexSink,
    DocumentStoreSink,
    LoggingSink,
)

# Repositories (new naming without I-prefix)
from fetchcraft.ingestion.repository import (
    JobRepository,
    DocumentRepository,
    TaskRepository,
    PostgresJobRepository,
    PostgresDocumentRepository,
    PostgresTaskRepository,
    # Backwards compatibility aliases
    IJobRepository,
    IDocumentRepository,
    ITaskRepository,
)

# Queue backends
from fetchcraft.ingestion.postgres_backend import AsyncPostgresQueue
from fetchcraft.ingestion.sqlite_backend import AsyncSQLiteQueue

# Legacy support (deprecated - use new pipeline)
from fetchcraft.ingestion.base import (
    IngestionPipeline as LegacyIngestionPipeline,
    Record,
    Source,
    Transformation,
    Sink,
    ConnectorSource as LegacyConnectorSource,
)

__all__ = [
    # Pipeline
    "TrackedIngestionPipeline",
    "PipelineStep",
    "Worker",
    # Models
    "IngestionJob",
    "DocumentRecord",
    "TaskRecord",
    "JobStatus",
    "DocumentStatus",
    "TaskStatus",
    "utcnow",
    # Interfaces (new naming)
    "Source",
    "Transformation",
    "AsyncTransformation",
    "Sink",
    "Connector",
    "QueueBackend",
    # Interfaces (backwards compatibility)
    "ISource",
    "ITransformation",
    "IRemoteTransformation",
    "ISink",
    "IConnector",
    "IQueueBackend",
    # Sources
    "ConnectorSource",
    # Transformations
    "ParsingTransformation",
    "AsyncParsingTransformation",
    "ChunkingTransformation",
    "DocumentSummarization",
    "ExtractKeywords",
    # Sinks
    "VectorIndexSink",
    "DocumentStoreSink",
    "LoggingSink",
    # Repositories (new naming)
    "JobRepository",
    "DocumentRepository",
    "TaskRepository",
    "PostgresJobRepository",
    "PostgresDocumentRepository",
    "PostgresTaskRepository",
    # Repositories (backwards compatibility)
    "IJobRepository",
    "IDocumentRepository",
    "ITaskRepository",
    # Queue Backends
    "AsyncPostgresQueue",
    "AsyncSQLiteQueue",
    # Legacy (deprecated)
    "LegacyIngestionPipeline",
    "Record",
    "LegacyConnectorSource",
]
