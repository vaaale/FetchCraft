# Async Job-Based Document Parsing

The Docling parsing server now supports asynchronous job-based parsing, allowing clients to submit documents for parsing and receive results without keeping a connection open during the entire parsing process.

## Overview

### Previous Behavior (Blocking)
- Client submits document via `/parse` endpoint
- Client keeps connection open
- Server parses document synchronously
- Server returns complete results
- Connection closes

**Problem**: For large documents or multiple files, this could timeout or tie up connections.

### New Behavior (Async Jobs)
- Client submits document via `/submit` endpoint
- Server returns job ID immediately
- Client can disconnect
- Server processes job in background
- Client polls `/jobs/{job_id}` for status
- Client fetches results via `/jobs/{job_id}/results` when complete

**Benefits**: 
- No timeout issues
- Client can handle other tasks while waiting
- Better resource utilization
- Can submit multiple jobs concurrently

## API Endpoints

### Submit Job
```http
POST /submit
Content-Type: multipart/form-data

files: <file1>
files: <file2>
...
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job submitted successfully with 2 file(s)"
}
```

### Check Job Status
```http
GET /jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "submitted_at": 1699999999.123,
  "started_at": 1700000000.456,
  "completed_at": null,
  "error": null
}
```

Status values: `pending`, `processing`, `completed`, `failed`

### Get Job Results
```http
GET /jobs/{job_id}/results
```

**Response** (when completed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": {
    "results": [...],
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "total_nodes": 150,
    "total_processing_time_ms": 1234.56
  },
  "error": null
}
```

## Client Usage

### Python Async Client

```python
from fetchcraft.parsing.docling import AsyncDoclingParserClient
import asyncio

async def parse_documents():
    client = AsyncDoclingParserClient()
    
    # Method 1: Submit and poll manually
    submit_response = await client.submit_job("document.pdf")
    job_id = submit_response['job_id']
    
    # Poll for completion
    while True:
        status = await client.get_job_status(job_id)
        if status['status'] == 'completed':
            break
        await asyncio.sleep(1)
    
    # Get results
    results = await client.get_job_results(job_id)
    print(f"Parsed {results['results']['total_nodes']} nodes")
    
    # Method 2: Submit and wait (convenience)
    results = await client.submit_and_wait(
        "doc1.pdf", 
        "doc2.docx",
        poll_interval=1.0,
        timeout=300
    )
    print(f"Parsed {results['total_nodes']} nodes")

asyncio.run(parse_documents())
```

### Python Sync Client

```python
from fetchcraft.parsing.docling import DoclingParserClient
import time

client = DoclingParserClient()

# Method 1: Submit and poll manually
submit_response = client.submit_job("document.pdf")
job_id = submit_response['job_id']

# Poll for completion
while True:
    status = client.get_job_status(job_id)
    if status['status'] == 'completed':
        break
    time.sleep(1)

# Get results
results = client.get_job_results(job_id)
print(f"Parsed {results['results']['total_nodes']} nodes")

# Method 2: Submit and wait (convenience)
results = client.submit_and_wait(
    "doc1.pdf", 
    "doc2.docx",
    poll_interval=1.0,
    timeout=300
)
print(f"Parsed {results['total_nodes']} nodes")
```

### cURL Examples

```bash
# Submit job
curl -X POST http://localhost:8080/submit \
  -F "files=@document.pdf" \
  -F "files=@report.docx"

# Response: {"job_id": "abc-123", "status": "pending", ...}

# Check status
curl http://localhost:8080/jobs/abc-123

# Get results (when completed)
curl http://localhost:8080/jobs/abc-123/results
```

## Architecture

### Server Components

1. **Job Storage**: In-memory dictionary storing job state
   - Job ID
   - File contents
   - Status (pending/processing/completed/failed)
   - Timestamps
   - Results

2. **Job Queue**: AsyncIO queue for job processing
   - FIFO ordering
   - Decouples submission from processing

3. **Background Processor**: AsyncIO task continuously processing jobs
   - Pulls jobs from queue
   - Updates job status
   - Stores results
   - Handles errors

4. **API Endpoints**:
   - `/submit` - Accept files, create job, return immediately
   - `/jobs/{job_id}` - Return job status
   - `/jobs/{job_id}/results` - Return results if available

### Data Models

```python
class JobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    submitted_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    error: Optional[str]

class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    results: Optional[BatchParseResponse]
    error: Optional[str]
```

## Backward Compatibility

The original `/parse` endpoint remains available and unchanged. Existing code using the blocking interface will continue to work without modification.

```python
# Old blocking interface still works
client = AsyncDoclingParserClient()
results = await client.parse("document.pdf")
```

## Production Considerations

### Current Implementation
- **Storage**: In-memory (jobs lost on restart)
- **Cleanup**: No automatic job cleanup
- **Concurrency**: Controlled via semaphores

### Recommendations for Production

1. **Persistent Storage**: Use Redis or database for job storage
   ```python
   # Example with Redis
   import redis
   redis_client = redis.Redis()
   redis_client.set(f"job:{job_id}", json.dumps(job_data))
   ```

2. **Job Cleanup**: Implement TTL or periodic cleanup
   ```python
   # Delete completed jobs older than 1 hour
   if job.completed_at and (time.time() - job.completed_at) > 3600:
       del app_state.jobs[job_id]
   ```

3. **Result Caching**: Store large results in object storage
   ```python
   # Upload results to S3, return reference
   s3_key = f"results/{job_id}.json"
   s3_client.put_object(Bucket=bucket, Key=s3_key, Body=json.dumps(results))
   ```

4. **Monitoring**: Add metrics and logging
   ```python
   # Track job metrics
   metrics.incr('jobs.submitted')
   metrics.timing('job.duration', processing_time)
   ```

5. **Multiple Workers**: Scale horizontally with message queue
   ```python
   # Use RabbitMQ/SQS instead of asyncio.Queue
   channel.basic_publish(exchange='jobs', routing_key='parse', body=job_id)
   ```

## Testing

Run the server:
```bash
cd packages/fetchcraft-parsing-docling
python -m fetchcraft.parsing.docling.server
```

Test with example:
```bash
python examples/async_job_example.py
```

Test with curl:
```bash
# Submit job
JOB_ID=$(curl -s -X POST http://localhost:8080/submit \
  -F "files=@test.pdf" | jq -r .job_id)

# Check status
curl http://localhost:8080/jobs/$JOB_ID | jq

# Get results
curl http://localhost:8080/jobs/$JOB_ID/results | jq
```

## Migration Guide

### For Existing Users

If you're currently using the blocking `/parse` endpoint:

**Option 1**: Continue using it (no changes needed)

**Option 2**: Switch to async jobs for better performance

```python
# Before
results = await client.parse("large_document.pdf")

# After (simple)
results = await client.submit_and_wait("large_document.pdf")

# After (advanced)
response = await client.submit_job("large_document.pdf")
# Do other work...
status = await client.get_job_status(response['job_id'])
results = await client.get_job_results(response['job_id'])
```

### Benefits of Migration
- Better handling of large documents
- Can process multiple documents concurrently
- No connection timeouts
- Client can do other work while waiting

## Summary

The async job interface provides a more robust and scalable way to parse documents, especially for:
- Large files that take time to process
- Batch processing of many files
- Applications that need to remain responsive
- Distributed systems with unreliable connections

The blocking `/parse` endpoint remains available for simple use cases and backward compatibility.
