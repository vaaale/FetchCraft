"""
Enhanced ingestion pipeline with job and document tracking.

This module implements an async, durable ingestion pipeline that tracks
documents through each processing step and provides visibility into
job execution status.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable, Any

from fetchcraft.ingestion.interfaces import (
    ISource,
    ITransformation,
    ISink,
    IQueueBackend,
    IRemoteTransformation,
)
from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    JobStatus,
    DocumentStatus,
    utcnow,
)
from fetchcraft.ingestion.repository import (
    IJobRepository,
    IDocumentRepository,
)

# Queue names
MAIN_QUEUE = "ingest.main"
REMOTE_QUEUE = "ingest.remote"
ERROR_QUEUE = "ingest.error"

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """Message in a queue."""
    id: str
    body: dict


@dataclass
class WorkerConfig:
    """Configuration for a worker."""
    queue_name: str
    lease_seconds: int = 60
    poll_interval: float = 0.2
    max_retries: int = 5
    backoff_seconds: float = 5.0


@dataclass
class PipelineStep:
    """
    Represents a step in the pipeline.
    
    Attributes:
        transformation: The transformation to apply
        is_remote: Whether this is a remote transformation
        name: Step name (for tracking)
    """
    transformation: ITransformation
    is_remote: bool = False
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = self.transformation.get_name()


class Worker:
    """
    Async worker that processes messages from a queue.
    
    Workers lease messages from a queue, process them using a handler,
    and handle retries and error cases.
    """
    
    def __init__(
        self,
        name: str,
        backend: IQueueBackend,
        handler: Callable[[dict], Awaitable[None]],
        config: WorkerConfig,
        error_queue: Optional[str] = None,
    ):
        self.name = name
        self.backend = backend
        self.handler = handler
        self.config = config
        self.error_queue = error_queue
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        logger.info(f"Worker '{name}' initialized for queue '{config.queue_name}'")
    
    async def start(self):
        """Start the worker."""
        logger.info(f"Starting worker '{self.name}'")
        self._task = asyncio.create_task(self._run(), name=f"worker:{self.name}")
    
    async def _run(self):
        """Main worker loop."""
        while not self._stopped.is_set():
            try:
                msg = await self.backend.lease_next(
                    self.config.queue_name,
                    lease_seconds=self.config.lease_seconds
                )
                
                if not msg:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                
                try:
                    await self.handler(msg.body)
                    await self.backend.ack(self.config.queue_name, msg.id)
                    logger.debug(f"Worker '{self.name}' processed message {msg.id}")
                    
                except Exception as e:
                    logger.error(
                        f"Worker '{self.name}' error processing message {msg.id}: {e}",
                        exc_info=True
                    )
                    await self._handle_error(msg, e)
                    
            except Exception as e:
                logger.error(f"Worker '{self.name}' unexpected error: {e}", exc_info=True)
                await asyncio.sleep(self.config.poll_interval)
    
    async def _handle_error(self, msg: QueueMessage, error: Exception):
        """Handle processing errors with retry logic."""
        body = msg.body
        attempts = int(body.get("__attempts__", 0)) + 1
        body["__attempts__"] = attempts
        
        if attempts >= self.config.max_retries:
            # Max retries reached, send to error queue
            if self.error_queue:
                error_body = {
                    "type": "error",
                    "original_queue": self.config.queue_name,
                    "message_id": msg.id,
                    "attempts": attempts,
                    "error": str(error),
                    "error_type": error.__class__.__name__,
                    "original_body": body,
                }
                await self.backend.enqueue(self.error_queue, body=error_body)
                logger.error(
                    f"Worker '{self.name}' moved message {msg.id} to error queue "
                    f"after {attempts} attempts: {error}"
                )
            else:
                logger.error(
                    f"Worker '{self.name}' dropping message {msg.id} "
                    f"after {attempts} attempts: {error}"
                )
            
            await self.backend.ack(self.config.queue_name, msg.id)
        else:
            # Retry with backoff
            delay = int(self.config.backoff_seconds * attempts)
            await self.backend.nack(
                self.config.queue_name,
                msg.id,
                requeue_delay_seconds=delay
            )
            logger.warning(
                f"Worker '{self.name}' requeuing message {msg.id} "
                f"for attempt {attempts + 1} with {delay}s delay"
            )
    
    async def stop(self):
        """Stop the worker."""
        logger.info(f"Stopping worker '{self.name}'")
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class TrackedIngestionPipeline:
    """
    Enhanced ingestion pipeline with job and document tracking.
    
    This pipeline provides:
    - Job-level tracking with status updates
    - Document-level tracking through each pipeline step
    - Support for remote/async transformations with callbacks
    - Comprehensive logging
    - Error handling and retry mechanisms
    """
    
    def __init__(
        self,
        job: IngestionJob,
        backend: IQueueBackend,
        job_repo: IJobRepository,
        doc_repo: IDocumentRepository,
        num_workers: int = 1,
    ):
        """
        Initialize the pipeline.
        
        Args:
            job: The ingestion job configuration
            backend: Queue backend for message passing
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            num_workers: Number of concurrent workers for processing documents (default: 1)
        """
        self.job = job
        self.backend = backend
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.num_workers = max(1, num_workers)  # Ensure at least 1 worker
        
        self._source: Optional[ISource] = None
        self._steps: List[PipelineStep] = []
        self._sinks: List[ISink] = []
        
        self._main_workers: List[Worker] = []
        self._remote_workers: List[Worker] = []
        
        logger.info(
            f"Initialized pipeline for job '{job.name}' (ID: {job.id}) "
            f"with {self.num_workers} worker(s)"
        )

    # ========== Builder API ==========
    
    def source(self, src: ISource) -> "TrackedIngestionPipeline":
        """
        Set the pipeline source.
        
        Args:
            src: The source to use
            
        Returns:
            Self for chaining
        """
        self._source = src
        logger.debug(f"Pipeline source set to {src.__class__.__name__}")
        return self
    
    def add_transformation(
        self,
        transformation: ITransformation,
        is_remote: bool = False,
    ) -> "TrackedIngestionPipeline":
        """
        Add a transformation step to the pipeline.
        
        Args:
            transformation: The transformation to add
            is_remote: Whether this is a remote/async transformation
            
        Returns:
            Self for chaining
        """
        step = PipelineStep(
            transformation=transformation,
            is_remote=is_remote,
        )
        self._steps.append(step)
        self.job.pipeline_steps.append(step.name)
        logger.debug(f"Added transformation: {step.name} (remote={is_remote})")
        return self
    
    def add_sink(self, sink: ISink) -> "TrackedIngestionPipeline":
        """
        Add a sink to the pipeline.
        
        Args:
            sink: The sink to add
            
        Returns:
            Self for chaining
        """
        self._sinks.append(sink)
        sink_name = sink.get_name()
        self.job.pipeline_steps.append(f"sink:{sink_name}")
        logger.debug(f"Added sink: {sink_name}")
        return self
    
    # ========== Pipeline Execution ==========
    
    def _validate(self):
        """Validate pipeline configuration."""
        if not self._source:
            raise ValueError("Pipeline requires a source. Call source() before run_job()")
        if not self._sinks:
            raise ValueError("Pipeline requires at least one sink. Call add_sink() before run_job()")
        logger.debug("Pipeline configuration validated")
    
    async def run_job(self) -> None:
        """
        Execute the ingestion job.
        
        This method:
        1. Validates the pipeline configuration
        2. Persists the job to the database
        3. Starts worker threads
        4. Enqueues documents from the source
        5. Waits for all processing to complete
        6. Updates job status and shuts down workers
        """
        self._validate()
        
        try:
            # Persist job if it doesn't already exist
            existing_job = await self.job_repo.get_job(self.job.id)
            if not existing_job:
                await self.job_repo.create_job(self.job)
                logger.info(f"Created job '{self.job.name}' (ID: {self.job.id})")
            else:
                logger.info(f"Job '{self.job.name}' already exists (ID: {self.job.id})")
            
            # Mark job as started
            await self.job_repo.set_job_started(self.job.id)
            logger.info(f"Job '{self.job.name}' started")
            
            # Start workers
            await self._start_workers()
            
            # Enqueue source documents
            await self._enqueue_source_documents()
            
            # Wait until all processing is complete
            await self._wait_until_idle()
            
            # Mark job as completed
            await self.job_repo.set_job_completed(self.job.id)
            logger.info(f"Job '{self.job.name}' completed successfully")
            
        except Exception as e:
            logger.error(f"Job '{self.job.name}' failed: {e}", exc_info=True)
            await self.job_repo.update_job_status(
                self.job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )
            raise
        finally:
            await self.shutdown()
    
    async def _start_workers(self):
        """Start worker threads for processing."""
        # Start multiple main workers for concurrent processing
        for i in range(self.num_workers):
            worker = Worker(
                name=f"main-{i+1}",
                backend=self.backend,
                handler=self._handle_main_message,
                config=WorkerConfig(queue_name=MAIN_QUEUE),
                error_queue=ERROR_QUEUE,
            )
            self._main_workers.append(worker)
        
        # Start multiple remote workers (same count as main workers)
        for i in range(self.num_workers):
            worker = Worker(
                name=f"remote-{i+1}",
                backend=self.backend,
                handler=self._handle_remote_callback,
                config=WorkerConfig(queue_name=REMOTE_QUEUE),
                error_queue=ERROR_QUEUE,
            )
            self._remote_workers.append(worker)
        
        # Start all workers concurrently
        all_workers = self._main_workers + self._remote_workers
        await asyncio.gather(*[worker.start() for worker in all_workers])
        
        logger.info(
            f"Pipeline workers started: {self.num_workers} main worker(s), "
            f"{self.num_workers} remote worker(s)"
        )
    
    async def _enqueue_source_documents(self):
        """Read documents from source and enqueue them."""
        count = 0
        logger.info(f"Enqueuing documents from source for job '{self.job.name}'")
        
        async for doc_record in self._source.read():  # type: ignore
            # Set job ID and initialize step statuses
            doc_record.job_id = self.job.id
            doc_record.step_statuses = {
                step: "pending" for step in self.job.pipeline_steps
            }
            
            # Persist document record
            await self.doc_repo.create_document(doc_record)
            
            # Enqueue for processing
            await self.backend.enqueue(
                MAIN_QUEUE,
                body={
                    "type": "document",
                    "doc_id": doc_record.id,
                    "job_id": self.job.id,
                }
            )
            count += 1
            
            if count % 10 == 0:
                logger.info(f"Enqueued {count} documents")
        
        logger.info(f"Finished enqueuing {count} documents for job '{self.job.name}'")
    
    async def _wait_until_idle(
        self,
        poll_interval: float = 0.5,
        grace_seconds: float = 2.0,
    ):
        """Wait until all queues are empty and no documents are in PROCESSING status."""
        logger.info("Waiting for pipeline to become idle...")
        iteration = 0
        
        while True:
            try:
                has_work = await self.backend.has_pending(MAIN_QUEUE, REMOTE_QUEUE)
                
                # Also check for documents still in PROCESSING (e.g., parent docs waiting for children)
                processing_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PROCESSING
                )
                
                if not has_work and len(processing_docs) == 0:
                    logger.info(f"Queues empty and no processing documents after {iteration} checks")
                    break
                
                if iteration % 20 == 0:
                    logger.info(
                        f"Still processing... (iteration {iteration}, "
                        f"processing_docs={len(processing_docs)})"
                    )
            
            except Exception as e:
                # Handle pool closing during shutdown gracefully
                if "pool is closing" in str(e):
                    logger.warning("Database pool closing - stopping idle check")
                    return
                raise
            
            iteration += 1
            await asyncio.sleep(poll_interval)
        
        # Grace period to ensure no new work appears
        logger.debug(f"Starting {grace_seconds}s grace period...")
        deadline = asyncio.get_event_loop().time() + grace_seconds
        
        while asyncio.get_event_loop().time() < deadline:
            try:
                has_work = await self.backend.has_pending(MAIN_QUEUE, REMOTE_QUEUE)
                processing_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PROCESSING
                )
                
                if has_work or len(processing_docs) > 0:
                    logger.debug("Work appeared during grace period, restarting wait...")
                    return await self._wait_until_idle(poll_interval, grace_seconds)
            
            except Exception as e:
                # Handle pool closing during shutdown gracefully
                if "pool is closing" in str(e):
                    logger.warning("Database pool closing during grace period - stopping idle check")
                    return
                raise
            
            await asyncio.sleep(poll_interval)
        
        logger.info("Pipeline is idle, all processing complete")
    
    async def shutdown(self):
        """Shutdown the pipeline workers."""
        logger.info("Shutting down pipeline workers...")
        
        # Stop all workers
        all_workers = self._main_workers + self._remote_workers
        await asyncio.gather(
            *[worker.stop() for worker in all_workers],
            return_exceptions=True
        )
        
        logger.info("Pipeline shutdown complete")
    
    # ========== Message Handlers ==========
    
    async def _handle_main_message(self, body: dict):
        """Handle messages in the main processing queue."""
        if body.get("type") != "document":
            logger.warning(f"Unknown message type in main queue: {body.get('type')}")
            return
        
        doc_id = body["doc_id"]
        job_id = body["job_id"]
        current_step_idx = body.get("current_step_idx", 0)
        
        # Load document record
        doc = await self.doc_repo.get_document(doc_id)
        if not doc:
            logger.error(f"Document {doc_id} not found")
            return
        
        # Update document status to processing
        if doc.status == DocumentStatus.PENDING:
            await self.doc_repo.update_document_status(
                doc_id,
                DocumentStatus.PROCESSING
            )
            logger.debug(f"Document {doc.source} started processing")
        elif doc.status in [DocumentStatus.PROCESSING, DocumentStatus.COMPLETED]:
            # Document is already being processed or completed - skip to avoid duplicates
            logger.warning(
                f"Document {doc.source} (ID: {doc_id}) is already in status {doc.status}, "
                "skipping duplicate processing"
            )
            return
        elif doc.status == DocumentStatus.FAILED:
            # Failed documents should only be retried explicitly
            logger.warning(
                f"Document {doc.source} (ID: {doc_id}) is in FAILED status, "
                "use retry mechanism to reprocess"
            )
            return
        
        # Process through remaining steps
        try:
            while current_step_idx < len(self._steps):
                step = self._steps[current_step_idx]
                
                # Update step status
                await self.doc_repo.update_step_status(doc_id, step.name, "processing")
                await self.doc_repo.update_document_status(
                    doc_id,
                    DocumentStatus.PROCESSING,
                    current_step=step.name
                )
                logger.debug(f"Document {doc.source} processing step '{step.name}'")
                
                # Handle remote transformations differently
                if step.is_remote and isinstance(step.transformation, IRemoteTransformation):
                    # Submit to remote service
                    tracking_id = await step.transformation.submit(doc)
                    logger.info(
                        f"Document {doc.source} submitted to remote service "
                        f"'{step.name}' (tracking: {tracking_id})"
                    )
                    # Don't continue - wait for callback
                    return
                
                # Apply transformation
                result = await step.transformation.process(doc)
                
                if result is None:
                    # Document filtered out
                    await self.doc_repo.update_step_status(doc_id, step.name, "filtered")
                    logger.info(f"Document {doc.source} filtered by step '{step.name}'")
                    return
                
                # Update step as completed
                await self.doc_repo.update_step_status(doc_id, step.name, "completed")
                
                # Check if this is a parent document for async parsing
                # If so, mark it as processing and stop here - it will be completed via callback
                if isinstance(result, DocumentRecord):
                    if result.metadata.get('is_parent_document') == 'true':
                        # IMPORTANT: Save the metadata first (includes docling_job_id)
                        await self.doc_repo.update_document_metadata(doc_id, result.metadata)
                        
                        await self.doc_repo.update_document_status(
                            doc_id,
                            DocumentStatus.PROCESSING,
                            current_step="waiting_for_async_parsing"
                        )
                        logger.info(
                            f"Document {doc.source} is parent for async parsing "
                            f"(job_id={result.metadata.get('docling_job_id')}), "
                            f"halting pipeline - will be completed via callback"
                        )
                        return
                
                # Handle fan-out (multiple documents from one)
                if not isinstance(result, DocumentRecord):
                    logger.debug(f"Step '{step.name}' produced fan-out for {doc.source}")
                    for child_doc in result:  # type: ignore
                        # Create new document records for children
                        child_doc.job_id = job_id
                        child_doc.step_statuses = {s: "pending" for s in self.job.pipeline_steps}
                        await self.doc_repo.create_document(child_doc)
                        
                        # Enqueue child from next step
                        await self.backend.enqueue(
                            MAIN_QUEUE,
                            body={
                                "type": "document",
                                "doc_id": child_doc.id,
                                "job_id": job_id,
                                "current_step_idx": current_step_idx + 1,
                            }
                        )
                    return
                
                doc = result
                current_step_idx += 1
            
            # All steps completed, write to sinks
            await self._write_to_sinks(doc)
            
            # Mark document as completed
            await self.doc_repo.update_document_status(
                doc_id,
                DocumentStatus.COMPLETED
            )
            logger.info(f"Document {doc.source} completed successfully")
            
            # Check if this is a child node from async parsing
            # If so, check if all siblings are done and mark parent as completed
            if doc.metadata.get('parent_document_id'):
                await self._check_parent_completion(doc.metadata['parent_document_id'], doc.metadata.get('total_nodes', 0))
            
        except Exception as e:
            # Mark document as failed
            step_name = self._steps[current_step_idx].name if current_step_idx < len(self._steps) else "sink"
            await self.doc_repo.update_document_status(
                doc_id,
                DocumentStatus.FAILED,
                error_message=str(e),
                error_step=step_name
            )
            if current_step_idx < len(self._steps):
                await self.doc_repo.update_step_status(doc_id, step_name, "failed")
            
            logger.error(
                f"Document {doc.source} failed at step '{step_name}': {e}",
                exc_info=True
            )
            raise
    
    async def _handle_remote_callback(self, body: dict):
        """Handle callbacks from remote transformation services."""
        if body.get("type") != "callback":
            logger.warning(f"Unknown message type in remote queue: {body.get('type')}")
            return
        
        tracking_id = body["tracking_id"]
        step_idx = body["step_idx"]
        result_data = body["result"]
        
        # Find the remote transformation
        step = self._steps[step_idx]
        if not isinstance(step.transformation, IRemoteTransformation):
            logger.error(f"Step {step_idx} is not a remote transformation")
            return
        
        # Process the callback
        doc = await step.transformation.handle_callback(tracking_id, result_data)
        
        # Mark step as completed
        await self.doc_repo.update_step_status(doc.id, step.name, "completed")
        logger.info(f"Remote step '{step.name}' completed for document {doc.source}")
        
        # Continue processing from next step
        await self.backend.enqueue(
            MAIN_QUEUE,
            body={
                "type": "document",
                "doc_id": doc.id,
                "job_id": doc.job_id,
                "current_step_idx": step_idx + 1,
            }
        )
    
    async def _check_parent_completion(self, parent_doc_id: str, expected_children: int):
        """
        Check if all child nodes for a parent document have completed.
        If so, mark the parent document as completed.
        """
        try:
            # Get all child documents for this parent
            from fetchcraft.ingestion.repository import PostgresDocumentRepository
            if not isinstance(self.doc_repo, PostgresDocumentRepository):
                return  # Can't check without direct DB access
            
            # Count completed children
            async with self.doc_repo.pool.acquire() as conn:
                completed_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM ingestion_documents
                    WHERE metadata->>'parent_document_id' = $1
                    AND status = 'completed'
                    """,
                    parent_doc_id
                )
                
                total_children = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM ingestion_documents
                    WHERE metadata->>'parent_document_id' = $1
                    """,
                    parent_doc_id
                )
                
                logger.info(
                    f"Parent {parent_doc_id}: {completed_count}/{total_children} children completed "
                    f"(expected: {expected_children})"
                )
                
                # Check if all children are done
                # Note: We check if all actual children are completed, not if count matches expected
                # The expected count might differ slightly due to parsing edge cases
                if completed_count == total_children and total_children > 0:
                    # Verify parent is in PROCESSING state before updating
                    parent = await self.doc_repo.get_document(parent_doc_id)
                    if parent and parent.status == DocumentStatus.PROCESSING:
                        # Mark parent as completed
                        await self.doc_repo.update_document_status(
                            parent_doc_id,
                            DocumentStatus.COMPLETED
                        )
                        logger.info(
                            f"All {total_children} child nodes completed - "
                            f"marked parent document {parent_doc_id} as COMPLETED"
                        )
                    else:
                        logger.debug(f"Parent {parent_doc_id} already in final state: {parent.status if parent else 'not found'}")
                elif total_children > 0:
                    logger.debug(
                        f"Parent {parent_doc_id} still waiting: {completed_count}/{total_children} children done"
                    )
        
        except Exception as e:
            logger.error(f"Error checking parent completion for {parent_doc_id}: {e}", exc_info=True)
    
    async def _write_to_sinks(self, doc: DocumentRecord):
        """Write document to all sinks."""
        logger.debug(f"Writing document {doc.source} to {len(self._sinks)} sinks")
        
        for sink in self._sinks:
            sink_name = f"sink:{sink.get_name()}"
            try:
                await self.doc_repo.update_step_status(doc.id, sink_name, "processing")
                await sink.write(doc)
                await self.doc_repo.update_step_status(doc.id, sink_name, "completed")
                logger.debug(f"Document {doc.source} written to sink '{sink.get_name()}'")
            except Exception as e:
                await self.doc_repo.update_step_status(doc.id, sink_name, "failed")
                logger.error(
                    f"Error writing document {doc.source} to sink '{sink.get_name()}': {e}",
                    exc_info=True
                )
                # Re-raise to trigger document failure
                raise
