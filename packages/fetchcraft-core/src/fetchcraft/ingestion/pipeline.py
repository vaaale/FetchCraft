"""
Enhanced ingestion pipeline with job and document tracking.

This module implements an async, durable ingestion pipeline that tracks
documents through each processing step and provides visibility into
job execution status.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable, Any, Iterable

from fetchcraft.ingestion.interfaces import (
    Source,
    Transformation,
    Sink,
    QueueBackend,
    Record,
    AsyncRemote,
    AsyncDeferred,
    TransformationResult,
)
from fetchcraft.ingestion.models import (
    IngestionJob,
    DocumentRecord,
    TaskRecord,
    JobStatus,
    DocumentStatus,
    TaskStatus,
    utcnow,
)
from fetchcraft.ingestion.repository import (
    JobRepository,
    DocumentRepository,
    TaskRepository,
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
        name: Step name (for tracking)
    """
    transformation: Transformation
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
        backend: QueueBackend,
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
        backend: QueueBackend,
        job_repo: JobRepository,
        doc_repo: DocumentRepository,
        task_repo: Optional[TaskRepository] = None,
        num_workers: int = 1,
        callback_base_url: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            job: The ingestion job configuration
            backend: Queue backend for message passing
            job_repo: Repository for job persistence
            doc_repo: Repository for document tracking
            task_repo: Repository for task tracking (optional, enables detailed task tracking)
            num_workers: Number of concurrent workers for processing documents (default: 1)
            callback_base_url: Base URL for async transformation callbacks
        """
        self.job = job
        self.backend = backend
        self.job_repo = job_repo
        self.doc_repo = doc_repo
        self.task_repo = task_repo
        self.num_workers = max(1, num_workers)  # Ensure at least 1 worker
        self.callback_base_url = callback_base_url

        self._source: Optional[Source] = None
        self._steps: List[PipelineStep] = []
        self._sinks: List[Sink] = []
        
        self._main_workers: List[Worker] = []
        self._context = context
        
        logger.info(
            f"Initialized pipeline for job '{job.name}' (ID: {job.id}) "
            f"with {self.num_workers} worker(s)"
        )

    # ========== Builder API ==========

    def context(self, items: dict):
        self._context.update(items)
        return self

    def source(self, src: Source) -> "TrackedIngestionPipeline":
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
        transformation: Transformation,
    ) -> "TrackedIngestionPipeline":
        """
        Add a transformation step to the pipeline.
        
        Args:
            transformation: The transformation to add
            
        Returns:
            Self for chaining
        """
        step = PipelineStep(transformation=transformation)
        self._steps.append(step)
        self.job.pipeline_steps.append(step.name)
        logger.debug(f"Added transformation: {step.name}")
        return self
    
    def add_sink(self, sink: Sink) -> "TrackedIngestionPipeline":
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
        
        # Start all workers concurrently
        await asyncio.gather(*[worker.start() for worker in self._main_workers])
        
        logger.info(f"Pipeline workers started: {self.num_workers} main worker(s)")
    
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
        """Wait until all queues are empty and no documents are in PENDING or PROCESSING status."""
        logger.info("Waiting for pipeline to become idle...")
        iteration = 0
        
        while True:
            try:
                has_work = await self.backend.has_pending(MAIN_QUEUE)
                
                # Check for documents still in PROCESSING (e.g., parent docs waiting for children)
                processing_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PROCESSING
                )
                
                # Also check for documents in PENDING status (e.g., child docs from async parsing)
                # These may have been created by callbacks but not yet picked up by workers
                pending_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PENDING
                )
                
                # Check for pending async tasks
                pending_tasks = []
                if self.task_repo:
                    pending_tasks = await self.task_repo.get_pending_async_tasks(self.job.id)
                
                if not has_work and len(processing_docs) == 0 and len(pending_docs) == 0 and len(pending_tasks) == 0:
                    logger.info(f"Queues empty and no pending/processing documents after {iteration} checks")
                    break
                
                if iteration % 20 == 0:
                    logger.info(
                        f"Still processing... (iteration {iteration}, "
                        f"pending_docs={len(pending_docs)}, processing_docs={len(processing_docs)}, "
                        f"pending_tasks={len(pending_tasks)})"
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
                has_work = await self.backend.has_pending(MAIN_QUEUE)
                processing_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PROCESSING
                )
                pending_docs = await self.doc_repo.get_documents_by_status(
                    self.job.id,
                    DocumentStatus.PENDING
                )
                pending_tasks = []
                if self.task_repo:
                    pending_tasks = await self.task_repo.get_pending_async_tasks(self.job.id)
                
                if has_work or len(processing_docs) > 0 or len(pending_docs) > 0 or len(pending_tasks) > 0:
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
        await asyncio.gather(
            *[worker.stop() for worker in self._main_workers],
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
                
                # Create task record for tracking
                task = None
                if self.task_repo:
                    task = TaskRecord(
                        job_id=job_id,
                        document_id=doc_id,
                        transformation_name=step.name,
                        step_index=current_step_idx,
                        status=TaskStatus.PENDING,
                        is_async=False,  # Will be updated if AsyncRemote is returned
                    )
                    await self.task_repo.create_task(task)
                    await self.task_repo.set_task_started(task.id)
                
                # Extract Record from DocumentRecord metadata for transformation
                record = Record(doc.metadata)
                
                # Use task.id as correlation_id for callback correlation
                correlation_id = task.id if task else doc_id
                
                # Apply transformation
                try:
                    result = await step.transformation.process(
                        record,
                        correlation_id=correlation_id,
                        context=self._context,
                    )
                except Exception as e:
                    # Send to error queue and mark as failed
                    await self._send_to_error_queue(
                        doc=doc,
                        step_name=step.name,
                        error=e,
                        task=task
                    )
                    raise
                
                # Handle different result types
                handled = await self._handle_transformation_result(
                    result=result,
                    doc=doc,
                    step=step,
                    task=task,
                    current_step_idx=current_step_idx,
                    job_id=job_id,
                )
                
                if handled:
                    # Result was handled (async, deferred, or fan-out) - stop processing
                    return
                
                # Result was a single Record - update doc metadata and continue
                if isinstance(result, Record):
                    doc.metadata = dict(result)
                
                # Mark task as completed
                if self.task_repo and task:
                    await self.task_repo.set_task_completed(task.id)
                
                # Update step as completed
                await self.doc_repo.update_step_status(doc_id, step.name, "completed")
                
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
                await self._check_parent_completion(
                    parent_doc_id=doc.metadata['parent_document_id'],
                    expected_children=doc.metadata.get('total_nodes', 0),
                    parent_task_id=doc.metadata.get('parent_task_id'),
                )
            
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
            
            # Check if this is a child node - if so, check parent completion
            # (parent may need to be marked as failed if all children are done)
            if doc.metadata.get('parent_document_id'):
                await self._check_parent_completion(
                    parent_doc_id=doc.metadata['parent_document_id'],
                    expected_children=doc.metadata.get('total_nodes', 0),
                    parent_task_id=doc.metadata.get('parent_task_id'),
                )
            
            raise
    
    async def _handle_transformation_result(
        self,
        result: TransformationResult,
        doc: DocumentRecord,
        step: PipelineStep,
        task: Optional[TaskRecord],
        current_step_idx: int,
        job_id: str,
    ) -> bool:
        """
        Handle the result from a transformation.
        
        Args:
            result: The transformation result
            doc: The document being processed
            step: The current pipeline step
            task: The task record (if tracking enabled)
            current_step_idx: Current step index
            job_id: The job ID
            
        Returns:
            True if result was handled and processing should stop,
            False if processing should continue to next step
        """
        doc_id = doc.id
        
        # Handle None - document filtered out
        if result is None:
            await self.doc_repo.update_step_status(doc_id, step.name, "filtered")
            if self.task_repo and task:
                await self.task_repo.set_task_completed(task.id)
            logger.info(f"Document {doc.source} filtered by step '{step.name}'")
            return True
        
        # Handle AsyncRemote - remote job submitted, wait for callback
        if isinstance(result, AsyncRemote):
            # Update task to async mode
            if self.task_repo and task:
                task.is_async = True
                await self.task_repo.update_task_metadata(task.id, {
                    "async_task_id": result.task_id,
                    **(result.metadata or {})
                })
                await self.task_repo.set_task_submitted(task.id)
            
            # Store the transformation reference for callback handling
            await self.doc_repo.update_document_metadata(doc_id, {
                **doc.metadata,
                "_async_task_id": result.task_id,
                "_async_step_idx": current_step_idx,
            })
            
            logger.info(
                f"Document {doc.source} submitted to remote service "
                f"'{step.name}' (task_id: {result.task_id})"
            )
            return True
        
        # Handle AsyncDeferred - execute in background without blocking
        if isinstance(result, AsyncDeferred):
            asyncio.create_task(
                self._execute_deferred(
                    deferred=result,
                    doc=doc,
                    step=step,
                    task=task,
                    current_step_idx=current_step_idx,
                    job_id=job_id,
                )
            )
            logger.debug(
                f"Document {doc.source} deferred execution started for '{step.name}'"
            )
            return True
        
        # Handle async generator - fan-out
        if hasattr(result, '__anext__'):
            await self._handle_async_generator_fanout(
                result=result,  # type: ignore
                doc=doc,
                step=step,
                task=task,
                current_step_idx=current_step_idx,
                job_id=job_id,
            )
            return True
        
        # Handle iterable (but not Record which is also iterable as dict)
        if isinstance(result, Iterable) and not isinstance(result, (Record, dict)):
            await self._handle_sync_fanout(
                result=result,
                doc=doc,
                step=step,
                task=task,
                current_step_idx=current_step_idx,
                job_id=job_id,
            )
            return True
        
        # Single Record result - continue processing
        return False
    
    async def _execute_deferred(
        self,
        deferred: AsyncDeferred,
        doc: DocumentRecord,
        step: PipelineStep,
        task: Optional[TaskRecord],
        current_step_idx: int,
        job_id: str,
    ) -> None:
        """Execute a deferred task and process its results."""
        doc_id = doc.id
        
        try:
            result = await deferred.execute()
            
            # Handle the result
            if isinstance(result, Record):
                # Single result - update doc and continue pipeline
                doc.metadata = dict(result)
                
                if self.task_repo and task:
                    await self.task_repo.set_task_completed(task.id)
                await self.doc_repo.update_step_status(doc_id, step.name, "completed")
                await self.doc_repo.update_document_metadata(doc_id, doc.metadata)
                
                # Re-enqueue for next step
                await self.backend.enqueue(
                    MAIN_QUEUE,
                    body={
                        "type": "document",
                        "doc_id": doc_id,
                        "job_id": job_id,
                        "current_step_idx": current_step_idx + 1,
                    }
                )
            else:
                # Fan-out result
                await self._handle_sync_fanout(
                    result=result,
                    doc=doc,
                    step=step,
                    task=task,
                    current_step_idx=current_step_idx,
                    job_id=job_id,
                )
                
        except Exception as e:
            await self._send_to_error_queue(
                doc=doc,
                step_name=step.name,
                error=e,
                task=task
            )
            logger.error(
                f"Deferred execution failed for {doc.source} at '{step.name}': {e}",
                exc_info=True
            )
    
    async def _handle_async_generator_fanout(
        self,
        result: Any,  # AsyncGenerator[Record, None]
        doc: DocumentRecord,
        step: PipelineStep,
        task: Optional[TaskRecord],
        current_step_idx: int,
        job_id: str,
    ) -> None:
        """Handle fan-out from an async generator."""
        doc_id = doc.id
        child_count = 0
        
        try:
            async for child_record in result:
                child_doc = self._create_child_document(
                    parent_doc=doc,
                    child_record=child_record,
                    job_id=job_id,
                    step_name=step.name,
                )
                await self.doc_repo.create_document(child_doc)
                
                await self.backend.enqueue(
                    MAIN_QUEUE,
                    body={
                        "type": "document",
                        "doc_id": child_doc.id,
                        "job_id": job_id,
                        "current_step_idx": current_step_idx + 1,
                    }
                )
                child_count += 1
            
            if self.task_repo and task:
                await self.task_repo.set_task_completed(task.id)
            await self.doc_repo.update_step_status(doc_id, step.name, "completed")
            
            logger.debug(
                f"Step '{step.name}' produced {child_count} children for {doc.source}"
            )
            
        except Exception as e:
            await self._send_to_error_queue(
                doc=doc,
                step_name=step.name,
                error=e,
                task=task
            )
            raise
    
    async def _handle_sync_fanout(
        self,
        result: Iterable[Record],
        doc: DocumentRecord,
        step: PipelineStep,
        task: Optional[TaskRecord],
        current_step_idx: int,
        job_id: str,
    ) -> None:
        """Handle fan-out from a sync iterable."""
        doc_id = doc.id
        child_count = 0
        
        for child_record in result:
            child_doc = self._create_child_document(
                parent_doc=doc,
                child_record=child_record,
                job_id=job_id,
                step_name=step.name,
            )
            await self.doc_repo.create_document(child_doc)
            
            await self.backend.enqueue(
                MAIN_QUEUE,
                body={
                    "type": "document",
                    "doc_id": child_doc.id,
                    "job_id": job_id,
                    "current_step_idx": current_step_idx + 1,
                }
            )
            child_count += 1
        
        if self.task_repo and task:
            await self.task_repo.set_task_completed(task.id)
        await self.doc_repo.update_step_status(doc_id, step.name, "completed")
        
        logger.debug(
            f"Step '{step.name}' produced {child_count} children for {doc.source}"
        )
    
    def _create_child_document(
        self,
        parent_doc: DocumentRecord,
        child_record: Record,
        job_id: str,
        step_name: str,
        parent_task_id: Optional[str] = None,
    ) -> DocumentRecord:
        """Create a child DocumentRecord from a Record."""
        child_doc = DocumentRecord(
            id=str(uuid.uuid4()),
            job_id=job_id,
            source=child_record.get("source", f"{parent_doc.source}#child"),
            status=DocumentStatus.PENDING,
            current_step=step_name,
            step_statuses={s: "pending" for s in self.job.pipeline_steps},
            metadata=dict(child_record),
        )
        # Mark current step as completed for child
        child_doc.step_statuses[step_name] = "completed"
        child_doc.metadata["parent_document_id"] = parent_doc.id
        if parent_task_id:
            child_doc.metadata["parent_task_id"] = parent_task_id
        return child_doc
    
    async def _send_to_error_queue(
        self,
        doc: DocumentRecord,
        step_name: str,
        error: Exception,
        task: Optional[TaskRecord] = None,
    ) -> None:
        """Send a failed document to the error queue."""
        error_body = {
            "type": "transformation_error",
            "doc_id": doc.id,
            "job_id": doc.job_id,
            "source": doc.source,
            "step_name": step_name,
            "error": str(error),
            "error_type": error.__class__.__name__,
            "metadata": doc.metadata,
        }
        
        if task:
            error_body["task_id"] = task.id
            if self.task_repo:
                await self.task_repo.update_task_status(
                    task.id, TaskStatus.FAILED, str(error)
                )
        
        await self.backend.enqueue(ERROR_QUEUE, body=error_body)
        await self.doc_repo.update_step_status(doc.id, step_name, "failed")
        
        logger.error(
            f"Document {doc.source} sent to error queue: {error}"
        )
    
    async def handle_task_callback(
        self,
        task_id: str,
        status: str,
        message: dict,
    ) -> bool:
        """
        Handle a callback for an async task.
        
        This method is called by the callback endpoint when an external service
        sends a callback. It processes the callback and continues document
        processing if the task is completed.
        
        Args:
            task_id: The task ID from the callback
            status: Callback status ('PROCESSING', 'COMPLETED', 'FAILED')
            message: Callback payload
            
        Returns:
            True if callback was handled successfully
        """
        # Get task record
        if not self.task_repo:
            logger.error("Task repository not configured - cannot handle callback")
            return False
        
        task = await self.task_repo.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        # Get the transformation
        if task.step_index >= len(self._steps):
            logger.error(f"Invalid step index {task.step_index} for task {task_id}")
            return False
        
        step = self._steps[task.step_index]
        
        # Get document
        doc = await self.doc_repo.get_document(task.document_id)
        if not doc:
            logger.error(f"Document {task.document_id} not found for task {task_id}")
            return False
        
        try:
            # Update task metadata with callback info
            task_metadata = task.metadata.copy()
            task_metadata['last_callback_status'] = status
            task_metadata['callback_count'] = task_metadata.get('callback_count', 0) + 1
            
            # Use transformation's on_callback method
            # task_id serves as the correlation_id
            try:
                result = await step.transformation.on_callback(task_id, message, status)
            except Exception as e:
                # Callback processing failed - send to error queue
                await self._send_to_error_queue(
                    doc=doc,
                    step_name=step.name,
                    error=e,
                    task=task
                )
                await self.doc_repo.update_document_status(
                    task.document_id,
                    DocumentStatus.FAILED,
                    error_message=str(e),
                    error_step=step.name
                )
                logger.error(f"Task {task_id} callback failed: {e}")
                return True
            
            if status == 'PROCESSING' and result is not None:
                # Transformation returned result(s) from callback - create child documents
                task_metadata['nodes_received'] = task_metadata.get('nodes_received', 0) + 1
                await self.task_repo.update_task_metadata(task_id, task_metadata)
                
                # Handle result (could be single Record or Iterable[Record])
                if isinstance(result, Record):
                    results = [result]
                else:
                    results = list(result)
                
                for idx, child_record in enumerate(results):
                    child_doc = self._create_child_document(
                        parent_doc=doc,
                        child_record=child_record,
                        job_id=task.job_id,
                        step_name=step.name,
                        parent_task_id=task_id,
                    )
                    await self.doc_repo.create_document(child_doc)
                    
                    # Enqueue child from next step
                    await self.backend.enqueue(
                        MAIN_QUEUE,
                        body={
                            "type": "document",
                            "doc_id": child_doc.id,
                            "job_id": task.job_id,
                            "current_step_idx": task.step_index + 1,
                        }
                    )
                    
                    logger.debug(
                        f"Created child document from callback (parent: {doc.id}, task: {task_id})"
                    )
                
                return True
            
            elif status == 'PROCESSING':
                # Processing callback with no result - just update status
                await self.task_repo.update_task_metadata(task_id, task_metadata)
                await self.task_repo.update_task_status(task_id, TaskStatus.WAITING)
                logger.debug(f"Task {task_id} still processing")
                return True
            
            elif status == 'COMPLETED':
                # Task completed - mark task and step as completed
                total_nodes = message.get('total_nodes', 0)
                task_metadata['total_nodes'] = total_nodes
                await self.task_repo.update_task_metadata(task_id, task_metadata)
                
                await self.task_repo.set_task_completed(task_id)
                await self.doc_repo.update_step_status(task.document_id, step.name, "completed")
                
                # Mark parent document as completed (children continue independently)
                await self.doc_repo.update_document_status(
                    task.document_id,
                    DocumentStatus.COMPLETED
                )
                
                logger.info(
                    f"Task {task_id} completed for document {doc.source} "
                    f"({total_nodes} nodes created)"
                )
                
                # Parent document processing stops here - children continue independently
                return True
            
            else:
                logger.warning(f"Unknown callback status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling callback for task {task_id}: {e}", exc_info=True)
            await self.task_repo.update_task_status(task_id, TaskStatus.FAILED, str(e))
            return False
    
    async def _check_parent_completion(
        self,
        parent_doc_id: str,
        expected_children: int,
        parent_task_id: Optional[str] = None,
    ):
        """
        Check if all child nodes for a parent document have completed or failed.
        Updates both the parent document and parent task status accordingly.
        
        - If all children completed: mark parent as COMPLETED
        - If any child failed and all are done: mark parent as FAILED
        """
        try:
            # Count children by status
            completed_count = await self.doc_repo.count_children_by_parent(
                parent_doc_id,
                DocumentStatus.COMPLETED
            )
            failed_count = await self.doc_repo.count_children_by_parent(
                parent_doc_id,
                DocumentStatus.FAILED
            )
            total_children = await self.doc_repo.count_children_by_parent(parent_doc_id)
            
            finished_count = completed_count + failed_count
            
            logger.info(
                f"Parent {parent_doc_id}: {completed_count} completed, {failed_count} failed, "
                f"{total_children} total (expected: {expected_children})"
            )
            
            # Check if all children are done (either completed or failed)
            if finished_count == total_children and total_children > 0:
                # Verify parent is not already in a final state
                parent = await self.doc_repo.get_document(parent_doc_id)
                if not parent:
                    logger.debug(f"Parent {parent_doc_id} not found")
                    return
                
                if parent.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED):
                    logger.debug(f"Parent {parent_doc_id} already in final state: {parent.status}")
                    return
                
                # Determine final status based on children
                if failed_count > 0:
                    # At least one child failed - mark parent as failed
                    final_status = DocumentStatus.FAILED
                    error_message = f"{failed_count} of {total_children} child documents failed"
                    
                    await self.doc_repo.update_document_status(
                        parent_doc_id,
                        final_status,
                        error_message=error_message
                    )
                    
                    # Update parent task if provided
                    if parent_task_id and self.task_repo:
                        await self.task_repo.update_task_status(
                            parent_task_id,
                            TaskStatus.FAILED,
                            error_message
                        )
                    
                    logger.warning(
                        f"Parent {parent_doc_id} marked as FAILED: {error_message}"
                    )
                else:
                    # All children completed successfully
                    final_status = DocumentStatus.COMPLETED
                    
                    await self.doc_repo.update_document_status(
                        parent_doc_id,
                        final_status
                    )
                    
                    # Update parent task if provided
                    if parent_task_id and self.task_repo:
                        await self.task_repo.set_task_completed(parent_task_id)
                    
                    logger.info(
                        f"All {total_children} child nodes completed - "
                        f"marked parent document {parent_doc_id} as COMPLETED"
                    )
            elif total_children > 0:
                logger.debug(
                    f"Parent {parent_doc_id} still waiting: {finished_count}/{total_children} children done"
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
                await sink.write(doc, context=self._context)
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
