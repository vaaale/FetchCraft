"""
Interface definitions for ingestion pipeline components.

This module defines the core interfaces that pipeline components must implement.
All interfaces use ABC (Abstract Base Class) to enforce contracts.

Naming Convention:
- Interfaces do NOT use 'I' prefix (e.g., Source, Transformation, Sink)
- Concrete implementations use descriptive names (e.g., ConnectorSource, ParsingTransformation)
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    AsyncIterable,
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Any,
    Dict,
    Union,
    TYPE_CHECKING,
)

from fetchcraft.ingestion.models import DocumentRecord

if TYPE_CHECKING:
    from fetchcraft.ingestion.models import DocumentRecord as DocRecord

logger = logging.getLogger(__name__)


# ============================================================================
# Record and Result Types
# ============================================================================

class Record(dict):
    """
    Flexible record type for transformation data.
    
    Behaves like a dict but provides attribute-style access for convenience.
    This is the primary data type passed between transformations.
    
    The 'content' field stores base64-encoded file content separately from metadata.
    """
    
    _content: Optional[str] = None
    
    def __init__(self, *args, content: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '_content', content)
    
    @property
    def content(self) -> Optional[str]:
        """Get the base64-encoded content."""
        return object.__getattribute__(self, '_content')
    
    @content.setter
    def content(self, value: Optional[str]) -> None:
        """Set the base64-encoded content."""
        object.__setattr__(self, '_content', value)
    
    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to dict keys."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Record has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Allow attribute-style setting of dict keys."""
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        """Allow attribute-style deletion of dict keys."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Record has no attribute '{key}'")

    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return {key: value for key, value in self.items() if key != 'content'}

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested value safely using a sequence of keys."""
        val: Any = self
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val
    
    def copy(self) -> "Record":
        """Create a shallow copy of the record."""
        return Record(super().copy(), content=self.content)
    
    def deep_copy(self) -> "Record":
        """Create a deep copy of the record."""
        import copy
        return Record(copy.deepcopy(dict(self)), content=self.content)


@dataclass
class AsyncRemote:
    """
    Indicates an async remote job was submitted.
    
    When a transformation returns this, the pipeline will:
    1. Mark the task as waiting for callback
    2. Stop processing this document until callback arrives
    3. Resume processing when callback is received
    
    Attributes:
        task_id: Unique identifier for callback correlation
        metadata: Optional metadata about the pending job
    """
    task_id: str
    metadata: Optional[Dict[str, Any]] = None


# Type alias for deferred execution callable
DeferredCallable = Callable[[], Awaitable[Union[Record, Iterable[Record]]]]


@dataclass
class AsyncDeferred:
    """
    Wraps a coroutine for deferred execution in a thread/process pool.
    
    When a transformation returns this, the pipeline will:
    1. Execute the wrapped function without blocking other tasks
    2. Process the results when execution completes
    
    Use this for long-running local operations (e.g., LLM calls, heavy computation)
    that shouldn't block the processing of other documents.
    
    Attributes:
        func: Async callable that returns Record or Iterable[Record]
        metadata: Optional metadata about the deferred task
    """
    func: DeferredCallable
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    async def execute(self) -> Union[Record, Iterable[Record]]:
        """Execute the deferred function."""
        return await self.func()


# Type alias for all possible transformation results
TransformationResult = Union[
    Record,                              # Single record (sync)
    Iterable[Record],                    # Multiple records / fan-out (sync)
    AsyncGenerator[Record, None],        # Async generator for fan-out
    AsyncRemote,                         # Remote callback-based
    AsyncDeferred,                       # Deferred local execution
    None,                                # Filter out
]

# Type alias for post_process results
PostProcessResult = Union[Record, Iterable[Record]]


class Connector(ABC):
    """
    Interface for data source connectors.
    
    A connector is responsible for discovering and reading files from a data source
    (filesystem, S3, HTTP, etc.). It produces File objects that can be parsed.
    """
    
    @abstractmethod
    async def read(self) -> AsyncIterable['File']:  # noqa: F821
        """
        Read files from the data source.
        
        Yields:
            File objects from the data source
        """
        pass


class Source(ABC):
    """
    Interface for pipeline sources.
    
    A source uses a connector to discover files, parses them using appropriate
    parsers, and produces DocumentRecord objects that enter the pipeline.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the source name.

        Returns:
            Name of the source (defaults to class name)
        """
        pass

    @abstractmethod
    async def read(self) -> AsyncIterable[DocumentRecord]:
        """
        Read and parse files from the connector.
        
        Yields:
            DocumentRecord objects ready for pipeline processing
        """
        pass
    
    @abstractmethod
    def get_connector(self) -> Connector:
        """
        Get the underlying connector.
        
        Returns:
            The connector instance
        """
        pass


class Transformation(ABC):
    """
    Interface for pipeline transformations.
    
    A transformation processes a Record and can return:
    - Record: A single transformed record (sync)
    - Iterable[Record]: Multiple records for fan-out (sync)
    - AsyncGenerator[Record]: Async generator for fan-out
    - AsyncRemote: Indicates remote job submitted, await callback
    - AsyncDeferred: Wraps long-running local task for deferred execution
    - None: Filter out this record
    
    The transformation behavior is determined by the return type at runtime,
    eliminating the need for separate sync/async transformation classes.
    """
    
    @abstractmethod
    async def process(
        self,
        record: Record,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransformationResult:
        """
        Process a record.
        
        Args:
            record: The record to process (extracted from DocumentRecord.metadata)
            correlation_id: Unique identifier for this processing task, used for
                callback correlation in async/remote transformations
            context: Optional pipeline context with shared data
            
        Returns:
            - Record: Single result (sync)
            - Iterable[Record]: Multiple results / fan-out (sync)
            - AsyncGenerator[Record]: Async fan-out
            - AsyncRemote: Remote job submitted, await callback
            - AsyncDeferred: Deferred execution in pool
            - None: Filter out this record
        """
        pass
    
    def post_process(self, message: Dict[str, Any]) -> PostProcessResult:
        """
        Post-process callback message from remote service.
        
        Override this method to transform callback data into the desired format.
        Default implementation wraps the message as a Record.
        
        This is called by the pipeline when handling AsyncRemote callbacks.
        Users should override this instead of on_callback for custom result transformation.
        
        Args:
            message: Raw message from remote service callback
            
        Returns:
            Record or Iterable[Record] for fan-out
        """
        return Record(message)
    
    async def on_callback(
        self,
        correlation_id: str,
        message: Dict[str, Any],
        status: str
    ) -> Optional[PostProcessResult]:
        """
        Handle callback for AsyncRemote tasks.
        
        This is called by the pipeline - users should NOT override this.
        Override post_process() instead to customize result transformation.
        
        Args:
            correlation_id: The correlation ID for matching callbacks to tasks
            message: Callback payload from remote service
            status: Callback status ('PROCESSING', 'COMPLETED', 'FAILED')
            
        Returns:
            - If PROCESSING: Result from post_process()
            - If COMPLETED: None (completion is handled by pipeline)
            - If FAILED: Raises exception
        """
        if status == "PROCESSING":
            return self.post_process(message)
        elif status == "FAILED":
            error = message.get("error", "Unknown error from remote service")
            raise RuntimeError(f"Remote task {correlation_id} failed: {error}")
        # COMPLETED status is handled by pipeline
        return None
    
    def get_name(self) -> str:
        """
        Get the transformation name.
        
        Returns:
            Name of the transformation (defaults to class name)
        """
        return self.__class__.__name__


class Sink(ABC):
    """
    Interface for pipeline sinks.
    
    A sink is the final destination for processed documents. Examples include
    vector stores, document stores, or external APIs.
    """
    
    @abstractmethod
    async def write(self, record: DocumentRecord, context: Optional[dict] = None) -> None:
        """
        Write a processed document record.
        
        Args:
            record: The processed document record
            :param record: The document record
            :param context: Pipeline context
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the sink name.
        
        Returns:
            Name of the sink (defaults to class name)
        """
        return self.__class__.__name__


class QueueBackend(ABC):
    """
    Interface for async queue backends.
    
    Queue backends provide durable message queuing for the pipeline.
    Implementations can use different storage backends (SQLite, PostgreSQL, Redis, etc.).
    """
    
    @abstractmethod
    async def enqueue(
        self,
        queue_name: str,
        body: dict,
        delay_seconds: int = 0
    ) -> str:
        """
        Add a message to the queue.
        
        Args:
            queue_name: Name of the queue
            body: Message body
            delay_seconds: Delay before message becomes available
            
        Returns:
            Message ID
        """
        pass
    
    @abstractmethod
    async def lease_next(
        self,
        queue_name: str,
        lease_seconds: int = 30
    ) -> Optional['QueueMessage']:  # noqa: F821
        """
        Lease the next available message from the queue.
        
        Args:
            queue_name: Name of the queue
            lease_seconds: How long to lease the message
            
        Returns:
            The leased message, or None if queue is empty
        """
        pass
    
    @abstractmethod
    async def ack(self, queue_name: str, message_id: str) -> None:
        """
        Acknowledge successful processing of a message.
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to acknowledge
        """
        pass
    
    @abstractmethod
    async def nack(
        self,
        queue_name: str,
        message_id: str,
        requeue_delay_seconds: int = 0
    ) -> None:
        """
        Negative acknowledge a message (requeue for retry).
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to requeue
            requeue_delay_seconds: Delay before message is available again
        """
        pass
    
    @abstractmethod
    async def has_pending(self, *queue_names: str) -> bool:
        """
        Check if any of the queues have pending messages.
        
        Args:
            queue_names: Names of queues to check
            
        Returns:
            True if any queue has pending messages
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> dict:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with:
            - total_messages: Total message count
            - by_state: Dict of state -> count
            - by_queue: Dict of queue -> count
            - failed_messages: Count of failed messages
            - oldest_pending: Timestamp of oldest pending message (or None)
        """
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """
        Check if the queue backend is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


# Backwards compatibility aliases (deprecated - will be removed in future version)
IConnector = Connector
ISource = Source
ITransformation = Transformation
ISink = Sink
IQueueBackend = QueueBackend

# Legacy alias - AsyncTransformation is no longer needed
# Transformations now signal async behavior via return type
AsyncTransformation = Transformation
