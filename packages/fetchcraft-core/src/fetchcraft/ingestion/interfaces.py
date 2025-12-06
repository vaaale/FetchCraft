"""
Interface definitions for ingestion pipeline components.

This module defines the core interfaces that pipeline components must implement.
All interfaces use ABC (Abstract Base Class) to enforce contracts.

Naming Convention:
- Interfaces do NOT use 'I' prefix (e.g., Source, Transformation, Sink)
- Concrete implementations use descriptive names (e.g., ConnectorSource, ParsingTransformation)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterable, Iterable, Optional, Any, Dict, Literal

from fetchcraft.ingestion.models import DocumentRecord


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
    
    A transformation processes a DocumentRecord and either:
    - Returns a modified DocumentRecord (1:1)
    - Returns multiple DocumentRecords (1:N fan-out)
    - Returns None (filter out the document)
    
    Transformations can be synchronous or asynchronous (callback-based).
    Set is_async=True for transformations that use external services with callbacks.
    """
    
    @property
    def is_async(self) -> bool:
        """
        Whether this transformation executes asynchronously via callbacks.
        
        Async transformations submit work to external services and receive
        results via callbacks. The pipeline will pause document processing
        until the callback is received.
        
        Returns:
            True if async, False for synchronous execution
        """
        return False
    
    @abstractmethod
    async def process(
        self,
        record: DocumentRecord,
        context: Optional[dict] = None
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Process a document record synchronously.
        
        Args:
            record: The document record to process
            context: Optional pipeline context with shared data
            
        Returns:
            - DocumentRecord: A single transformed record
            - Iterable[DocumentRecord]: Multiple records (fan-out)
            - None: Filter out this record
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the transformation name.
        
        Returns:
            Name of the transformation (defaults to class name)
        """
        return self.__class__.__name__


class AsyncTransformation(Transformation):
    """
    Interface for async transformations that use callbacks.
    
    Async transformations submit work to external services and receive
    results via callbacks. The pipeline pauses document processing until
    the callback is received with status COMPLETED.
    
    Lifecycle:
    1. Pipeline calls submit() with document and task_id
    2. Transformation sends work to external service with callback_url
    3. External service sends callbacks to callback_url
    4. Pipeline calls on_message() for each callback
    5. When status is COMPLETED, pipeline continues with returned document
    """
    
    @property
    def is_async(self) -> bool:
        """Async transformations always return True."""
        return True
    
    async def process(
        self,
        record: DocumentRecord
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Not used for async transformations - raises error if called.
        
        Use submit() and on_message() instead.
        """
        raise NotImplementedError(
            "AsyncTransformation does not support process(). "
            "Use submit() and on_message() instead."
        )
    
    @abstractmethod
    async def submit(
        self,
        record: DocumentRecord,
        task_id: str,
        callback_url: str
    ) -> None:
        """
        Submit work to external service.
        
        Args:
            record: The document record to process
            task_id: Unique task identifier for correlation
            callback_url: URL where callbacks should be sent
        """
        pass
    
    @abstractmethod
    async def on_message(
        self,
        message: Dict[str, Any],
        status: Literal['PROCESSING', 'COMPLETED', 'FAILED']
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Handle callback message from external service.
        
        Args:
            message: Callback payload from external service
            status: Callback status
            
        Returns:
            - If status is COMPLETED: Processed DocumentRecord(s)
            - If status is PROCESSING: None (continue waiting)
            - If status is FAILED: Should raise an exception
        """
        pass


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
IRemoteTransformation = AsyncTransformation
ISink = Sink
IQueueBackend = QueueBackend
