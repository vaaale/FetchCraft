"""
Interface definitions for ingestion pipeline components.

This module defines the core interfaces that pipeline components must implement.
All interfaces use ABC (Abstract Base Class) to enforce contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterable, Iterable, Optional

from fetchcraft.ingestion.models import DocumentRecord


class IConnector(ABC):
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


class ISource(ABC):
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
    def get_connector(self) -> IConnector:
        """
        Get the underlying connector.
        
        Returns:
            The connector instance
        """
        pass


class ITransformation(ABC):
    """
    Interface for pipeline transformations.
    
    A transformation processes a DocumentRecord and either:
    - Returns a modified DocumentRecord (1:1)
    - Returns multiple DocumentRecords (1:N fan-out)
    - Returns None (filter out the document)
    
    Transformations can be synchronous or asynchronous.
    """
    
    @abstractmethod
    async def process(
        self,
        record: DocumentRecord
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Process a document record.
        
        Args:
            record: The document record to process
            
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


class IRemoteTransformation(ITransformation):
    """
    Interface for remote/async transformations with callbacks.
    
    Remote transformations delegate processing to external services and
    use callbacks to receive results. They maintain persistent state
    to survive restarts.
    """
    
    @abstractmethod
    async def submit(self, record: DocumentRecord) -> str:
        """
        Submit a document for remote processing.
        
        Args:
            record: The document record to process
            
        Returns:
            A tracking ID for the submitted work
        """
        pass
    
    @abstractmethod
    async def handle_callback(
        self,
        tracking_id: str,
        result: dict
    ) -> DocumentRecord:
        """
        Handle a callback from the remote service.
        
        Args:
            tracking_id: The tracking ID from submit()
            result: The processing result from the remote service
            
        Returns:
            The processed document record
        """
        pass
    
    @abstractmethod
    async def get_pending_count(self) -> int:
        """
        Get the count of pending remote operations.
        
        Returns:
            Number of documents awaiting callback
        """
        pass


class ISink(ABC):
    """
    Interface for pipeline sinks.
    
    A sink is the final destination for processed documents. Examples include
    vector stores, document stores, or external APIs.
    """
    
    @abstractmethod
    async def write(self, record: DocumentRecord) -> None:
        """
        Write a processed document record.
        
        Args:
            record: The processed document record
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the sink name.
        
        Returns:
            Name of the sink (defaults to class name)
        """
        return self.__class__.__name__


class IQueueBackend(ABC):
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
