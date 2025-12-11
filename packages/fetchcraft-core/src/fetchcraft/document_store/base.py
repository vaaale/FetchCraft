"""
Base class for document stores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic

from ..node import Node

D = TypeVar('D', bound=Node)


class DocumentStore(ABC, Generic[D]):
    """
    Abstract base class for document stores.
    
    A document store manages the storage and retrieval of full documents
    (typically DocumentNodes) separately from their vector embeddings.
    
    This separation allows:
    - Efficient vector search in vector stores
    - Full document retrieval from document stores
    - Independent scaling of storage layers
    """
    
    @abstractmethod
    async def add_document(self, document: Node) -> str:
        """
        Add a single document to the store.
        
        Args:
            document: The document to store
            
        Returns:
            The document ID
        """
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Node]) -> List[str]:
        """
        Add multiple documents to the store.
        
        Args:
            documents: List of documents to store
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Node]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_documents(self, document_ids: List[str]) -> List[Node]:
        """
        Retrieve multiple documents by their IDs.
        
        Args:
            document_ids: List of document IDs to retrieve
            
        Returns:
            List of documents (may be fewer than requested if some not found)
        """
        pass
    
    @abstractmethod
    async def update_document(self, document: Node) -> bool:
        """
        Update an existing document.
        
        Args:
            document: The document with updated content
            
        Returns:
            True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete multiple documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the store.
        
        Args:
            document_id: The ID of the document to check
            
        Returns:
            True if document exists, False otherwise
        """
        pass

    @abstractmethod
    async def all_ids(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Return all document IDs in the store.

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[D]:
        """
        List documents with pagination and optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Optional filters to apply (implementation-specific)
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the store.
        
        Args:
            filters: Optional filters to apply (implementation-specific)
            
        Returns:
            Number of documents
        """
        pass
