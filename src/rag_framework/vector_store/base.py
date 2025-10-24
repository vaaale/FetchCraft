from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic, Type

from pydantic import BaseModel

# Type variable for document type
D = TypeVar('D', bound=BaseModel)

class VectorStore(ABC, Generic[D]):
    """
    Abstract base class for vector store implementations.
    
    This class defines the interface that all vector store implementations must follow.
    """
    
    @abstractmethod
    async def add_documents(self, documents: List[D], index_id: Optional[str] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document objects to add
            index_id: Optional index identifier to isolate documents
            
        Returns:
            List of document IDs that were added
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_embedding: List[float],
        k: int = 4,
        index_id: Optional[str] = None,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            index_id: Optional index identifier to filter search results
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str], index_id: Optional[str] = None) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            index_id: Optional index identifier to filter deletions
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str, index_id: Optional[str] = None) -> Optional[D]:
        """
        Retrieve a single document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            index_id: Optional index identifier to filter retrieval
            
        Returns:
            The document if found, None otherwise
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VectorStore':
        """
        Create a vector store instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            An instance of the vector store
        """
        pass
