from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic, Type, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr

# Type variable for document type
D = TypeVar('D', bound=BaseModel)

# Import filter types
try:
    from fetchcraft.filters import MetadataFilter
except ImportError:
    MetadataFilter = None

class VectorStore(BaseModel, ABC, Generic[D]):
    """
    Abstract base class for vector store implementations.
    
    This class defines the interface that all vector store implementations must follow.
    The vector store now handles embeddings internally.
    """
    
    _embeddings: Any = PrivateAttr(default=None)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text using the vector store's embeddings model.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model not configured for this vector store")
        return await self._embeddings.embed_query(text)
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple document texts using the vector store's embeddings model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model not configured for this vector store")
        return await self._embeddings.embed_documents(texts)
    
    @abstractmethod
    async def find(self, key: str, value: str):
        """
        Find documents by a specific key-value pair.

        Args:
            key: The key to search by
            value: The value to search for

        Returns:
            List of documents that match the search criteria
        """
        pass

    @abstractmethod
    async def insert_nodes(self, documents: List[D], index_id: Optional[str] = None, show_progress: bool = False) -> List[str]:
        """
        Add documents to the vector store.
        
        Automatically generates embeddings for documents that don't have them.
        
        Args:
            documents: List of document objects to add
            index_id: Optional index identifier to isolate documents
            
        Returns:
            List of document IDs that were added
            :param show_progress:
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_embedding: List[float],
        k: int = 4,
        index_id: Optional[str] = None,
        query_text: Optional[str] = None,
        filters: Optional[Union['MetadataFilter', Dict[str, Any]]] = None,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            index_id: Optional index identifier to filter search results
            query_text: Original query text (required for hybrid search)
            filters: Optional metadata filters to apply
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        pass
    
    async def search_by_text(
        self,
        query: str,
        k: int = 4,
        index_id: Optional[str] = None,
        filters: Optional[Union['MetadataFilter', Dict[str, Any]]] = None,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a text query.
        Generates the query embedding automatically.
        
        Args:
            query: The query text
            k: Number of results to return
            index_id: Optional index identifier to filter search results
            filters: Optional metadata filters to apply
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Generate query embedding
        query_embedding = await self.embed_query(query)
        
        # Perform search, passing query text for hybrid search support
        return await self.similarity_search(
            query_embedding=query_embedding,
            k=k,
            index_id=index_id,
            query_text=query,
            filters=filters,
            **kwargs
        )
    
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
    async def get_node(self, document_id: str, index_id: Optional[str] = None) -> Optional[D]:
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
