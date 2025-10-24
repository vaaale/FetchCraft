from typing import List, TypeVar, Generic, Type, Optional, Dict, Any
from abc import ABC, abstractmethod
from uuid import uuid4

from pydantic import BaseModel
from .vector_store.base import VectorStore

D = TypeVar('D', bound=BaseModel)

class VectorIndex(Generic[D]):
    """
    A vector index that works with any vector store implementation.
    
    This class provides a higher-level interface for working with vector stores,
    making it easier to perform common operations like adding documents,
    searching, and managing the index.
    
    Multiple indices can coexist in the same vector store by using unique index_id values.
    """
    
    def __init__(self, vector_store: VectorStore[D], index_id: Optional[str] = None):
        """
        Initialize the vector index with a vector store.
        
        Args:
            vector_store: An instance of a VectorStore implementation
            index_id: Unique identifier for this index. If None, a UUID will be generated.
                     Multiple indices can share the same vector store with different index_ids.
        """
        self.vector_store = vector_store
        self.index_id = index_id or str(uuid4())
    
    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore[D],
        index_id: Optional[str] = None,
    ) -> 'VectorIndex[D]':
        """
        Create a VectorIndex from an existing vector store.
        
        Args:
            vector_store: An instance of a VectorStore implementation
            index_id: Unique identifier for this index
            
        Returns:
            A new VectorIndex instance
        """
        return cls(vector_store=vector_store, index_id=index_id)
    
    async def add_documents(self, documents: List[D]) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            documents: List of document objects to add
            
        Returns:
            List of document IDs that were added
        """
        return await self.vector_store.add_documents(documents, index_id=self.index_id)
    
    async def search(
        self,
        query_embedding: List[float],
        k: int = 4,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        Only searches within this index's documents.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        return await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            index_id=self.index_id,
            **kwargs
        )
    
    async def get_document(self, document_id: str) -> Optional[D]:
        """
        Retrieve a single document by its ID.
        Only retrieves if the document belongs to this index.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        return await self.vector_store.get_document(document_id, index_id=self.index_id)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        Only deletes documents that belong to this index.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        return await self.vector_store.delete(document_ids, index_id=self.index_id)
    
    @property
    def store(self) -> VectorStore[D]:
        """
        Get the underlying vector store instance.
        
        Returns:
            The vector store instance
        """
        return self.vector_store
