from typing import List, Dict, Any, Optional, Type, TypeVar, Generic, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, Field

from .base import VectorStore, D

class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "documents"
    vector_size: int = 384  # Default to a common embedding size
    distance: str = "Cosine"  # Can be "Cosine", "Euclid", or "Dot"

class QdrantVectorStore(VectorStore[D]):
    """
    Qdrant implementation of the VectorStore interface.
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        document_class: Type[D],
        vector_size: int = 384,
        distance: str = "Cosine"
    ):
        """
        Initialize the Qdrant vector store.
        
        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            document_class: The document model class
            vector_size: Size of the embedding vectors
            distance: Distance metric to use ("Cosine", "Euclid", or "Dot")
        """
        self.client = client
        self.collection_name = collection_name
        self.document_class = document_class
        self.vector_size = vector_size
        self.distance = getattr(models.Distance, distance.upper())
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create it if it doesn't."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
    
    async def add_documents(self, documents: List[D]) -> List[str]:
        """
        Add documents to the Qdrant collection.
        
        Args:
            documents: List of document objects with embeddings
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        # Generate IDs for new documents
        ids = [str(uuid4()) for _ in documents]
        
        # Convert documents to Qdrant points
        points = []
        for doc_id, doc in zip(ids, documents):
            # Extract vector and payload
            if not hasattr(doc, 'embedding') or not doc.embedding:  # type: ignore
                raise ValueError("Document must have an 'embedding' field")
                
            payload = doc.dict()
            vector = payload.pop('embedding')
            
            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    async def similarity_search(
        self, 
        query_embedding: List[float],
        k: int = 4,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            **kwargs
        )
        
        results = []
        for hit in search_result:
            doc_dict = hit.payload
            doc_dict['id'] = hit.id  # Include the document ID
            doc = self.document_class(**doc_dict)
            results.append((doc, hit.score))
            
        return results
    
    async def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
        return True
    
    async def get_document(self, document_id: str) -> Optional[D]:
        """
        Retrieve a single document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[document_id]
        )
        
        if not result:
            return None
            
        doc_data = result[0].payload
        doc_data['id'] = result[0].id
        return self.document_class(**doc_data)
    
    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], QdrantConfig]) -> 'QdrantVectorStore':
        """
        Create a QdrantVectorStore instance from a configuration.
        
        Args:
            config: Either a QdrantConfig or a dictionary with configuration
            
        Returns:
            An instance of QdrantVectorStore
        """
        if not isinstance(config, QdrantConfig):
            config = QdrantConfig(**config)
            
        client = QdrantClient(
            url=config.url,
            api_key=config.api_key
        )
        
        return cls(
            client=client,
            collection_name=config.collection_name,
            document_class=BaseModel,  # This should be overridden by the actual document class
            vector_size=config.vector_size,
            distance=config.distance
        )
