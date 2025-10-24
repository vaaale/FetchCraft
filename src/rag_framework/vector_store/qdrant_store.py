from typing import List, Dict, Any, Optional, Type, TypeVar, Generic, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, Field

from .base import VectorStore, D
from ..node import Node, Chunk, SymNode

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
        document_class: Optional[Type[D]] = None,
        vector_size: int = 384,
        distance: str = "Cosine"
    ):
        """
        Initialize the Qdrant vector store.
        
        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            document_class: The document model class (defaults to Node if not provided)
            vector_size: Size of the embedding vectors
            distance: Distance metric to use ("Cosine", "Euclid", or "Dot")
        """
        self.client = client
        self.collection_name = collection_name
        self.document_class = document_class or Node  # type: ignore
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
    
    def _get_doc_class(self, class_name: Optional[str]) -> Type[D]:
        """
        Get the document class based on the stored class name.
        
        Args:
            class_name: Name of the class stored in the payload
            
        Returns:
            The appropriate document class
        """
        if class_name == 'SymNode':
            return SymNode  # type: ignore
        elif class_name == 'Chunk':
            return Chunk  # type: ignore
        elif class_name == 'Node':
            return Node  # type: ignore
        else:
            # Fall back to the default document class
            return self.document_class
    
    async def add_documents(self, documents: List[D], index_id: Optional[str] = None) -> List[str]:
        """
        Add documents to the Qdrant collection.
        
        Args:
            documents: List of document objects with embeddings
            index_id: Optional index identifier to isolate documents
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        # Use existing document IDs or generate new ones
        ids = []
        for doc in documents:
            if hasattr(doc, 'id') and doc.id:  # type: ignore
                ids.append(doc.id)  # type: ignore
            else:
                ids.append(str(uuid4()))
        
        # Convert documents to Qdrant points
        points = []
        for doc_id, doc in zip(ids, documents):
            # Extract vector and payload
            if not hasattr(doc, 'embedding') or not doc.embedding:  # type: ignore
                raise ValueError("Document must have an 'embedding' field")
                
            payload = doc.dict()
            vector = payload.pop('embedding')
            
            # Store the document ID in the payload as well
            payload['id'] = doc_id
            
            # Store the document class type for proper reconstruction
            payload['_doc_class'] = doc.__class__.__name__
            
            # Add index_id to payload if provided
            if index_id is not None:
                payload['_index_id'] = index_id
            
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
        # Build filter for index_id if provided
        query_filter = kwargs.pop('query_filter', None)
        if index_id is not None:
            index_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="_index_id",
                        match=models.MatchValue(value=index_id)
                    )
                ]
            )
            # Merge with existing filter if provided
            if query_filter:
                if hasattr(query_filter, 'must'):
                    query_filter.must.extend(index_filter.must)
                else:
                    query_filter = index_filter
            else:
                query_filter = index_filter
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=query_filter,
            with_vectors=True,  # Include vectors in results
            **kwargs
        )
        
        results = []
        for hit in search_result:
            doc_dict = hit.payload.copy()
            # Remove internal fields from payload
            doc_dict.pop('_index_id', None)
            doc_class_name = doc_dict.pop('_doc_class', None)
            
            # Ensure document ID is present
            if 'id' not in doc_dict:
                doc_dict['id'] = hit.id
            # Add embedding back if the document class expects it
            if 'embedding' not in doc_dict and hasattr(self.document_class, 'model_fields'):
                if 'embedding' in self.document_class.model_fields:
                    doc_dict['embedding'] = hit.vector
            
            # Reconstruct using the correct class type
            doc_class = self._get_doc_class(doc_class_name)
            doc = doc_class(**doc_dict)
            results.append((doc, hit.score))
            
        return results
    
    async def delete(self, ids: List[str], index_id: Optional[str] = None) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            index_id: Optional index identifier to filter deletions
            
        Returns:
            True if deletion was successful
        """
        if index_id is not None:
            # Filter by both ID and index_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.HasIdCondition(has_id=ids),
                            models.FieldCondition(
                                key="_index_id",
                                match=models.MatchValue(value=index_id)
                            )
                        ]
                    )
                )
            )
        else:
            # Delete by ID only
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
        return True
    
    async def get_document(self, document_id: str, index_id: Optional[str] = None) -> Optional[D]:
        """
        Retrieve a single document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            index_id: Optional index identifier to filter retrieval
            
        Returns:
            The document if found, None otherwise
        """
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[document_id],
            with_vectors=True  # Include vectors in results
        )
        
        if not result:
            return None
        
        # Check if index_id matches if specified
        doc_data = result[0].payload.copy()
        if index_id is not None:
            stored_index_id = doc_data.get('_index_id')
            if stored_index_id != index_id:
                return None
        
        # Remove internal fields from payload
        doc_data.pop('_index_id', None)
        doc_class_name = doc_data.pop('_doc_class', None)
        
        # Ensure document ID is present
        if 'id' not in doc_data:
            doc_data['id'] = result[0].id
        # Add embedding back if the document class expects it
        if 'embedding' not in doc_data and hasattr(self.document_class, 'model_fields'):
            if 'embedding' in self.document_class.model_fields:
                doc_data['embedding'] = result[0].vector
        
        # Reconstruct using the correct class type
        doc_class = self._get_doc_class(doc_class_name)
        return doc_class(**doc_data)
    
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
            document_class=Node,  # Defaults to Node
            vector_size=config.vector_size,
            distance=config.distance
        )
