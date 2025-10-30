from typing import List, Dict, Any, Optional, Type, Union, Literal

from pydantic import BaseModel, Field, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from .base import VectorStore, D
from ..embeddings import Embeddings
from ..node import Node, DocumentNode, Chunk, SymNode, ObjectNode

try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    SparseTextEmbedding = None

class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "documents"
    distance: str = "Cosine"  # Can be "Cosine", "Euclid", or "Dot"
    enable_hybrid: bool = False  # Enable hybrid search with sparse + dense vectors
    fusion_method: Literal["rrf", "dbsf"] = "rrf"  # Fusion method for hybrid search

class QdrantVectorStore(VectorStore[Node]):
    """
    Qdrant implementation of the VectorStore interface.
    """
    
    client: Any = Field(description="QdrantClient instance")
    collection_name: str = Field(description="Name of the collection")
    document_class: Optional[Type[Node]] = Field(default=None, description="Document class type")
    distance: str = Field(default="Cosine", description="Distance metric (Cosine, Euclid, or Dot)")
    enable_hybrid: bool = Field(default=False, description="Enable hybrid search with sparse + dense vectors")
    fusion_method: Literal["rrf", "dbsf", "mmr"] = Field(default="rrf", description="Fusion method for hybrid search")
    _distance_metric: Any = None  # Computed field for models.Distance
    _sparse_embedder: Any = None  # Fastembed sparse embedder
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Optional[Embeddings] = None,
        document_class: Optional[Type[Node]] = None,
        distance: str = "Cosine",
        enable_hybrid: bool = False,
        fusion_method: Literal["rrf", "dbsf"] = "rrf",
        sparse_model: str = "Qdrant/bm25",
        **kwargs
    ):
        """
        Initialize the Qdrant vector store.
        
        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            embeddings: Embeddings model for generating document embeddings
            document_class: The document model class (defaults to Node if not provided)
            distance: Distance metric to use ("Cosine", "Euclid", or "Dot")
            enable_hybrid: Enable hybrid search with sparse + dense vectors
            fusion_method: Fusion method for hybrid search ("rrf" or "dbsf")
            sparse_model: Model name for sparse embeddings (default: "Qdrant/bm25")
        """
        super().__init__(
            client=client,
            collection_name=collection_name,
            document_class=document_class or Node,  # type: ignore
            distance=distance,
            enable_hybrid=enable_hybrid,
            fusion_method=fusion_method,
            **kwargs
        )
        self._embeddings = embeddings
        self._distance_metric = getattr(models.Distance, distance.upper())
        
        # Initialize sparse embedder if hybrid search is enabled
        if enable_hybrid:
            if not FASTEMBED_AVAILABLE:
                raise ImportError(
                    "fastembed is required for hybrid search. "
                    "Install it with: pip install fastembed"
                )
            self._sparse_embedder = SparseTextEmbedding(model_name=sparse_model)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create it if it doesn't."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            if self.enable_hybrid:
                # Create collection with named dense vectors and sparse vectors for hybrid search
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self._embeddings.dimension,
                            distance=self._distance_metric
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF  # Use IDF modifier for BM25-style scoring
                        )
                    }
                )
            else:
                # Create collection with single vector for standard search
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embeddings.dimension,
                        distance=self._distance_metric
                    )
                )
    
    def _get_doc_class(self, class_name: Optional[str]) -> Type[Node]:
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
        elif class_name == 'DocumentNode':
            return DocumentNode  # type: ignore
        elif class_name == 'Node':
            return Node  # type: ignore
        elif class_name == 'ObjectNode':
            return ObjectNode  # type: ignore
        else:
            # Fall back to the default document class
            return self.document_class
    
    async def insert_nodes(self, nodes: List[Node], index_id: Optional[str] = None, show_progress: bool = False) -> List[str]:
        """
        Add documents to the Qdrant collection.
        
        Automatically generates embeddings for documents that don't have them.
        Checks for existing documents and only updates if content has changed (based on hash).
        
        Args:
            nodes: List of node objects (with or without embeddings)
            index_id: Optional index identifier to isolate documents
            show_progress: Whether to show progress bar
            
        Returns:
            List of document IDs that were added or updated
        """
        if not nodes:
            return []
        
        # Single loop: check hash, generate embeddings if needed, upsert if changed
        ids = []
        if show_progress:
            nodes = tqdm(nodes, desc="Processing documents")

        for node in nodes:
            # Generate dense embedding if needed
            if not node.embedding:  # type: ignore
                if self._embeddings is None:
                    raise ValueError("Document missing embedding and no embeddings model configured")
                # Embed this single document
                embedding = await self._embeddings.embed_documents([node.text])  # type: ignore
                node.embedding = embedding[0]  # type: ignore
            
            # Generate sparse embedding if hybrid mode
            sparse_vector = None
            if self.enable_hybrid:
                # Generate sparse embedding for this document
                sparse_emb = next(self._sparse_embedder.embed([node.text]))  # type: ignore
                sparse_vector = models.SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                )
            
            # Extract vector and payload
            payload = node.model_dump()
            dense_vector = payload.pop('embedding')
            
            # Store the document ID in the payload
            payload['id'] = node.id
            
            # Store the document class type for proper reconstruction
            payload['_doc_class'] = node.__class__.__name__
            
            # Add index_id to payload if provided
            if index_id is not None:
                payload['_index_id'] = index_id
            
            # Prepare vectors based on hybrid mode
            if self.enable_hybrid:
                vector = {
                    "dense": dense_vector,
                    "sparse": sparse_vector
                }
            else:
                vector = dense_vector
            
            # Create point and upsert (insert or update)
            point = models.PointStruct(
                id=node.id,
                vector=vector,
                payload=payload
            )
            
            # Upsert into index (insert if new, update if exists)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            ids.append(node.id)
        
        return ids
    
    async def similarity_search(
        self, 
        query_embedding: List[float],
        k: int = 4,
        index_id: Optional[str] = None,
        query_text: Optional[str] = None,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            index_id: Optional index identifier to filter search results
            query_text: Original query text (required for hybrid search)
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
        
        # Perform hybrid search if enabled
        if self.enable_hybrid:
            if query_text is None:
                raise ValueError("query_text is required for hybrid search")
            
            # Generate sparse embedding for query
            sparse_query = list(self._sparse_embedder.embed([query_text]))[0]
            
            # Use prefetch + fusion query for hybrid search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query.indices.tolist(),
                            values=sparse_query.values.tolist()
                        ),
                        using="sparse",
                        limit=k * 2,  # Fetch more candidates for fusion
                    ),
                    models.Prefetch(
                        query=query_embedding,
                        using="dense",
                        limit=k * 2,  # Fetch more candidates for fusion
                    ),
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF if self.fusion_method == "rrf" else models.Fusion.DBSF
                ),
                limit=k,
                query_filter=query_filter,
                with_vectors=True,
                **kwargs
            )
        else:
            # Standard dense vector search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                query_filter=query_filter,
                with_vectors=True,  # Include vectors in results
                **kwargs
            )
        
        results = []
        for hit in search_result.points:
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
                    # For hybrid mode, store the dense vector
                    if self.enable_hybrid and isinstance(hit.vector, dict):
                        doc_dict['embedding'] = hit.vector.get('dense')
                    else:
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
    
    async def get_node(self, node_id: str, index_id: Optional[str] = None) -> Optional[Node]:
        """
        Retrieve a single document by its ID.
        
        Args:
            node_id: The ID of the document to retrieve
            index_id: Optional index identifier to filter retrieval
            
        Returns:
            The document if found, None otherwise
        """
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[node_id],
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
                # For hybrid mode, store the dense vector
                if self.enable_hybrid and isinstance(result[0].vector, dict):
                    doc_data['embedding'] = result[0].vector.get('dense')
                else:
                    doc_data['embedding'] = result[0].vector
        
        # Reconstruct using the correct class type
        doc_class = self._get_doc_class(doc_class_name)
        return doc_class(**doc_data)
    
    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], QdrantConfig], embeddings: Optional[Any] = None) -> 'QdrantVectorStore':
        """
        Create a QdrantVectorStore instance from a configuration.
        
        Args:
            config: Either a QdrantConfig or a dictionary with configuration
            embeddings: Optional embeddings model for generating document embeddings
            
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
            embeddings=embeddings,
            document_class=Node,  # Defaults to Node
            distance=config.distance,
            enable_hybrid=config.enable_hybrid,
            fusion_method=config.fusion_method
        )
