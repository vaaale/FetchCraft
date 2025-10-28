"""
ChromaDB vector store implementation.
"""
from typing import List, Dict, Any, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .base import VectorStore, D
from ..node import Node, Chunk, SymNode


try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaConfig(BaseModel):
    """Configuration for Chroma vector store."""
    collection_name: str = "documents"
    persist_directory: Optional[str] = None
    distance: str = "cosine"  # Can be "cosine", "l2", or "ip" (inner product)


class ChromaVectorStore(VectorStore[D]):
    """
    ChromaDB implementation of the VectorStore interface.
    
    This implementation uses ChromaDB as the underlying vector database.
    It supports both persistent and in-memory modes.
    """
    
    client: Any = Field(description="ChromaDB client instance")
    collection_name: str = Field(description="Name of the collection")
    document_class: Optional[Type[D]] = Field(default=None, description="Document class type")
    distance: str = Field(default="cosine", description="Distance metric (cosine, l2, or ip)")
    _collection: Any = None  # ChromaDB collection instance
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    def __init__(
        self,
        client: Any,  # chromadb.Client
        collection_name: str,
        embeddings: Optional[Any] = None,
        document_class: Optional[Type[D]] = None,
        distance: str = "cosine",
        **kwargs
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            client: ChromaDB Client instance
            collection_name: Name of the collection to use
            embeddings: Embeddings model for generating document embeddings
            document_class: The document model class (defaults to Node if not provided)
            distance: Distance metric to use ("cosine", "l2", or "ip")
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        super().__init__(
            client=client,
            collection_name=collection_name,
            document_class=document_class or Node,  # type: ignore
            distance=distance,
            **kwargs
        )
        self._embeddings = embeddings
        
        # Get or create collection
        self._collection = self._ensure_collection()
    
    def _ensure_collection(self) -> Any:
        """Ensure the collection exists, create it if it doesn't."""
        # Map distance metric names
        distance_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
            "dot": "ip",  # Alias for inner product
        }
        
        distance_metric = distance_map.get(self.distance.lower(), "cosine")
        
        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance_metric}
        )
        
        return collection
    
    def _get_doc_class(self, class_name: Optional[str]) -> Type[D]:
        """
        Get the document class based on the stored class name.
        
        Args:
            class_name: Name of the class stored in the metadata
            
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
    
    async def add_documents(self, documents: List[D], index_id: Optional[str] = None, show_progress: bool = False) -> List[str]:
        """
        Add documents to the Chroma collection.
        
        Automatically generates embeddings for documents that don't have them.
        Checks for existing documents and only updates if content has changed (based on hash).
        
        Args:
            documents: List of document objects (with or without embeddings)
            index_id: Optional index identifier to isolate documents
            show_progress: Whether to show progress bar
            
        Returns:
            List of document IDs that were added or updated
        """
        if not documents:
            return []
        
        # Prepare data for Chroma
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            # Get or generate document ID
            if hasattr(doc, 'id') and doc.id:  # type: ignore
                doc_id = doc.id  # type: ignore
            else:
                doc_id = str(uuid4())
            
            # Check if document already exists (hash is computed automatically via property)
            try:
                existing_doc = await self.get_document(doc_id, index_id=index_id)
                if existing_doc and hasattr(existing_doc, 'hash') and hasattr(doc, 'hash'):  # type: ignore
                    if existing_doc.hash == doc.hash:  # type: ignore
                        # Document hasn't changed, skip
                        ids.append(doc_id)
                        continue
            except Exception:
                # Document doesn't exist or error retrieving, proceed with insert
                pass
            
            # Generate embedding if needed
            if not hasattr(doc, 'embedding') or not doc.embedding:  # type: ignore
                if self._embeddings is None:
                    raise ValueError("Document missing embedding and no embeddings model configured")
                # Generate embedding
                embedding = await self._embeddings.embed_documents([doc.text])  # type: ignore
                doc.embedding = embedding[0]  # type: ignore
            
            ids.append(doc_id)
            
            # Extract embedding
            if not hasattr(doc, 'embedding') or not doc.embedding:  # type: ignore
                raise ValueError("Document must have an 'embedding' field")
            embeddings.append(doc.embedding)  # type: ignore
            
            # Prepare metadata
            doc_dict = doc.model_dump(exclude={'embedding', 'text'})
            doc_dict['id'] = doc_id
            doc_dict['_doc_class'] = doc.__class__.__name__
            
            # Add index_id to metadata if provided
            if index_id is not None:
                doc_dict['_index_id'] = index_id
            
            # Convert all metadata values to strings (Chroma requirement)
            metadata = {}
            for key, value in doc_dict.items():
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        # Convert complex types to JSON strings
                        import json
                        metadata[key] = json.dumps(value)
            
            metadatas.append(metadata)
            documents_text.append(doc.text if hasattr(doc, 'text') else "")  # type: ignore
        
        # Upsert to Chroma collection (will update if exists)
        if ids:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
        
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
            query_text: Original query text (not used by Chroma, included for compatibility)
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Build where filter for index_id if provided
        where_filter = kwargs.pop('where', None)
        if index_id is not None:
            index_filter = {"_index_id": index_id}
            if where_filter:
                # Merge filters using $and
                where_filter = {"$and": [where_filter, index_filter]}
            else:
                where_filter = index_filter
        
        # Query Chroma
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["embeddings", "metadatas", "documents", "distances"],
            **kwargs
        )
        
        # Parse results
        output = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i] if results.get('metadatas') is not None and len(results['metadatas']) > 0 else {}
                embedding = results['embeddings'][0][i] if results.get('embeddings') is not None and len(results['embeddings']) > 0 else None
                text = results['documents'][0][i] if results.get('documents') is not None and len(results['documents']) > 0 else ""
                distance = results['distances'][0][i] if results.get('distances') is not None and len(results['distances']) > 0 else 0.0
                
                # Convert distance to similarity score (1 - distance for cosine)
                # Chroma returns distances, we want similarity scores
                if self.distance.lower() == "cosine":
                    score = 1.0 - distance
                elif self.distance.lower() == "l2":
                    # For L2, smaller is better, convert to similarity
                    score = 1.0 / (1.0 + distance)
                else:  # ip (inner product)
                    score = distance  # Inner product is already a similarity
                
                # Reconstruct document
                doc_dict = {'text': text}
                
                # Parse metadata back to proper types
                import json
                for key, value in metadata.items():
                    if key.startswith('_'):
                        continue  # Skip internal fields for now
                    
                    # Try to parse JSON strings back to objects
                    if isinstance(value, str):
                        try:
                            doc_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            doc_dict[key] = value
                    else:
                        doc_dict[key] = value
                
                # Add back essential fields
                doc_dict['id'] = metadata.get('id', doc_id)
                if embedding is not None:
                    doc_dict['embedding'] = embedding
                
                # Get document class
                doc_class_name = metadata.get('_doc_class')
                doc_class = self._get_doc_class(doc_class_name)
                
                # Create document instance
                doc = doc_class(**doc_dict)
                output.append((doc, score))
        
        return output
    
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
            # First get all documents with the index_id
            results = self._collection.get(
                ids=ids,
                where={"_index_id": index_id}
            )
            if results and results['ids']:
                self._collection.delete(ids=results['ids'])
        else:
            # Delete by ID only
            self._collection.delete(ids=ids)
        
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
        # Get document from Chroma
        result = self._collection.get(
            ids=[document_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not result or not result['ids'] or len(result['ids']) == 0:
            return None
        
        metadata = result['metadatas'][0] if result.get('metadatas') is not None and len(result['metadatas']) > 0 else {}
        
        # Check if index_id matches if specified
        if index_id is not None:
            stored_index_id = metadata.get('_index_id')
            if stored_index_id != index_id:
                return None
        
        # Reconstruct document
        text = result['documents'][0] if result.get('documents') is not None and len(result['documents']) > 0 else ""
        embedding = result['embeddings'][0] if result.get('embeddings') is not None and len(result['embeddings']) > 0 else None
        
        doc_dict = {'text': text}
        
        # Parse metadata back to proper types
        import json
        for key, value in metadata.items():
            if key.startswith('_') and key != '_doc_class':
                continue  # Skip internal fields
            
            # Try to parse JSON strings back to objects
            if isinstance(value, str):
                try:
                    doc_dict[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    doc_dict[key] = value
            else:
                doc_dict[key] = value
        
        # Add back essential fields
        doc_dict['id'] = metadata.get('id', document_id)
        if embedding is not None:
            doc_dict['embedding'] = embedding
        
        # Get document class
        doc_class_name = metadata.get('_doc_class')
        doc_class = self._get_doc_class(doc_class_name)
        
        return doc_class(**doc_dict)
    
    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], ChromaConfig], embeddings: Optional[Any] = None) -> 'ChromaVectorStore':
        """
        Create a ChromaVectorStore instance from a configuration.
        
        Args:
            config: Either a ChromaConfig or a dictionary with configuration
            embeddings: Optional embeddings model for generating document embeddings
            
        Returns:
            An instance of ChromaVectorStore
        """
        if not isinstance(config, ChromaConfig):
            config = ChromaConfig(**config)
        
        # Import chromadb
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        # Create client
        if config.persist_directory:
            client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            client = chromadb.Client()
        
        return cls(
            client=client,
            collection_name=config.collection_name,
            embeddings=embeddings,
            document_class=Node,  # Defaults to Node
            distance=config.distance
        )
