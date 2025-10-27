from typing import List, TypeVar, Optional, Annotated
from uuid import uuid4

from pydantic import Field, SkipValidation, PrivateAttr
from tqdm import tqdm

from fetchcraft.embeddings.base import Embeddings
from fetchcraft.index.base import BaseIndex
from fetchcraft.node import Node
from fetchcraft.vector_store.base import VectorStore

D = TypeVar('D', bound=Node)

class VectorIndex(BaseIndex[D]):
    """
    A vector index that works with any vector store implementation.
    
    This class provides a higher-level interface for working with vector stores,
    making it easier to perform common operations like adding documents,
    searching, and managing the index.
    
    The index handles embedding generation automatically when adding documents.
    Multiple indices can coexist in the same vector store by using unique index_id values.
    """
    
    vector_store: Annotated[VectorStore[D], SkipValidation()] = Field(description="Vector store instance")
    _embeddings: Embeddings = PrivateAttr(default=None)

    def __init__(
        self, 
        vector_store: VectorStore[D], 
        embeddings: Embeddings,
        index_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the vector index with a vector store and embeddings model.
        
        Args:
            vector_store: An instance of a VectorStore implementation
            embeddings: An embeddings model for generating document embeddings
            index_id: Unique identifier for this index. If None, a UUID will be generated.
                     Multiple indices can share the same vector store with different index_ids.
        """
        super().__init__(
            vector_store=vector_store,
            index_id=index_id or str(uuid4()),
            **kwargs
        )
        self._embeddings=embeddings

    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore[D],
        embeddings: Embeddings,
        index_id: Optional[str] = None,
    ) -> 'VectorIndex[D]':
        """
        Create a VectorIndex from an existing vector store.
        
        Args:
            vector_store: An instance of a VectorStore implementation
            embeddings: An embeddings model for generating document embeddings
            index_id: Unique identifier for this index
            
        Returns:
            A new VectorIndex instance
        """
        return cls(vector_store=vector_store, embeddings=embeddings, index_id=index_id)
    
    async def add_documents(self, documents: List[D], show_progress: bool = False) -> List[str]:
        """
        Add documents to the index.
        
        Automatically generates embeddings for documents that don't have them.
        
        Args:
            documents: List of document objects to add
            auto_embed: If True, automatically generate embeddings for documents without them
            
        Returns:
            List of document IDs that were added
        """

        # Generate embeddings for documents that don't have them
        ids = []
        if show_progress:
            documents = tqdm(documents, desc="Adding documents")

        for i, doc in enumerate(documents):
            if doc.embedding is None:
                embeddings = await self._embeddings.embed_documents([doc.text])
                doc.embedding = embeddings[0]
                added_ids = await self.vector_store.add_documents([doc], index_id=self.index_id)
                ids.append(added_ids[0])
            else:
                added_ids = await self.vector_store.add_documents([doc], index_id=self.index_id)
                ids.append(added_ids[0])
        return ids


    async def search(
        self,
        query_embedding: List[float],
        k: int = 4,
        resolve_parents: bool = True,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        Only searches within this index's documents.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            resolve_parents: If True, automatically resolve parent nodes for SymNode results
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            index_id=self.index_id,
            **kwargs
        )
        
        # Resolve parent nodes if enabled
        if resolve_parents:
            results = await self._resolve_parent_nodes(results)
        
        return results
    
    async def search_by_text(
        self,
        query: str,
        k: int = 4,
        resolve_parents: bool = True,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a text query.
        Automatically generates the query embedding.
        
        Args:
            query: The query text
            k: Number of results to return
            resolve_parents: If True, automatically resolve parent nodes for SymNode results
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Generate query embedding
        query_embedding = await self._embeddings.embed_query(query)
        
        # Perform search with the generated embedding
        return await self.search(
            query_embedding=query_embedding,
            k=k,
            resolve_parents=resolve_parents,
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
    
    def as_retriever(
        self,
        top_k: int = 4,
        resolve_parents: bool = True,
        **search_kwargs
    ) -> 'VectorIndexRetriever[D]':
        """
        Create a retriever from this index.
        
        Uses the index's embeddings model for query encoding.
        
        Args:
            top_k: Number of results to return (default: 4)
            resolve_parents: Whether to resolve parent nodes for SymNodes (default: True)
            **search_kwargs: Additional keyword arguments to pass to search
            
        Returns:
            A VectorIndexRetriever instance
        """
        from ..retriever import VectorIndexRetriever
        
        return VectorIndexRetriever(
            vector_index=self,
            embeddings=self._embeddings,
            top_k=top_k,
            resolve_parents=resolve_parents,
            **search_kwargs
        )
