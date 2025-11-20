from typing import List, TypeVar, Optional, Annotated, Any
from uuid import uuid4

from pydantic import Field, SkipValidation
from tqdm import tqdm

from fetchcraft.document_store import DocumentStore
from fetchcraft.index.base import BaseIndex
from fetchcraft.node import Node, Chunk, ObjectMapper, DocumentNode
from fetchcraft.retriever import Retriever
from fetchcraft.vector_store.base import VectorStore

D = TypeVar('D', bound=Node)

class VectorIndex(BaseIndex[D]):
    """
    A vector index that works with any vector store implementation.
    
    This class provides a higher-level interface for working with vector stores,
    making it easier to perform common operations like adding documents,
    searching, and managing the index.
    
    The vector store handles embedding generation automatically when adding documents.
    Multiple indices can coexist in the same vector store by using unique index_id values.
    """
    
    vector_store: Annotated[VectorStore[D], SkipValidation()] = Field(description="Vector store instance")

    def __init__(
        self, 
        vector_store: VectorStore[D],
        doc_store: Optional[DocumentStore] = None,
        index_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the vector index with a vector store.
        
        Args:
            vector_store: An instance of a VectorStore implementation (with embeddings configured)
            index_id: Unique identifier for this index. If None, a UUID will be generated.
                     Multiple indices can share the same vector store with different index_ids.
        """
        super().__init__(
            vector_store=vector_store,
            doc_store=doc_store,
            index_id=index_id or str(uuid4()),
            **kwargs
        )

    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore[D],
        doc_store: Optional[DocumentStore] = None,
        index_id: Optional[str] = None,
    ) -> 'VectorIndex[D]':
        """
        Create a VectorIndex from an existing vector store.
        
        Args:
            vector_store: An instance of a VectorStore implementation (with embeddings configured)
            index_id: Unique identifier for this index
            
        Returns:
            A new VectorIndex instance
        """
        return cls(vector_store=vector_store, doc_store=doc_store, index_id=index_id)

    async def delete_document_nodes(self, doc: D):
        """
        Delete all nodes associated with a document.

        Args:
            doc_id: The ID of the document to delete nodes for
        """
        documents = await self._doc_store.list_documents(filters={"id": doc.id})
        if documents:
            document = documents[0]
            children = document.children_ids
            await self.vector_store.delete(children, index_id=self.index_id)


    async def add_nodes(self, doc: Optional[D], nodes: List[D], show_progress: bool = False) -> List[str]:
        """
        Add documents to the index.
        
        The vector store automatically generates embeddings for documents that don't have them.
        
        Args:
            nodes: List of document objects to add
            show_progress: If True, show a progress bar
            
        Returns:
            List of document IDs that were added
            :param nodes: The nodes to add
            :param doc: The source document
        """
        existing_docs = {}
        if self._doc_store:
            for node in nodes:
                _docs = await self._doc_store.list_documents(filters={"metadata.source": node.metadata.get("source")})
                existing_docs.update({doc.id: doc for doc in _docs})

        existing_nodes = list(existing_docs.keys()) if len(existing_docs) > 0 else []
        for existing_id, existing_doc in existing_docs.items():
            existing_nodes.extend(existing_doc.children_ids)

        if len(existing_nodes) > 0:
            await self.vector_store.delete(existing_nodes, index_id=self.index_id)
            if self._doc_store:
                await self._doc_store.delete_documents(existing_nodes)

        if self._doc_store:
            await self._doc_store.add_documents([doc] + nodes)

        return await self.vector_store.insert_nodes(nodes, index_id=self.index_id, show_progress=show_progress)


    async def search(
        self,
        query_embedding: List[float],
        k: int = 4,
        resolve_parents: bool = True,
        query_text: Optional[str] = None,
        **kwargs
    ) -> List[tuple[D, float]]:
        """
        Search for similar documents using a query embedding.
        Only searches within this index's documents.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            resolve_parents: If True, automatically resolve parent nodes for SymNode results
            query_text: Original query text (required for hybrid search)
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            index_id=self.index_id,
            query_text=query_text,
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
        Automatically generates the query embedding using the vector store's embeddings.
        
        Args:
            query: The query text
            k: Number of results to return
            resolve_parents: If True, automatically resolve parent nodes for SymNode results
            **kwargs: Additional search parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Generate query embedding using vector store's embeddings
        query_embedding = await self.vector_store.embed_query(query)
        
        # Perform search with the generated embedding, passing query text for hybrid search
        return await self.search(
            query_embedding=query_embedding,
            k=k,
            resolve_parents=resolve_parents,
            query_text=query,  # Pass query text for hybrid search support
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
        return await self.vector_store.get_node(document_id, index_id=self.index_id)
    
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
        object_mapper: Optional[ObjectMapper] = None,
        filters: Optional[Any] = None,
        **search_kwargs
    ) -> Retriever[D]:
        """
        Create a retriever from this index.
        
        Uses the vector store's embeddings model for query encoding.
        
        Args:
            top_k: Number of results to return (default: 4)
            resolve_parents: Whether to resolve parent nodes for SymNodes (default: True)
            object_mapper: Optional object mapper for resolving ObjectNodes
            filters: Default metadata filters to apply to all queries
            **search_kwargs: Additional keyword arguments to pass to search
            
        Returns:
            A VectorIndexRetriever instance
            
        Example:
            ```python
            from fetchcraft import eq, and_, gte
            
            # Retriever with default filters
            retriever = index.as_retriever(
                top_k=5,
                filters=eq("category", "tutorial")
            )
            
            # All queries will use the default filter
            results = retriever.retrieve("machine learning")
            
            # Can override filters per query
            results = retriever.retrieve("ML", filters=gte("year", 2024))
            ```
        """
        from ..retriever import VectorIndexRetriever
        
        return VectorIndexRetriever(
            vector_index=self,
            top_k=top_k,
            resolve_parents=resolve_parents,
            object_mapper=object_mapper,
            filters=filters,
            **search_kwargs
        )
