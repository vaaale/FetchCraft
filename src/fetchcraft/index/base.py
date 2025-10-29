from abc import ABCMeta, abstractmethod
from typing import List, Set, Generic, TypeVar, Optional
from uuid import uuid4

from pydantic import Field, ConfigDict, BaseModel, PrivateAttr

from fetchcraft.document_store import DocumentStore
from fetchcraft.node import SymNode, Chunk, Node

D = TypeVar('D', bound=Node)


class BaseIndex(BaseModel, Generic[D], metaclass=ABCMeta):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    index_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique index identifier")
    _doc_store: Optional[DocumentStore[Node]] = PrivateAttr()

    def __init__(self, doc_store: Optional[DocumentStore] = None, **kwargs):
        super().__init__(**kwargs)
        self._doc_store = doc_store


    @abstractmethod
    def add_nodes(self, documents, show_progress):
        """
        Add documents to the index.

        Automatically generates embeddings for documents that don't have them.

        Args:
            documents: List of document objects to add
            auto_embed: If True, automatically generate embeddings for documents without them

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    def search(self, query_embedding, k, resolve_parents, kwargs):
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
        pass

    @abstractmethod
    def search_by_text(self, query, k, resolve_parents, kwargs):
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
        pass

    async def _resolve_parent_nodes(
        self,
        results: List[tuple[D, float]]
    ) -> List[tuple[D, float]]:
        """
        Resolve parent nodes for SymNode instances in search results.

        Args:
            results: List of (document, score) tuples from search

        Returns:
            List of (document, score) tuples with parent nodes resolved
        """
        resolved_results = []
        seen_parent_ids: Set[str] = set()

        for doc, score in results:
            # Check if this is a SymNode that requires parent resolution
            if isinstance(doc, SymNode) and doc.requires_parent_resolution():
                parent_id = doc.parent_id

                # Skip if we've already resolved this parent
                if parent_id in seen_parent_ids:
                    continue

                # Fetch the parent node
                # Note: parent might be in a different index, so we don't pass index_id
                parent_node = await self.vector_store.get_node(parent_id, index_id=None)

                if parent_node:
                    resolved_results.append((parent_node, score))  # type: ignore
                    seen_parent_ids.add(parent_id)
                else:
                    # If parent not found, fall back to the SymNode itself
                    resolved_results.append((doc, score))
            else:
                # Not a SymNode or doesn't require resolution
                # Check if this document is already a parent we've seen
                if doc.id not in seen_parent_ids:
                    resolved_results.append((doc, score))
                    # If this could be a parent (Chunk), track it to avoid duplicates
                    if isinstance(doc, Chunk):
                        seen_parent_ids.add(doc.id)

        return resolved_results

    @abstractmethod
    def get_document(self, document_id):
        """
        Retrieve a single document by its ID.
        Only retrieves if the document belongs to this index.

        Args:
            document_id: The ID of the document to retrieve

        Returns:
            The document if found, None otherwise
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids):
        """
        Delete documents by their IDs.
        Only deletes documents that belong to this index.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def as_retriever(self, top_k, resolve_parents, search_kwargs):
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
        pass