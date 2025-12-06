from abc import ABCMeta, abstractmethod
from typing import List, Set, Generic, TypeVar, Optional, Type
from uuid import uuid4

from pydantic import Field, ConfigDict, BaseModel, PrivateAttr

from fetchcraft.document_store import DocumentStore
from fetchcraft.node import Chunk, Node, NodeType, ObjectMapper

D = TypeVar('D', bound=Node)


class BaseIndex(BaseModel, Generic[D], metaclass=ABCMeta):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    index_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique index identifier")
    _object_mapper: Optional[ObjectMapper] = PrivateAttr()
    _doc_store: Optional[DocumentStore[Node]] = PrivateAttr()

    def __init__(self, doc_store: Optional[DocumentStore] = None, **kwargs):
        super().__init__(**kwargs)
        self._doc_store = doc_store

    @abstractmethod
    def add_nodes(self, doc: Optional[D], nodes: List[D], show_progress: bool = False):
        """
        Add documents to the index.

        Automatically generates embeddings for documents that don't have them.

        Args:
            documents: List of document objects to add
            auto_embed: If True, automatically generate embeddings for documents without them

        Returns:
            List of document IDs that were added
            :param doc:
            :param nodes: The nodes to add
            :param show_progress: show progress
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

    async def _resolve_to_top_parent(
        self,
        node: D,
        max_depth: int = 10
    ) -> Optional[D]:
        """
        Recursively resolve a node to its top-level parent.

        Args:
            node: The node to resolve
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            The top-level parent node, or None if not found
        """
        current_node = node
        depth = 0

        while depth < max_depth:
            # If current node is a SymNode with a parent, get its parent
            if current_node.node_type == NodeType.SYMNODE and current_node.parent_id:
                parent_id = current_node.parent_id

                # Fetch the parent node
                parent_node = await self.vector_store.get_node(parent_id, index_id=None)

                if not parent_node:
                    # Parent not found, return current node
                    return current_node

                # Continue with the parent
                current_node = parent_node  # type: ignore
                depth += 1
            else:
                # Reached a non-SymNode
                return current_node

        # Max depth reached, return current node
        return current_node

    async def _resolve_parent_nodes(
        self,
        results: List[tuple[D, float]]
    ) -> List[tuple[D, float]]:
        """
        Resolve parent nodes for SymNode instances in search results.
        Recursively resolves to the top-level parent if the parent is also a SymNode.

        Args:
            results: List of (document, score) tuples from search

        Returns:
            List of (document, score) tuples with parent nodes resolved to top level
        """
        resolved_results = []
        seen_parent_ids: Set[str] = set()

        for doc, score in results:
            # Check if this is a SymNode that needs parent resolution
            if doc.node_type == NodeType.SYMNODE and doc.parent_id:
                # Recursively resolve to top-level parent
                top_parent = await self._resolve_to_top_parent(doc)

                if top_parent and top_parent.id not in seen_parent_ids:
                    resolved_results.append((top_parent, score))
                    seen_parent_ids.add(top_parent.id)
                elif not top_parent:
                    # Resolution failed, fall back to original node
                    if doc.id not in seen_parent_ids:
                        resolved_results.append((doc, score))
                        seen_parent_ids.add(doc.id)
            else:
                # Not a SymNode
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
    def as_retriever(self, top_k, resolve_parents, object_mapper: Optional[ObjectMapper] = None, **search_kwargs):
        """
        Create a retriever from this index.

        Uses the index's embeddings model for query encoding.

        Args:
            top_k: Number of results to return (default: 4)
            resolve_parents: Whether to resolve parent nodes for SymNodes (default: True)
            **search_kwargs: Additional keyword arguments to pass to search

        Returns:
            A VectorIndexRetriever instance
            :param top_k: Number of results to return (default: 4)
            :param resolve_parents: Whether to resolve parent nodes for SymNodes (default: True)
            :param object_mapper: ObjectMapper to use for object resolution
        """
        pass


class IndexFactory(BaseModel):
    options: dict = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs

    @abstractmethod
    def create_index(self, index_cls: Type[BaseIndex], **kwargs) -> BaseIndex:
        pass

