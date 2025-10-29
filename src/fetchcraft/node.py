from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, computed_field
from uuid import uuid4
import hashlib
import json


class Node(BaseModel):
    """
    Base class for representing nodes in a document hierarchy.
    
    Nodes can have parent-child relationships and can be linked
    in a sequence using next/previous pointers.
    
    Relationships:
    - parent_id/children_ids: Hierarchical relationships (any node can have both)
    - next_id/previous_id: Sequential relationships (sibling ordering)
    
    All relationship methods work with node IDs rather than Node objects.
    This allows for efficient serialization and avoids circular references.
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    # Document reference
    doc_id: Optional[str] = None  # All nodes from same document share this ID
    
    # Hierarchical relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    
    # Sequential relationships (sibling ordering)
    next_id: Optional[str] = None
    previous_id: Optional[str] = None
    
    # Cached hash (not persisted)
    _hash: Optional[str] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    @property
    def parent(self) -> Optional[str]:
        """Get the parent node ID."""
        return self.parent_id
    
    @parent.setter
    def parent(self, node_id: Optional[str]) -> None:
        """Set the parent node ID."""
        self.parent_id = node_id
    
    @property
    def children(self) -> List[str]:
        """Get the child node IDs."""
        return self.children_ids
    
    @children.setter
    def children(self, node_ids: List[str]) -> None:
        """Set the child node IDs."""
        self.children_ids = node_ids
    
    @property
    def next(self) -> Optional[str]:
        """Get the next node ID in the sequence."""
        return self.next_id
    
    @next.setter
    def next(self, node_id: Optional[str]) -> None:
        """Set the next node ID in the sequence."""
        self.next_id = node_id
    
    @property
    def previous(self) -> Optional[str]:
        """Get the previous node ID in the sequence."""
        return self.previous_id
    
    @previous.setter
    def previous(self, node_id: Optional[str]) -> None:
        """Set the previous node ID in the sequence."""
        self.previous_id = node_id
    
    def set_relationships(
        self,
        parent_id: Optional[str] = None,
        next_id: Optional[str] = None,
        previous_id: Optional[str] = None
    ) -> None:
        """
        Set multiple relationships at once.
        
        Args:
            parent_id: The parent node ID
            next_id: The next node ID in sequence
            previous_id: The previous node ID in sequence
        """
        if parent_id is not None:
            self.parent_id = parent_id
        if next_id is not None:
            self.next_id = next_id
        if previous_id is not None:
            self.previous_id = previous_id
    
    def has_parent(self) -> bool:
        """Check if this node has a parent."""
        return self.parent_id is not None
    
    def has_next(self) -> bool:
        """Check if this node has a next node."""
        return self.next_id is not None
    
    def has_previous(self) -> bool:
        """Check if this node has a previous node."""
        return self.previous_id is not None
    
    def has_children(self) -> bool:
        """Check if this node has children."""
        return len(self.children_ids) > 0
    
    def add_child(self, node_id: str) -> None:
        """
        Add a child node ID to this node.
        
        Args:
            node_id: The child node ID to add
        """
        if node_id not in self.children_ids:
            self.children_ids.append(node_id)
    
    def remove_child(self, node_id: str) -> None:
        """
        Remove a child node ID from this node.
        
        Args:
            node_id: The child node ID to remove
        """
        if node_id in self.children_ids:
            self.children_ids.remove(node_id)
    
    @property
    def hash(self) -> str:
        """
        Compute MD5 hash of text + metadata for change detection.
        
        The hash is computed dynamically and cached for performance.
        
        Returns:
            MD5 hash as hex string
        """
        if self._hash is None:
            # Create a deterministic string from text and metadata
            # Sort metadata keys for consistent hashing
            metadata_str = json.dumps(self.metadata, sort_keys=True)
            content = f"{self.text}|{metadata_str}"
            self._hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return self._hash
    
    def get_text_with_context(self, include_parent: bool = True) -> str:
        """
        Get the text. Note: context from parent cannot be included as this method
        no longer has access to Node objects, only IDs.
        
        Args:
            include_parent: Deprecated - no longer used
            
        Returns:
            The node's text
        """
        return self.text


class DocumentNode(Node):
    """
    A node that represents an original document.
    
    DocumentNode is the root of a document hierarchy and typically has no parent.
    It serves as the parent for all chunks created from the document.
    
    DocumentNodes typically don't have next/previous relationships (siblings),
    as managing relationships between all documents would be memory-intensive.
    """
    
    def __init__(self, **data):
        """
        Initialize a DocumentNode.
        
        Args:
            **data: Keyword arguments for initialization
        """
        super().__init__(**data)
        
        # Set doc_id to self.id (document references itself)
        if self.doc_id is None:
            self.doc_id = self.id
        
        # Mark as document in metadata
        if 'document' not in self.metadata:
            self.metadata['document'] = True
        
        # Validate that DocumentNode typically has no parent
        if self.parent_id is not None:
            import warnings
            warnings.warn(
                "DocumentNode typically should not have a parent. "
                "If this is intentional, you can ignore this warning."
            )
    
    @classmethod
    def from_text(
        cls,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'DocumentNode':
        """
        Create a DocumentNode from text.
        
        Args:
            text: The document text
            metadata: Additional metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            A new DocumentNode instance
        """
        return cls(
            text=text,
            metadata=metadata or {},
            **kwargs
        )


class Chunk(Node):
    """
    A node that represents a document chunk.
    
    Chunks are fragments of a larger document, typically created
    through a chunking strategy. They maintain relationships to
    their parent document and to sibling chunks.
    """
    
    chunk_index: Optional[int] = None
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
    
    def __init__(self, **data):
        """
        Initialize a Chunk.
        
        Args:
            **data: Keyword arguments for initialization
        """
        super().__init__(**data)
        # Ensure metadata has chunk-specific fields
        if 'chunk' not in self.metadata:
            self.metadata['chunk'] = True
    
    @classmethod
    def from_text(
        cls,
        text: str,
        chunk_index: Optional[int] = None,
        start_char_idx: Optional[int] = None,
        end_char_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Chunk':
        """
        Create a Chunk from text.
        
        Args:
            text: The chunk text
            chunk_index: Index of this chunk in the sequence
            start_char_idx: Starting character index in the parent document
            end_char_idx: Ending character index in the parent document
            metadata: Additional metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            A new Chunk instance
        """
        return cls(
            text=text,
            chunk_index=chunk_index,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            metadata=metadata or {},
            **kwargs
        )
    
    def link_to_previous(self, previous_chunk_id: str) -> None:
        """
        Create a link to the previous chunk ID.
        Note: This method now only sets the previous_id on this chunk.
        You need to manually set the next_id on the previous chunk.
        
        Args:
            previous_chunk_id: The previous chunk ID in the sequence
        """
        self.previous_id = previous_chunk_id
    
    def get_surrounding_context(
        self,
        num_chunks_before: int = 1,
        num_chunks_after: int = 1
    ) -> str:
        """
        Get text. Note: surrounding chunks context cannot be included as this method
        no longer has access to Node objects, only IDs. To get surrounding context,
        you need to retrieve the nodes by their IDs from a document store.
        
        Args:
            num_chunks_before: Deprecated - no longer used
            num_chunks_after: Deprecated - no longer used
            
        Returns:
            The chunk's text only
        """
        return self.text
    
    def create_symbolic_nodes(
        self,
        sub_texts: List[str],
        preserve_metadata: bool = True
    ) -> List['SymNode']:
        """
        Create multiple SymNode instances that reference this chunk as parent.
        
        Args:
            sub_texts: List of text strings for the SymNodes (should be substrings of this chunk)
            preserve_metadata: Whether to copy this chunk's metadata to the SymNodes
            
        Returns:
            List of SymNode instances
        """
        # Forward reference - SymNode is defined later in this file
        sym_nodes = []
        for text in sub_texts:
            metadata = self.metadata.copy() if preserve_metadata else {}
            # Will be resolved when SymNode class is available
            sym_node = SymNode.create(
                text=text,
                parent_id=self.id,
                metadata=metadata
            )
            sym_nodes.append(sym_node)
        
        return sym_nodes


class SymNode(Node):
    """
    A symbolic node that references a parent node for hierarchical relationships.
    
    SymNodes are useful for creating smaller, more granular chunks for semantic search
    while returning larger parent nodes with more context during retrieval.
    When a SymNode is retrieved from a vector index, the parent node is automatically
    resolved and returned instead.
    """
    
    # Flag to indicate this is a symbolic node
    is_symbolic: bool = True
    
    def __init__(self, **data):
        """
        Initialize a SymNode.
        
        Args:
            **data: Keyword arguments for initialization
        """
        super().__init__(**data)
        # Ensure metadata has symbolic flag
        if 'symbolic' not in self.metadata:
            self.metadata['symbolic'] = True
        
        # Validate that parent_id is set
        if not self.parent_id:
            raise ValueError("SymNode must have a parent_id set")
    
    @classmethod
    def create(
        cls,
        text: str,
        parent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'SymNode':
        """
        Create a SymNode with a parent reference.
        
        Args:
            text: The text content for this symbolic node
            parent_id: The ID of the parent node
            metadata: Additional metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            A new SymNode instance
        """
        return cls(
            text=text,
            parent_id=parent_id,
            metadata=metadata or {},
            **kwargs
        )
    
    def requires_parent_resolution(self) -> bool:
        """
        Check if this node requires parent resolution during retrieval.
        
        Returns:
            True if parent should be resolved
        """
        return self.is_symbolic and self.parent_id is not None


class NodeWithScore(BaseModel):
    """
    Wrapper class that pairs a Node with its relevance score.
    
    This is typically used in retrieval results where documents
    are returned with their similarity scores.
    """
    
    node: Node = Field(description="The document node")
    score: float = Field(description="Relevance score (typically 0.0 to 1.0)")
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    def __repr__(self) -> str:
        return f"NodeWithScore(score={self.score:.3f}, node_id={self.node.id[:8]}, {self.node.text[:100]}...)"
    
    @property
    def text(self) -> str:
        """Convenience property to access the node's text."""
        return self.node.text
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience property to access the node's metadata."""
        return self.node.metadata
    
    @property
    def id(self) -> str:
        """Convenience property to access the node's id."""
        return self.node.id


class ObjectNode(Node):
    """
    A node that represents an object.
    """
    object_id: Any = Field(description="The object_id", default_factory=lambda: str(uuid4()))
    object_type: Any = Field(description="The object_type", default=None)
    _object: Any | None = PrivateAttr()

