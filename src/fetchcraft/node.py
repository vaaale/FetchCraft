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
    - parent/children: Hierarchical relationships (any node can have both)
    - next/previous: Sequential relationships (sibling ordering)
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
    
    # Cached references (not persisted)
    _parent: Optional['Node'] = PrivateAttr(default=None)
    _children: List['Node'] = PrivateAttr(default_factory=list)
    _next: Optional['Node'] = PrivateAttr(default=None)
    _previous: Optional['Node'] = PrivateAttr(default=None)
    _hash: Optional[str] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    @property
    def parent(self) -> Optional['Node']:
        """Get the parent node."""
        return self._parent
    
    @parent.setter
    def parent(self, node: Optional['Node']) -> None:
        """Set the parent node and update parent's children."""
        self._parent = node
        self.parent_id = node.id if node else None
        if node and self.id not in node.children_ids:
            node.children_ids.append(self.id)
            if self not in node._children:
                node._children.append(self)
    
    @property
    def children(self) -> List['Node']:
        """Get the child nodes."""
        return self._children
    
    @children.setter
    def children(self, nodes: List['Node']) -> None:
        """Set the child nodes."""
        self._children = nodes
        self.children_ids = [node.id for node in nodes]
        # Update each child's parent reference
        for node in nodes:
            node._parent = self
            node.parent_id = self.id
    
    @property
    def next(self) -> Optional['Node']:
        """Get the next node in the sequence."""
        return self._next
    
    @next.setter
    def next(self, node: Optional['Node']) -> None:
        """Set the next node in the sequence."""
        self._next = node
        self.next_id = node.id if node else None
    
    @property
    def previous(self) -> Optional['Node']:
        """Get the previous node in the sequence."""
        return self._previous
    
    @previous.setter
    def previous(self, node: Optional['Node']) -> None:
        """Set the previous node in the sequence."""
        self._previous = node
        self.previous_id = node.id if node else None
    
    def set_relationships(
        self,
        parent: Optional['Node'] = None,
        next_node: Optional['Node'] = None,
        previous_node: Optional['Node'] = None
    ) -> None:
        """
        Set multiple relationships at once.
        
        Args:
            parent: The parent node
            next_node: The next node in sequence
            previous_node: The previous node in sequence
        """
        if parent is not None:
            self.parent = parent
        if next_node is not None:
            self.next = next_node
        if previous_node is not None:
            self.previous = previous_node
    
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
    
    def add_child(self, node: 'Node') -> None:
        """
        Add a child node to this node.
        
        Args:
            node: The child node to add
        """
        if node.id not in self.children_ids:
            self.children_ids.append(node.id)
            self._children.append(node)
        node._parent = self
        node.parent_id = self.id
    
    def remove_child(self, node: 'Node') -> None:
        """
        Remove a child node from this node.
        
        Args:
            node: The child node to remove
        """
        if node.id in self.children_ids:
            self.children_ids.remove(node.id)
        if node in self._children:
            self._children.remove(node)
        if node.parent_id == self.id:
            node._parent = None
            node.parent_id = None
    
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
        Get the text with optional context from parent.
        
        Args:
            include_parent: Whether to include parent text as context
            
        Returns:
            The text, optionally with parent context
        """
        if include_parent and self._parent:
            return f"Context: {self._parent.text}\n\n{self.text}"
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
    
    def link_to_previous(self, previous_chunk: 'Chunk') -> None:
        """
        Create a bidirectional link to the previous chunk.
        
        Args:
            previous_chunk: The previous chunk in the sequence
        """
        self.previous = previous_chunk
        previous_chunk.next = self
    
    def get_surrounding_context(
        self,
        num_chunks_before: int = 1,
        num_chunks_after: int = 1
    ) -> str:
        """
        Get text with surrounding chunks as context.
        
        Args:
            num_chunks_before: Number of previous chunks to include
            num_chunks_after: Number of next chunks to include
            
        Returns:
            Combined text with context
        """
        context_parts = []
        
        # Get previous chunks
        current = self._previous
        prev_chunks = []
        for _ in range(num_chunks_before):
            if current is None or not isinstance(current, Chunk):
                break
            prev_chunks.insert(0, current.text)
            current = current._previous
        
        context_parts.extend(prev_chunks)
        
        # Add current chunk
        context_parts.append(self.text)
        
        # Get next chunks
        current = self._next
        for _ in range(num_chunks_after):
            if current is None or not isinstance(current, Chunk):
                break
            context_parts.append(current.text)
            current = current._next
        
        return "\n\n".join(context_parts)
    
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

