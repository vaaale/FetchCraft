from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from uuid import uuid4


class Node(BaseModel):
    """
    Base class for representing nodes in a document hierarchy.
    
    Nodes can have parent-child relationships and can be linked
    in a sequence using next/previous pointers.
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    # Relationships
    parent_id: Optional[str] = None
    next_id: Optional[str] = None
    previous_id: Optional[str] = None
    
    # Cached references (not persisted)
    _parent: Optional['Node'] = PrivateAttr(default=None)
    _next: Optional['Node'] = PrivateAttr(default=None)
    _previous: Optional['Node'] = PrivateAttr(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    @property
    def parent(self) -> Optional['Node']:
        """Get the parent node."""
        return self._parent
    
    @parent.setter
    def parent(self, node: Optional['Node']) -> None:
        """Set the parent node."""
        self._parent = node
        self.parent_id = node.id if node else None
    
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
