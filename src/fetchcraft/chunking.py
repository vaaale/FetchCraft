"""
Chunking strategies for splitting text into chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import re
from .node import Chunk, Node, SymNode


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    A chunking strategy defines how to split text into chunks.
    """
    
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        parent_node: Optional[Node] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to chunks
            parent_node: Optional parent node for the chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CharacterChunkingStrategy(ChunkingStrategy):
    """
    Simple character-based chunking with overlap.
    
    This is the traditional approach: split text into fixed-size chunks
    with overlap between consecutive chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 200,
        separator: str = " ",
        keep_separator: bool = True
    ):
        """
        Initialize the character chunking strategy.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            separator: Character/string to use as split boundaries
            keep_separator: Whether to keep the separator in chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        self.keep_separator = keep_separator
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        parent_node: Optional[Node] = None
    ) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to chunks
            parent_node: Optional parent node for the chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunk_tuples = self._split_text(text)
        chunk_nodes = []
        
        for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_tuples):
            chunk = Chunk.from_text(
                text=chunk_text,
                chunk_index=idx,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                metadata={**metadata, "total_chunks": len(chunk_tuples), "chunk_strategy": "character"}
            )
            
            # Set doc_id from parent_node (but don't set parent relationship for first-level chunks)
            if parent_node:
                chunk.doc_id = parent_node.doc_id if hasattr(parent_node, 'doc_id') else parent_node.id
            
            # Link to previous chunk (sibling relationship)
            if chunk_nodes:
                chunk.link_to_previous(chunk_nodes[-1])
            
            chunk_nodes.append(chunk)
        
        return chunk_nodes
    
    def _split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to split
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at separator
            if end < len(text):
                # Look for the last separator within the chunk
                chunk_text = text[start:end]
                last_sep_idx = chunk_text.rfind(self.separator)
                
                if last_sep_idx != -1 and last_sep_idx > self.overlap:
                    # Adjust end to the separator position
                    if self.keep_separator:
                        end = start + last_sep_idx + len(self.separator)
                    else:
                        end = start + last_sep_idx
            else:
                # Last chunk, take everything
                end = len(text)
            
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = end - self.overlap
            
            # Ensure we make progress
            if start <= chunks[-1][1]:
                start = end
        
        return chunks
    
    def __repr__(self) -> str:
        return f"CharacterChunkingStrategy(chunk_size={self.chunk_size}, overlap={self.overlap})"


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """
    Hierarchical chunking strategy with parent-child relationships.
    
    This strategy creates:
    1. Large parent chunks for context (base chunks)
    2. Multiple levels of smaller child chunks (SymNodes) for precise retrieval
    
    Features:
    - Supports multiple child chunk sizes for multi-level hierarchy
    - Uses recursive splitting with semantic boundaries (paragraph -> line -> sentence -> space)
    - All child nodes reference their parent, so retrieval returns the full context
    
    When a child chunk matches a query, the full parent chunk is returned,
    providing more context while maintaining precise semantic search.
    """
    
    # Recursive separators in order of preference
    SEPARATORS = [
        "\n\n",  # Paragraph
        "\n",    # Line break
        ". ",    # Sentence (period)
        "? ",    # Sentence (question)
        "! ",    # Sentence (exclamation)
        "; ",    # Clause
        ", ",    # Phrase
        " ",     # Word
    ]
    
    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 200,
        child_chunks: Optional[List[int]] = None,
        child_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        Initialize the hierarchical chunking strategy.
        
        Args:
            chunk_size: Size of parent chunks (base chunks for context)
            overlap: Overlap between parent chunks
            child_chunks: List of child chunk sizes (e.g., [1024, 512, 256])
                         If None, defaults to [chunk_size // 8]
            child_overlap: Overlap between child chunks
            separators: List of separators for recursive splitting (optional)
                       If None, uses default: ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ']
            keep_separator: Whether to keep the separator in chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.child_chunks = child_chunks or [chunk_size // 8]  # Default: 512 for 4096
        self.child_overlap = child_overlap
        self.separators = separators or self.SEPARATORS
        self.keep_separator = keep_separator
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        parent_node: Optional[Node] = None
    ) -> List[Chunk | SymNode]:
        """
        Split text into hierarchical chunks with recursive splitting.
        
        Creates parent chunks (full context) and multiple levels of child SymNodes (for search).
        All child nodes reference their parent chunk, so retrieval returns the full context.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to chunks
            parent_node: Optional parent node for the chunks
            
        Returns:
            List containing both Chunk (parents) and SymNode (children) objects
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Step 1: Create parent chunks using recursive splitting
        parent_chunks = self._split_into_parent_chunks(
            text=text,
            metadata={**metadata, "chunk_strategy": "hierarchical", "chunk_type": "parent"},
            parent_node=parent_node
        )
        
        # Step 2: Create child SymNodes for each parent chunk at multiple sizes
        all_nodes = []
        
        for parent_chunk in parent_chunks:
            # Add the parent chunk to the result
            all_nodes.append(parent_chunk)
            
            # Create child chunks as SymNodes for each specified size
            for child_size in self.child_chunks:
                child_nodes = self._create_child_nodes(
                    parent_chunk=parent_chunk,
                    child_size=child_size,
                    metadata=metadata
                )
                all_nodes.extend(child_nodes)
        
        return all_nodes
    
    def _split_into_parent_chunks(
        self,
        text: str,
        metadata: dict,
        parent_node: Optional[Node] = None
    ) -> List[Chunk]:
        """
        Split text into parent chunks using recursive splitting.
        
        Args:
            text: The text to split
            metadata: Metadata to attach to chunks
            parent_node: Optional parent node
            
        Returns:
            List of parent Chunk objects
        """
        chunk_tuples = self._recursive_split(
            text=text,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        chunk_nodes = []
        for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_tuples):
            chunk = Chunk.from_text(
                text=chunk_text,
                chunk_index=idx,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                metadata={**metadata, "total_chunks": len(chunk_tuples)}
            )
            
            # Set doc_id from parent_node (but don't set parent relationship for first-level chunks)
            if parent_node:
                chunk.doc_id = parent_node.doc_id if hasattr(parent_node, 'doc_id') else parent_node.id
            
            # Link to previous chunk (sibling relationship)
            if chunk_nodes:
                chunk.link_to_previous(chunk_nodes[-1])
            
            chunk_nodes.append(chunk)
        
        return chunk_nodes
    
    def _create_child_nodes(
        self,
        parent_chunk: Chunk,
        child_size: int,
        metadata: dict
    ) -> List[SymNode]:
        """
        Create child SymNodes for a parent chunk at a specific size.
        
        Args:
            parent_chunk: The parent chunk
            child_size: Size of child chunks
            metadata: Metadata to attach to child nodes
            
        Returns:
            List of SymNode objects
        """
        # Split the parent chunk text into smaller pieces using recursive splitting
        child_tuples = self._recursive_split(
            text=parent_chunk.text,
            chunk_size=child_size,
            overlap=self.child_overlap
        )
        
        child_nodes = []
        for idx, (child_text, start_idx, end_idx) in enumerate(child_tuples):
            # Create SymNode that references the parent
            sym_node = SymNode.create(
                text=child_text,
                parent_id=parent_chunk.id,
                metadata={
                    **metadata,
                    "chunk_strategy": "hierarchical",
                    "chunk_type": "child",
                    "child_size": child_size,
                    "child_index": idx,
                    "total_children": len(child_tuples),
                    "parent_chunk_index": parent_chunk.chunk_index
                }
            )
            # Set doc_id from parent chunk
            sym_node.doc_id = parent_chunk.doc_id
            # Add child to parent's children list (SymNodes are children of chunks, not first-level)
            parent_chunk.add_child(sym_node)
            child_nodes.append(sym_node)
        
        return child_nodes
    
    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separator_index: int = 0
    ) -> List[Tuple[str, int, int]]:
        """
        Recursively split text using multiple separators.
        
        Tries separators in order (paragraph -> line -> sentence -> space)
        and recursively splits chunks that are too large.
        
        Args:
            text: The text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            separator_index: Current index in the separators list
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= chunk_size:
            return [(text, 0, len(text))]
        
        # If we've exhausted all separators, fall back to character splitting
        if separator_index >= len(self.separators):
            return self._character_split(text, chunk_size, overlap)
        
        separator = self.separators[separator_index]
        splits = self._split_by_separator(text, separator)
        
        # If we can't split by this separator, try the next one
        if len(splits) == 1:
            return self._recursive_split(text, chunk_size, overlap, separator_index + 1)
        
        # Merge splits into chunks
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for split_text, split_start, split_end in splits:
            # If adding this split would exceed chunk_size and we have content, save current chunk
            if len(current_chunk) + len(split_text) > chunk_size and current_chunk:
                chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                
                # Start new chunk with overlap
                overlap_start = max(0, current_start + len(current_chunk) - overlap)
                current_chunk = text[overlap_start:split_end]
                current_start = overlap_start
            else:
                # Add to current chunk
                if not current_chunk:
                    current_start = split_start
                current_chunk = text[current_start:split_end]
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
        
        # Recursively split any chunks that are still too large
        final_chunks = []
        for chunk_text, start_idx, end_idx in chunks:
            if len(chunk_text) > chunk_size:
                # Recursively split with next separator
                sub_chunks = self._recursive_split(
                    chunk_text,
                    chunk_size,
                    overlap,
                    separator_index + 1
                )
                # Adjust indices relative to original text
                for sub_text, sub_start, sub_end in sub_chunks:
                    final_chunks.append((sub_text, start_idx + sub_start, start_idx + sub_end))
            else:
                final_chunks.append((chunk_text, start_idx, end_idx))
        
        return final_chunks if final_chunks else [(text, 0, len(text))]
    
    def _split_by_separator(
        self,
        text: str,
        separator: str
    ) -> List[Tuple[str, int, int]]:
        """
        Split text by a separator, keeping track of positions.
        
        Args:
            text: Text to split
            separator: Separator string
            
        Returns:
            List of tuples (split_text, start_idx, end_idx)
        """
        if separator not in text:
            return [(text, 0, len(text))]
        
        splits = []
        current_pos = 0
        
        for part in text.split(separator):
            if self.keep_separator and splits:
                # Add separator to previous split
                prev_text, prev_start, prev_end = splits[-1]
                splits[-1] = (prev_text + separator, prev_start, prev_end + len(separator))
                current_pos += len(separator)
            
            if part:  # Only add non-empty parts
                end_pos = current_pos + len(part)
                splits.append((part, current_pos, end_pos))
                current_pos = end_pos
        
        return splits if splits else [(text, 0, len(text))]
    
    def _character_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """
        Fall back to simple character-based splitting.
        
        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start = end - overlap
            
            # Ensure we make progress
            if start <= chunks[-1][1]:
                start = end
        
        return chunks
    
    def __repr__(self) -> str:
        return (
            f"HierarchicalChunkingStrategy("
            f"parent_size={self.chunk_size}, "
            f"child_sizes={self.child_chunks})"
        )
