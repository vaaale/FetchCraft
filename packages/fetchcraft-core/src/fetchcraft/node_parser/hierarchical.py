from typing import *

from fetchcraft.node import Chunk, Node, DocumentNode, SymNode
from fetchcraft.node_parser.base import NodeParser


class HierarchicalNodeParser(NodeParser):
    """
    Node parser that creates a hierarchical structure with parent chunks and child SymNodes.
    
    This parser creates:
    1. Parent chunks at the specified chunk_size (for context)
    2. Multiple levels of child SymNodes at smaller sizes (for semantic search)
    
    When a SymNode is retrieved, it can be resolved to its parent chunk to get full context.
    """
    
    chunk_size: int = 4096
    overlap: int = 200
    child_sizes: List[int] = [1024, 512]
    child_overlap: int = 50
    keep_separator: bool = True
    
    # Recursive separators in order of preference
    separators: List[str] = [
        "\n\n",  # Paragraph
        "\n",  # Line break
        ". ",  # Sentence (period)
        "? ",  # Sentence (question)
        "! ",  # Sentence (exclamation)
        "; ",  # Clause
        ", ",  # Phrase
        " ",  # Word
    ]
    
    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 200,
        child_sizes: Optional[List[int]] = None,
        child_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        if child_sizes is not None:
            self.child_sizes = child_sizes
        self.child_overlap = child_overlap
        if separators is not None:
            self.separators = separators
    
    def get_nodes(self, documents: List[DocumentNode], metadata: Optional[Dict[str, Any]] = None) -> List[Node]:
        """
        Parse documents into hierarchical chunks and SymNodes.
        
        Returns both parent Chunk objects and child SymNode objects.
        """
        if metadata is None:
            metadata = {}
        
        all_nodes = []
        
        for document in documents:
            document_nodes = []
            text = document.text
            metadata.update(document.metadata)
            
            # Step 1: Create parent chunks
            parent_chunks = self._create_parent_chunks(document, text, metadata)
            
            # Step 2: For each parent chunk, create child SymNodes
            self.child_sizes.sort() # Ascending order
            for parent_chunk in parent_chunks:
                document_nodes.append(parent_chunk)
                
                # Create child SymNodes at each specified size
                for child_size in self.child_sizes:
                    child_nodes = self._create_child_nodes(parent_chunk, child_size, metadata)
                    document_nodes.extend(child_nodes)
                    for cn in child_nodes:
                        document.add_child(cn.id)

                    # If the node fit the whole document, skip the remaining child sizes
                    last_node = document_nodes[-1]
                    last_node_size = len(last_node.text)
                    if last_node_size >= len(text):
                        break

            node_ids = list(set([n.id for n in document_nodes] + document.children_ids))
            document.children_ids = node_ids

            all_nodes.extend(document_nodes)

        return all_nodes
    
    def _create_parent_chunks(
        self,
        document: DocumentNode,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create parent chunks using recursive splitting."""
        chunk_tuples = self._recursive_split(text, self.chunk_size, self.overlap)
        
        parent_chunks = []
        for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_tuples):
            chunk = Chunk.from_text(
                text=chunk_text,
                chunk_index=idx,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                metadata={
                    **metadata,
                    "total_chunks": len(chunk_tuples),
                    "chunk_strategy": "hierarchical",
                    "chunk_type": "parent"
                }
            )
            
            # Set doc_id
            chunk.doc_id = document.doc_id
            
            # Link to previous chunk
            if parent_chunks:
                prev_chunk = parent_chunks[-1]
                chunk.previous_id = prev_chunk.id
                prev_chunk.next_id = chunk.id
            
            parent_chunks.append(chunk)
        
        return parent_chunks
    
    def _create_child_nodes(
        self,
        parent_chunk: Chunk,
        child_size: int,
        metadata: Dict[str, Any]
    ) -> List[SymNode]:
        """Create child SymNodes for a parent chunk at a specific size."""
        child_tuples = self._recursive_split(
            parent_chunk.text,
            child_size,
            self.child_overlap
        )
        
        child_nodes = []
        for idx, (child_text, start_idx, end_idx) in enumerate(child_tuples):
            sym_node = SymNode.create(
                text=child_text,
                parent_id=parent_chunk.id,
                metadata={
                    **metadata,
                    "chunk_size": child_size,
                    "child_index": idx,
                }
            )
            
            # Set doc_id from parent chunk
            sym_node.doc_id = parent_chunk.doc_id
            
            # Add child to parent's children list
            parent_chunk.add_child(sym_node.id)
            
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
        """Split text by a separator, keeping track of positions."""
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
        """Fall back to simple character-based splitting."""
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
        return (f"HierarchicalNodeParser(chunk_size={self.chunk_size}, "
                f"overlap={self.overlap}, child_sizes={self.child_sizes})")
