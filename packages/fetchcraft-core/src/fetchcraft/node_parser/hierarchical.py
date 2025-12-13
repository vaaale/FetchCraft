from typing import *

from fetchcraft.text_splitter import TextSplitter
from fetchcraft.node import Chunk, Node, DocumentNode, SymNode
from fetchcraft.node_parser.base import NodeParser
from fetchcraft.text_splitter import RecursiveTextSplitter


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
    text_splitter: TextSplitter
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 200,
        child_sizes: Optional[List[int]] = None,
        child_overlap: int = 50,
        text_splitter: Optional[TextSplitter] = None
    ):
        if text_splitter is None:
            text_splitter = RecursiveTextSplitter()
        super().__init__(chunk_size=chunk_size, overlap=overlap, text_splitter=text_splitter)
        if child_sizes is not None:
            self.child_sizes = child_sizes
        self.child_overlap = child_overlap
    
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
        """Create parent chunks using the configured text splitter."""
        chunk_tuples = self.text_splitter.split(text, self.chunk_size, self.overlap)
        
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
                    "chunk_size": self.chunk_size,
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
        child_tuples = self.text_splitter.split(
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

    def __repr__(self) -> str:
        return (f"HierarchicalNodeParser(chunk_size={self.chunk_size}, "
                f"overlap={self.overlap}, child_sizes={self.child_sizes})")
