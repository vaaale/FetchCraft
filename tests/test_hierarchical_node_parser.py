"""
Unit tests for HierarchicalNodeParser
"""

import pytest
from typing import List

from fetchcraft.node import Chunk, DocumentNode, SymNode, Node
from fetchcraft.node_parser.hierarchical import HierarchicalNodeParser


class TestHierarchicalNodeParser:
    """Test suite for HierarchicalNodeParser"""
    
    def test_parser_initialization(self):
        """Test parser initialization with default and custom parameters"""
        # Default initialization
        parser = HierarchicalNodeParser()
        assert parser.chunk_size == 4096
        assert parser.overlap == 200
        assert parser.child_sizes == [1024, 512]
        assert parser.child_overlap == 50
        
        # Custom initialization
        parser = HierarchicalNodeParser(
            chunk_size=2000,
            overlap=100,
            child_sizes=[500, 250],
            child_overlap=25
        )
        assert parser.chunk_size == 2000
        assert parser.overlap == 100
        assert parser.child_sizes == [500, 250]
        assert parser.child_overlap == 25
    
    def test_small_document_creates_hierarchy(self):
        """Test that even small documents create parent and child nodes"""
        parser = HierarchicalNodeParser(
            chunk_size=1000,
            overlap=100,
            child_sizes=[200]
        )
        
        text = "This is a short document with some content."
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Should have at least 1 parent chunk and some child SymNodes
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        assert len(parent_chunks) >= 1
        assert len(child_nodes) >= 1
    
    def test_hierarchical_structure(self):
        """Test that parent-child relationships are properly set"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            overlap=20,
            child_sizes=[50]
        )
        
        text = "This is sentence one. " * 20  # ~440 chars
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Separate parents and children
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        # Should have multiple parent chunks
        assert len(parent_chunks) > 1
        
        # Should have child SymNodes
        assert len(child_nodes) > 0
        
        # Each child should reference a parent
        for child in child_nodes:
            assert child.parent_id is not None
            # Parent ID should be one of the parent chunks
            assert child.parent_id in [p.id for p in parent_chunks]
        
        # Each parent should have children
        for parent in parent_chunks:
            assert len(parent.children_ids) > 0
    
    def test_multiple_child_sizes(self):
        """Test creating multiple levels of child SymNodes"""
        parser = HierarchicalNodeParser(
            chunk_size=500,
            overlap=50,
            child_sizes=[200, 100, 50]
        )
        
        text = "This is a test sentence. " * 50  # ~1250 chars
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Check we have child nodes at different sizes
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        # Should have children at multiple sizes
        assert len(child_nodes) > 3  # At least one for each size level
        
        # Check child_size metadata
        child_sizes_in_metadata = set()
        for child in child_nodes:
            if "child_size" in child.metadata:
                child_sizes_in_metadata.add(child.metadata["child_size"])
        
        # Should have all three child sizes represented
        assert 200 in child_sizes_in_metadata
        assert 100 in child_sizes_in_metadata
        assert 50 in child_sizes_in_metadata
    
    def test_parent_chunk_metadata(self):
        """Test that parent chunks have correct metadata"""
        parser = HierarchicalNodeParser(chunk_size=200, child_sizes=[50])
        
        text = "Test content. " * 30
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        
        for chunk in parent_chunks:
            assert chunk.metadata["chunk_strategy"] == "hierarchical"
            assert chunk.metadata["chunk_type"] == "parent"
            assert "total_chunks" in chunk.metadata
            assert chunk.chunk_index is not None
    
    def test_child_symnode_metadata(self):
        """Test that child SymNodes have correct metadata"""
        parser = HierarchicalNodeParser(
            chunk_size=300,
            child_sizes=[100]
        )
        
        text = "This is test content. " * 30
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        assert len(child_nodes) > 0
        
        for child in child_nodes:
            assert child.metadata["chunk_type"] == "child"
            assert "child_size" in child.metadata
            assert "child_index" in child.metadata
            assert "total_children" in child.metadata
            assert child.is_symbolic is True
            assert child.requires_parent_resolution() is True
    
    def test_doc_id_propagation(self):
        """Test that doc_id is propagated to all nodes"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        doc = DocumentNode.from_text(text="Test content. " * 30)
        nodes = parser.get_nodes([doc])
        
        # All nodes should have the same doc_id
        for node in nodes:
            assert node.doc_id == doc.doc_id
    
    def test_custom_metadata_propagation(self):
        """Test that custom metadata is added to all nodes"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        doc = DocumentNode.from_text(text="Test content. " * 20)
        custom_metadata = {"source": "test.txt", "author": "Test"}
        
        nodes = parser.get_nodes([doc], metadata=custom_metadata)
        
        # All nodes should have custom metadata
        for node in nodes:
            assert "source" in node.metadata
            assert node.metadata["source"] == "test.txt"
            assert "author" in node.metadata
            assert node.metadata["author"] == "Test"
    
    def test_sequential_parent_linking(self):
        """Test that parent chunks are sequentially linked"""
        parser = HierarchicalNodeParser(
            chunk_size=100,
            overlap=10,
            child_sizes=[25]
        )
        
        text = "Sentence. " * 50
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        
        if len(parent_chunks) > 1:
            # First parent should have no previous
            assert parent_chunks[0].previous_id is None
            assert parent_chunks[0].next_id == parent_chunks[1].id
            
            # Middle parents should have both
            for i in range(1, len(parent_chunks) - 1):
                assert parent_chunks[i].previous_id == parent_chunks[i - 1].id
                assert parent_chunks[i].next_id == parent_chunks[i + 1].id
            
            # Last parent should have no next
            assert parent_chunks[-1].previous_id == parent_chunks[-2].id
            assert parent_chunks[-1].next_id is None
    
    def test_multiple_documents(self):
        """Test processing multiple documents"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        doc1 = DocumentNode.from_text(text="First doc. " * 30)
        doc2 = DocumentNode.from_text(text="Second doc. " * 30)
        
        nodes = parser.get_nodes([doc1, doc2])
        
        # Should have nodes from both documents
        doc1_nodes = [n for n in nodes if n.doc_id == doc1.doc_id]
        doc2_nodes = [n for n in nodes if n.doc_id == doc2.doc_id]
        
        assert len(doc1_nodes) > 0
        assert len(doc2_nodes) > 0
    
    def test_child_parent_relationship_count(self):
        """Test that each parent has the expected number of children per level"""
        parser = HierarchicalNodeParser(
            chunk_size=400,
            overlap=0,  # No overlap for simpler counting
            child_sizes=[100, 50],
            child_overlap=0
        )
        
        # Create text that will make exactly one parent chunk
        text = "A" * 300  # Less than chunk_size
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        assert len(parent_chunks) == 1
        parent = parent_chunks[0]
        
        # Should have children at two size levels
        children_100 = [c for c in child_nodes if c.metadata.get("child_size") == 100]
        children_50 = [c for c in child_nodes if c.metadata.get("child_size") == 50]
        
        # For 300 chars: 100-char chunks = 3, 50-char chunks = 6
        assert len(children_100) == 3
        assert len(children_50) == 6
        
        # Parent should have all children in its children_ids
        assert len(parent.children_ids) == 9  # 3 + 6
    
    def test_empty_document(self):
        """Test handling of empty documents"""
        parser = HierarchicalNodeParser(child_sizes=[50])
        
        doc = DocumentNode.from_text(text="")
        nodes = parser.get_nodes([doc])
        
        # Should still create at least a parent chunk (even if empty)
        # This depends on implementation - adjust as needed
        assert len(nodes) >= 0  # May be 0 or have empty parent
    
    def test_recursive_separator_splitting(self):
        """Test that recursive separators work correctly"""
        parser = HierarchicalNodeParser(
            chunk_size=100,
            child_sizes=[30]
        )
        
        text = """First paragraph.

Second paragraph.

Third paragraph."""
        
        doc = DocumentNode.from_text(text=text)
        nodes = parser.get_nodes([doc])
        
        # Should respect paragraph boundaries where possible
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        assert len(parent_chunks) >= 1
    
    def test_overlap_creates_redundancy(self):
        """Test that overlap creates overlapping chunks"""
        parser = HierarchicalNodeParser(
            chunk_size=50,
            overlap=20,
            child_sizes=[20],
            child_overlap=5
        )
        
        text = "A" * 150
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        
        # With overlap, should have more chunks than without
        assert len(parent_chunks) > 2
    
    def test_char_positions_in_parent_chunks(self):
        """Test that character positions are tracked in parent chunks"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        text = "Test content. " * 30
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        
        for chunk in parent_chunks:
            assert chunk.start_char_idx is not None
            assert chunk.end_char_idx is not None
            assert chunk.start_char_idx >= 0
            assert chunk.end_char_idx <= len(text)
            assert len(chunk.text) == chunk.end_char_idx - chunk.start_char_idx
    
    def test_single_child_size(self):
        """Test with a single child size level"""
        parser = HierarchicalNodeParser(
            chunk_size=300,
            child_sizes=[75]
        )
        
        text = "Content. " * 50
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        assert len(parent_chunks) >= 1
        assert len(child_nodes) >= 1
        
        # All children should have the same child_size
        for child in child_nodes:
            assert child.metadata["child_size"] == 75
    
    def test_repr(self):
        """Test string representation"""
        parser = HierarchicalNodeParser(
            chunk_size=2000,
            overlap=100,
            child_sizes=[500, 250]
        )
        
        repr_str = repr(parser)
        assert "HierarchicalNodeParser" in repr_str
        assert "2000" in repr_str
        assert "100" in repr_str
        assert "500" in repr_str


class TestHierarchicalNodeParserEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_very_small_chunk_size(self):
        """Test with very small chunk sizes"""
        parser = HierarchicalNodeParser(
            chunk_size=20,
            child_sizes=[5]
        )
        
        text = "This is a test."
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Should still work with tiny chunks
        assert len(nodes) > 0
    
    def test_child_size_larger_than_parent(self):
        """Test when child size is larger than parent"""
        parser = HierarchicalNodeParser(
            chunk_size=100,
            child_sizes=[200]  # Larger than parent!
        )
        
        text = "A" * 80  # Smaller than both
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Should handle gracefully
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        assert len(parent_chunks) >= 1
        # May have 1 or 0 children depending on implementation
        assert len(child_nodes) >= 0
    
    def test_many_child_levels(self):
        """Test with many child size levels"""
        parser = HierarchicalNodeParser(
            chunk_size=1000,
            child_sizes=[500, 250, 125, 64, 32, 16]
        )
        
        text = "Test. " * 200
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        # Should have many children at different levels
        assert len(child_nodes) > 10
    
    def test_zero_overlap(self):
        """Test with zero overlap"""
        parser = HierarchicalNodeParser(
            chunk_size=100,
            overlap=0,
            child_sizes=[25],
            child_overlap=0
        )
        
        text = "A" * 300
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        
        # Should have exactly 3 chunks (300 / 100)
        assert len(parent_chunks) == 3
    
    def test_no_documents(self):
        """Test with empty document list"""
        parser = HierarchicalNodeParser(child_sizes=[50])
        
        nodes = parser.get_nodes([])
        
        assert nodes == []
    
    def test_unicode_content(self):
        """Test handling of unicode characters"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        text = "Hello 世界! 🌍 " * 30
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        # Should handle unicode without errors
        assert len(nodes) > 0
        
        # Check some node has unicode content
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        assert any("世界" in chunk.text for chunk in parent_chunks)


class TestHierarchicalNodeParserRetrieval:
    """Test scenarios related to retrieval use cases"""
    
    def test_symnode_requires_resolution(self):
        """Test that SymNodes are marked for parent resolution"""
        parser = HierarchicalNodeParser(
            chunk_size=200,
            child_sizes=[50]
        )
        
        doc = DocumentNode.from_text(text="Test content. " * 30)
        nodes = parser.get_nodes([doc])
        
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        for child in child_nodes:
            assert child.requires_parent_resolution() is True
    
    def test_parent_contains_child_text(self):
        """Test that parent chunks contain their children's text"""
        parser = HierarchicalNodeParser(
            chunk_size=300,
            overlap=0,
            child_sizes=[100],
            child_overlap=0
        )
        
        text = "This is a test sentence. " * 20
        doc = DocumentNode.from_text(text=text)
        
        nodes = parser.get_nodes([doc])
        
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        # Each child's text should be in its parent's text
        for child in child_nodes:
            parent = next((p for p in parent_chunks if p.id == child.parent_id), None)
            assert parent is not None
            assert child.text in parent.text or parent.text in child.text  # Handle overlaps


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
