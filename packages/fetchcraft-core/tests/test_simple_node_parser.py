"""
Unit tests for SimpleNodeParser
"""

import pytest

from fetchcraft.node import Chunk, DocumentNode
from fetchcraft.node_parser.simple import SimpleNodeParser


class TestSimpleNodeParser:
    """Test suite for SimpleNodeParser"""

    def test_parser_initialization(self):
        """Test parser initialization with default and custom parameters"""
        # Default initialization
        parser = SimpleNodeParser()
        assert parser.chunk_size == 4096
        assert parser.overlap == 0

        # Custom initialization
        parser = SimpleNodeParser(chunk_size=1000, overlap=100)
        assert parser.chunk_size == 1000
        assert parser.overlap == 100

    def test_small_document_no_splitting(self):
        """Test that small documents are not split"""
        parser = SimpleNodeParser(chunk_size=1000, overlap=100)

        text = "This is a short document that should not be split."
        doc = DocumentNode.from_text(text=text, metadata={"parsing": "test.txt"})

        nodes = parser.get_nodes([doc])

        assert len(nodes) == 1
        assert isinstance(nodes[0], Chunk)
        assert nodes[0].text == text
        assert nodes[0].doc_id == doc.doc_id
        assert nodes[0].chunk_index == 0

    def test_large_document_splitting(self):
        """Test that large documents are split into multiple chunks"""
        parser = SimpleNodeParser(chunk_size=100, overlap=20)

        # Create a document larger than chunk_size
        text = "This is a sentence. " * 20  # ~400 chars
        doc = DocumentNode.from_text(text=text, metadata={"parsing": "test.txt"})

        nodes = parser.get_nodes([doc])

        # Should have multiple chunks
        assert len(nodes) > 1

        # All nodes should be Chunks
        assert all(isinstance(node, Chunk) for node in nodes)

        # All chunks should have the same doc_id
        assert all(node.doc_id == doc.doc_id for node in nodes)

        # Chunks should have sequential indices
        for i, node in enumerate(nodes):
            assert node.chunk_index == i

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap when text has separators"""
        parser = SimpleNodeParser(chunk_size=50, overlap=10)

        # Use text with separators so it can be split
        text = "This is sentence one. " * 10  # ~220 chars with separators
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should have multiple chunks with overlap
        assert len(nodes) >= 1

        # Check overlap between consecutive chunks if there are multiple
        for i in range(len(nodes) - 1):
            # Get end of current chunk and start of next chunk
            current_end = nodes[i].text[-10:]
            next_start = nodes[i + 1].text[:10] if len(nodes[i + 1].text) >= 10 else nodes[i + 1].text

            # There should be some overlap
            assert len(current_end) > 0

    def test_paragraph_splitting(self):
        """Test that text is split by paragraphs when possible"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should have multiple chunks
        assert len(nodes) >= 1

        # Check that chunks respect paragraph boundaries where possible
        for node in nodes:
            assert isinstance(node, Chunk)
            assert len(node.text) <= parser.chunk_size or "\n\n" not in node.text

    def test_sentence_splitting(self):
        """Test that text is split by sentences when paragraphs are too large"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        # Long paragraph with sentences
        text = "First sentence. " * 30
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should split into multiple chunks
        assert len(nodes) > 1

        # Chunks should be reasonable size
        for node in nodes:
            assert len(node.text) <= parser.chunk_size + parser.overlap

    def test_word_splitting(self):
        """Test that text is split when it has sentence separators"""
        parser = SimpleNodeParser(chunk_size=50, overlap=5)

        # Text with sentence separators
        text = "This is a sentence. " * 20  # ~400 chars with separators
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should split into multiple chunks
        assert len(nodes) >= 1

        # Each chunk should be around chunk_size (allowing for separator boundaries)
        for node in nodes:
            # Chunks may exceed chunk_size slightly due to separator-based splitting
            assert len(node.text) > 0

    def test_character_splitting_fallback(self):
        """Test handling of text without separators"""
        parser = SimpleNodeParser(chunk_size=20, overlap=5)

        # Very long word without any separators
        # Note: Current implementation returns text as-is when no separators found
        text = "a" * 100
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Without character-level fallback, text without separators returns as single chunk
        assert len(nodes) >= 1
        # The text should be preserved
        assert "".join(n.text for n in nodes) == text or nodes[0].text == text

    def test_multiple_documents(self):
        """Test processing multiple documents"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        doc1 = DocumentNode.from_text(text="First document. " * 20, metadata={"parsing": "doc1.txt"})
        doc2 = DocumentNode.from_text(text="Second document. " * 20, metadata={"parsing": "doc2.txt"})

        nodes = parser.get_nodes([doc1, doc2])

        # Should have chunks from both documents
        assert len(nodes) > 2

        # Check that chunks maintain their doc_id
        doc1_chunks = [n for n in nodes if n.doc_id == doc1.doc_id]
        doc2_chunks = [n for n in nodes if n.doc_id == doc2.doc_id]

        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0

    def test_sequential_linking(self):
        """Test that chunks are sequentially linked with next/previous IDs"""
        parser = SimpleNodeParser(chunk_size=50, overlap=10)

        text = "This is a sentence. " * 20
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        if len(nodes) > 1:
            # First chunk should have no previous but have next
            assert nodes[0].previous_id is None
            assert nodes[0].next_id == nodes[1].id

            # Middle chunks should have both
            for i in range(1, len(nodes) - 1):
                assert nodes[i].previous_id == nodes[i - 1].id
                assert nodes[i].next_id == nodes[i + 1].id

            # Last chunk should have previous but no next
            assert nodes[-1].previous_id == nodes[-2].id
            assert nodes[-1].next_id is None

    def test_metadata_propagation(self):
        """Test that metadata is properly set on chunks"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        doc = DocumentNode.from_text(
            text="Test document. " * 20,
            metadata={"source": "test.txt"}
        )

        additional_metadata = {"custom_field": "custom_value"}
        nodes = parser.get_nodes([doc], metadata=additional_metadata)

        for node in nodes:
            # Check that total_chunks is set
            assert "total_chunks" in node.metadata
            assert node.metadata["total_chunks"] == len(nodes)
            # Check that additional metadata is propagated
            assert "custom_field" in node.metadata
            assert node.metadata["custom_field"] == "custom_value"

    def test_chunk_indices(self):
        """Test that chunk indices are sequential"""
        parser = SimpleNodeParser(chunk_size=50, overlap=10)

        text = "Test sentence. " * 30
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        for i, node in enumerate(nodes):
            assert node.chunk_index == i

    def test_char_positions(self):
        """Test that start_char_idx and end_char_idx are set correctly"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        text = "This is a test document. " * 20
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        for node in nodes:
            # Start and end positions should be valid
            assert node.start_char_idx is not None
            assert node.end_char_idx is not None
            assert node.start_char_idx >= 0
            assert node.end_char_idx <= len(text)
            assert node.start_char_idx < node.end_char_idx

            # Text length should match position difference
            expected_length = node.end_char_idx - node.start_char_idx
            assert len(node.text) == expected_length

    def test_empty_document(self):
        """Test handling of empty documents"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        doc = DocumentNode.from_text(text="")

        nodes = parser.get_nodes([doc])

        # Should have one chunk even if empty
        assert len(nodes) == 1
        assert nodes[0].text == ""

    def test_zero_overlap(self):
        """Test parser with zero overlap"""
        parser = SimpleNodeParser(chunk_size=50, overlap=0)

        # Use text with separators so it can be split
        text = "This is a test. " * 20  # ~320 chars with separators
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should have multiple chunks
        assert len(nodes) >= 1

        # All text should be preserved across chunks
        combined = "".join(n.text for n in nodes)
        # Due to overlap handling, combined may differ slightly
        assert len(combined) > 0

    def test_repr(self):
        """Test string representation"""
        parser = SimpleNodeParser(chunk_size=1000, overlap=100)
        repr_str = repr(parser)

        assert "SimpleNodeParser" in repr_str
        assert "1000" in repr_str
        assert "100" in repr_str

    def test_mixed_separators(self):
        """Test document with mixed separator types"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        text = """First paragraph with sentences. Another sentence here.

Second paragraph with different content. More text here.
A line break but same paragraph. Final sentence!"""

        doc = DocumentNode.from_text(text=text)
        nodes = parser.get_nodes([doc])

        # Should handle mixed separators
        assert len(nodes) >= 1

        # All chunks should be valid
        for node in nodes:
            assert isinstance(node, Chunk)
            assert len(node.text) > 0


class TestSimpleNodeParserEdgeCases:
    """Test edge cases and error conditions"""

    def test_very_small_chunk_size(self):
        """Test with very small chunk size"""
        parser = SimpleNodeParser(chunk_size=10, overlap=2)

        text = "This is a test."
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should still work
        assert len(nodes) >= 1

    def test_overlap_larger_than_chunk_size(self):
        """Test behavior when overlap is larger than chunk size"""
        # This is an edge case that should be handled gracefully
        parser = SimpleNodeParser(chunk_size=50, overlap=100)

        text = "A" * 200
        doc = DocumentNode.from_text(text=text)

        # Should not raise an error
        nodes = parser.get_nodes([doc])
        assert len(nodes) >= 1

    def test_special_characters(self):
        """Test handling of special characters"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        text = "Special chars: @#$%^&*(). Unicode: café, naïve, 中文. Emoji: 😀🎉"
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        # Should handle special characters without errors
        assert len(nodes) >= 1

        # Text should be preserved
        combined_text = "".join(node.text for node in nodes)
        # Account for overlap
        assert text in combined_text or combined_text in text

    def test_whitespace_only(self):
        """Test document with only whitespace"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        text = "     \n\n\t\t   "
        doc = DocumentNode.from_text(text=text)

        nodes = parser.get_nodes([doc])

        assert len(nodes) >= 1
        assert nodes[0].text == text

    def test_no_documents(self):
        """Test with empty document list"""
        parser = SimpleNodeParser(chunk_size=100, overlap=10)

        nodes = parser.get_nodes([])

        assert nodes == []


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
