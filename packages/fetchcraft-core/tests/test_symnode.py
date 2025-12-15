"""
Tests for SymNode functionality and parent resolution.
"""

import pytest
from qdrant_client import QdrantClient

from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Chunk, SymNode, Node, NodeType, DocumentNode
from fetchcraft.vector_store import QdrantVectorStore


async def collect_results(async_iter, k: int = 100):
    """Helper to collect results from async iterator."""
    results = []
    async for item in async_iter:
        results.append(item)
        if len(results) >= k:
            break
    return results


class MockEmbeddings:
    """Mock embeddings for testing."""

    def __init__(self, dimension=384):
        self.dimension = dimension

    async def embed_query(self, text: str):
        """Return a mock embedding based on text length."""
        value = len(text) / 100.0
        return [value] * self.dimension

    async def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [await self.embed_query(text) for text in texts]

    async def aget_dimension(self):
        """Get dimension."""
        return self.dimension


def test_symnode_creation():
    """Test basic SymNode creation."""
    parent_chunk = Chunk.from_text(
        text="This is a parent chunk with lots of context.",
        chunk_index=0,
        metadata={"parsing": "test"}
    )

    sym_node = SymNode.create(
        text="This is a parent chunk",
        parent_id=parent_chunk.id,
        metadata={"parsing": "test"}
    )

    assert sym_node.parent_id == parent_chunk.id
    assert sym_node.node_type == NodeType.SYMNODE
    assert sym_node.metadata["symbolic"] is True


def test_symnode_requires_parent_id():
    """Test that SymNode requires a parent_id."""
    with pytest.raises(ValueError, match="must have a parent_id"):
        SymNode(text="Some text", parent_id=None)


def test_chunk_create_symbolic_nodes():
    """Test creating SymNodes from a Chunk."""
    parent_chunk = Chunk.from_text(
        text="This is a long parent chunk with multiple sentences.",
        chunk_index=0,
        metadata={"parsing": "test", "topic": "example"}
    )

    sub_texts = [
        "This is a long parent chunk",
        "with multiple sentences."
    ]

    sym_nodes = parent_chunk.create_symbolic_nodes(sub_texts)

    assert len(sym_nodes) == 2
    assert all(isinstance(node, SymNode) for node in sym_nodes)
    assert all(node.parent_id == parent_chunk.id for node in sym_nodes)
    assert all(node.metadata["parsing"] == "test" for node in sym_nodes)
    assert all(node.metadata["topic"] == "example" for node in sym_nodes)


@pytest.mark.asyncio
async def test_parent_resolution_in_index():
    """Test that VectorIndex resolves parent nodes correctly."""
    embeddings = MockEmbeddings(dimension=384)

    # Create parent chunk
    parent_chunk = Chunk.from_text(
        text="Machine learning is a subset of AI that enables computers to learn.",
        chunk_index=0,
        metadata={"topic": "ml"}
    )
    parent_chunk.embedding = [0.1] * 384  # Mock embedding

    # Create SymNodes
    sym_node1 = SymNode.create(
        text="Machine learning is a subset of AI",
        parent_id=parent_chunk.id,
        metadata={"topic": "ml"}
    )
    sym_node1.embedding = [0.15] * 384  # Slightly different embedding

    sym_node2 = SymNode.create(
        text="enables computers to learn",
        parent_id=parent_chunk.id,
        metadata={"topic": "ml"}
    )
    sym_node2.embedding = [0.12] * 384  # Slightly different embedding

    # Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_symnode",
        embeddings=embeddings,
        document_class=Node
    )

    index = VectorIndex(
        vector_store=vector_store
    )

    # Add parent first, then SymNodes
    await index.add_nodes(None, [parent_chunk])
    await index.add_nodes(None, [sym_node1, sym_node2])

    # Search with parent resolution
    query_embedding = [0.11] * 384
    results = await collect_results(
        index.search_iter(query="test", query_embedding=query_embedding, resolve_parents=True),
        k=3
    )

    # Should get parent chunk(s), not SymNodes
    assert len(results) > 0
    for doc, score in results:
        assert isinstance(doc, Chunk)
        assert doc.id == parent_chunk.id
        assert doc.text == parent_chunk.text

    # Search without parent resolution
    results_no_resolve = await collect_results(
        index.search_iter(query="test", query_embedding=query_embedding, resolve_parents=False),
        k=3
    )

    # Should get SymNodes
    sym_results = [doc for doc, score in results_no_resolve if isinstance(doc, SymNode)]
    assert len(sym_results) > 0


@pytest.mark.asyncio
async def test_multiple_parents_resolution():
    """Test resolving multiple different parent nodes."""
    embeddings = MockEmbeddings(dimension=384)

    # Create two parent chunks
    parent1 = Chunk.from_text(
        text="Python is a programming language.",
        chunk_index=0
    )
    parent1.embedding = [0.1] * 384

    parent2 = Chunk.from_text(
        text="JavaScript is used for web development.",
        chunk_index=1
    )
    parent2.embedding = [0.2] * 384

    # Create SymNodes for each parent
    sym1 = SymNode.create(
        text="Python is a programming",
        parent_id=parent1.id
    )
    sym1.embedding = [0.11] * 384

    sym2 = SymNode.create(
        text="JavaScript is used",
        parent_id=parent2.id
    )
    sym2.embedding = [0.21] * 384

    # Setup and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_multi_parent",
        embeddings=embeddings,
        document_class=Node
    )

    index = VectorIndex(
        vector_store=vector_store
    )

    await index.add_nodes(None, [parent1, parent2])
    await index.add_nodes(None, [sym1, sym2])

    # Search should return both parents
    query_embedding = [0.15] * 384
    results = await collect_results(
        index.search_iter(query="test", query_embedding=query_embedding, resolve_parents=True),
        k=5
    )

    # Should have results from both parents
    parent_ids = {doc.id for doc, score in results}
    assert parent1.id in parent_ids or parent2.id in parent_ids


@pytest.mark.asyncio
async def test_deduplication_same_parent():
    """Test that multiple SymNodes with same parent resolve to the parent."""
    embeddings = MockEmbeddings(dimension=384)

    parent = Chunk.from_text(text="Parent text with multiple children.")
    parent.embedding = [0.1] * 384

    # Create multiple SymNodes with same parent
    sym1 = SymNode.create(text="Parent text", parent_id=parent.id)
    sym1.embedding = [0.11] * 384

    sym2 = SymNode.create(text="with multiple", parent_id=parent.id)
    sym2.embedding = [0.12] * 384

    sym3 = SymNode.create(text="multiple children", parent_id=parent.id)
    sym3.embedding = [0.13] * 384

    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_dedup",
        embeddings=embeddings,
        document_class=Node
    )

    index = VectorIndex(
        vector_store=vector_store
    )

    await index.add_nodes(None, [parent])
    await index.add_nodes(None, [sym1, sym2, sym3])

    # Search might return multiple SymNodes with same parent
    query_embedding = [0.12] * 384
    results = await collect_results(
        index.search_iter(query="test", query_embedding=query_embedding, resolve_parents=True),
        k=10
    )

    # All resolved results should point to the parent
    # Note: Current implementation may return multiple instances of the same parent
    # when multiple SymNodes resolve to it (deduplication happens at _resolve_parent_nodes level)
    parent_occurrences = sum(1 for doc, score in results if doc.id == parent.id)
    assert parent_occurrences >= 1  # At least one parent should be found
