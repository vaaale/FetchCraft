"""
Tests for hash-based update detection in vector stores.
"""

import pytest
from uuid import uuid4
from qdrant_client import QdrantClient
from fetchcraft import Node, QdrantVectorStore, OpenAIEmbeddings, VectorIndex


@pytest.fixture
def mock_embeddings():
    """Mock embeddings that return predictable values."""
    class MockEmbeddings:
        dimension = 384
        
        async def embed_query(self, text: str):
            return [0.1] * self.dimension
        
        async def embed_documents(self, texts: list[str]):
            return [[0.1] * self.dimension for _ in texts]
    
    return MockEmbeddings()


@pytest.mark.asyncio
async def test_node_hash_computation():
    """Test that nodes can compute their hash correctly."""
    node1 = Node(text="Hello world", metadata={"source": "test"})
    
    assert node1.hash is not None
    assert len(node1.hash) == 32  # MD5 hash is 32 hex chars
    
    # Same content should produce same hash
    node2 = Node(text="Hello world", metadata={"source": "test"})
    
    assert node1.hash == node2.hash


@pytest.mark.asyncio
async def test_node_hash_differs_with_different_content():
    """Test that different content produces different hashes."""
    node1 = Node(text="Hello world", metadata={"source": "test"})
    
    node2 = Node(text="Different text", metadata={"source": "test"})
    
    assert node1.hash != node2.hash


@pytest.mark.asyncio
async def test_node_hash_differs_with_different_metadata():
    """Test that different metadata produces different hashes."""
    node1 = Node(text="Hello world", metadata={"source": "test1"})
    
    node2 = Node(text="Hello world", metadata={"source": "test2"})
    
    assert node1.hash != node2.hash


@pytest.mark.asyncio
async def test_vector_store_skips_unchanged_documents(mock_embeddings):
    """Test that vector store skips documents that haven't changed."""
    client = QdrantClient(":memory:")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_hash",
        embeddings=mock_embeddings
    )
    
    index = VectorIndex(vector_store=vector_store, index_id="test")
    
    # Add a document with a proper UUID
    doc_id = str(uuid4())
    node1 = Node(id=doc_id, text="Hello world", metadata={"source": "test"})
    await index.add_nodes([node1])
    
    # Retrieve it to verify it was added
    result = await vector_store.get_node(doc_id, index_id="test")
    assert result is not None
    assert result.text == "Hello world"
    original_hash = result.hash
    
    # Try to add the same document again (unchanged)
    node2 = Node(id=doc_id, text="Hello world", metadata={"source": "test"})
    
    # Hash should be the same
    assert node2.hash == original_hash
    
    # Add it again - should be skipped
    await index.add_nodes([node2])
    
    # Retrieve it - should still be the same
    result2 = await vector_store.get_node(doc_id, index_id="test")
    assert result2 is not None
    assert result2.hash == original_hash


@pytest.mark.asyncio
async def test_vector_store_updates_changed_documents(mock_embeddings):
    """Test that vector store updates documents that have changed."""
    client = QdrantClient(":memory:")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_hash_update",
        embeddings=mock_embeddings
    )
    
    index = VectorIndex(vector_store=vector_store, index_id="test")
    
    # Add a document with proper UUID
    doc_id = str(uuid4())
    node1 = Node(id=doc_id, text="Hello world", metadata={"source": "test"})
    await index.add_nodes([node1])
    
    # Retrieve it
    result = await vector_store.get_node(doc_id, index_id="test")
    assert result is not None
    assert result.text == "Hello world"
    original_hash = result.hash
    
    # Update the document (change text)
    node2 = Node(id=doc_id, text="Hello world updated!", metadata={"source": "test"})
    
    # Hash should be different
    assert node2.hash != original_hash
    
    # Add it again - should update
    await index.add_nodes([node2])
    
    # Retrieve it - should have new content
    result2 = await vector_store.get_node(doc_id, index_id="test")
    assert result2 is not None
    assert result2.text == "Hello world updated!"
    assert result2.hash == node2.hash
    assert result2.hash != original_hash


@pytest.mark.asyncio
async def test_vector_store_updates_on_metadata_change(mock_embeddings):
    """Test that vector store updates when only metadata changes."""
    client = QdrantClient(":memory:")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_hash_meta",
        embeddings=mock_embeddings
    )
    
    index = VectorIndex(vector_store=vector_store, index_id="test")
    
    # Add a document with proper UUID
    doc_id = str(uuid4())
    node1 = Node(id=doc_id, text="Hello world", metadata={"version": 1})
    await index.add_nodes([node1])
    
    # Retrieve it
    result = await vector_store.get_node(doc_id, index_id="test")
    assert result is not None
    original_hash = result.hash
    
    # Update metadata only
    node2 = Node(id=doc_id, text="Hello world", metadata={"version": 2})
    
    # Hash should be different
    assert node2.hash != original_hash
    
    # Add it again - should update
    await index.add_nodes([node2])
    
    # Retrieve it - should have new metadata
    result2 = await vector_store.get_node(doc_id, index_id="test")
    assert result2 is not None
    assert result2.metadata["version"] == 2
    assert result2.hash == node2.hash
    assert result2.hash != original_hash


@pytest.mark.asyncio
async def test_hash_automatically_computed_if_missing(mock_embeddings):
    """Test that hash is automatically computed if not present."""
    client = QdrantClient(":memory:")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_auto_hash",
        embeddings=mock_embeddings
    )
    
    index = VectorIndex(vector_store=vector_store, index_id="test")
    
    # Add a document without computing hash manually, but with proper UUID
    doc_id = str(uuid4())
    node = Node(id=doc_id, text="Hello world", metadata={"source": "test"})
    # Don't call update_hash() - should be computed automatically
    
    await index.add_nodes([node])
    
    # Retrieve it - should have hash
    result = await vector_store.get_node(doc_id, index_id="test")
    assert result is not None
    assert result.hash is not None
    assert len(result.hash) == 32
