"""
Tests for Retriever functionality.
"""

import pytest
from qdrant_client import QdrantClient

from fetchcraft import (
    Node,
    Chunk,
    SymNode,
    NodeWithScore,
    QdrantVectorStore,
    VectorIndex,
    VectorIndexRetriever
)


class MockEmbeddings:
    """Mock embeddings for testing."""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    async def embed_query(self, text: str):
        """Return a mock embedding based on text length."""
        # Simple deterministic embedding based on text
        value = len(text) / 100.0
        return [value] * self.dimension
    
    async def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [await self.embed_query(text) for text in texts]
    
    async def aget_dimension(self):
        """Get dimension."""
        return self.dimension


@pytest.mark.asyncio
async def test_basic_retriever():
    """Test basic retriever functionality."""
    embeddings = MockEmbeddings(dimension=384)
    
    # Create documents
    documents = [
        Node(text="Python programming", embedding=[0.1] * 384),
        Node(text="JavaScript development", embedding=[0.2] * 384),
        Node(text="Machine learning", embedding=[0.3] * 384),
    ]
    
    # Setup index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_retriever",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_documents(documents)
    
    # Create retriever
    retriever = index.as_retriever(top_k=2)
    
    # Retrieve
    results = await retriever.aretrieve("test query")
    
    assert len(results) == 2
    assert all(isinstance(result, NodeWithScore) for result in results)
    assert all(isinstance(result.node, Node) for result in results)
    assert all(isinstance(result.score, float) for result in results)


@pytest.mark.asyncio
async def test_retriever_top_k_override():
    """Test overriding top_k in retrieve call."""
    embeddings = MockEmbeddings(dimension=384)
    
    documents = [
        Node(text=f"Document {i}", embedding=[i/10.0] * 384)
        for i in range(5)
    ]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_topk",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_documents(documents)
    
    # Create retriever with top_k=2
    retriever = index.as_retriever(top_k=2)
    
    # Default top_k
    results = await retriever.aretrieve("query")
    assert len(results) == 2
    
    # Override top_k
    results = await retriever.aretrieve("query", top_k=4)
    assert len(results) == 4


@pytest.mark.asyncio
async def test_retriever_update_config():
    """Test updating retriever configuration."""
    embeddings = MockEmbeddings(dimension=384)
    
    documents = [
        Node(text=f"Doc {i}", embedding=[i/10.0] * 384)
        for i in range(5)
    ]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_config",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_documents(documents)
    
    retriever = index.as_retriever(top_k=2)
    
    # Update config
    retriever.update_config(top_k=3)
    assert retriever.top_k == 3
    
    # Verify new config is used
    results = await retriever.aretrieve("query")
    assert len(results) == 3


@pytest.mark.asyncio
async def test_retriever_with_symnode():
    """Test retriever with SymNode parent resolution."""
    embeddings = MockEmbeddings(dimension=384)
    
    # Create parent chunk
    parent = Chunk.from_text(text="Parent chunk with context")
    parent.embedding = [0.5] * 384
    
    # Create SymNodes
    sym1 = SymNode.create(text="Parent chunk", parent_id=parent.id)
    sym1.embedding = [0.51] * 384
    
    sym2 = SymNode.create(text="with context", parent_id=parent.id)
    sym2.embedding = [0.52] * 384
    
    # Setup index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_symnode_retriever",
        embeddings=embeddings,
        document_class=Node
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    
    await index.add_documents([parent])
    await index.add_documents([sym1, sym2])
    
    # Retriever with parent resolution
    retriever = index.as_retriever(
        top_k=5,
        resolve_parents=True
    )
    
    results = await retriever.aretrieve("query")
    
    # Should get parent, not SymNodes
    for result in results:
        if result.node.id == parent.id:
            assert isinstance(result.node, Chunk)
            break
    else:
        pytest.fail("Parent chunk not found in results")


@pytest.mark.asyncio
async def test_retriever_without_parent_resolution():
    """Test retriever with parent resolution disabled."""
    embeddings = MockEmbeddings(dimension=384)
    
    parent = Chunk.from_text(text="Parent text")
    parent.embedding = [0.5] * 384
    
    sym = SymNode.create(text="Parent", parent_id=parent.id)
    sym.embedding = [0.51] * 384
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_no_resolve",
        embeddings=embeddings,
        document_class=Node
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    
    await index.add_documents([parent, sym])
    
    # Retriever without parent resolution
    retriever = index.as_retriever(
        top_k=5,
        resolve_parents=False
    )
    
    results = await retriever.aretrieve("query")
    
    # Should include SymNode
    has_symnode = any(isinstance(result.node, SymNode) for result in results)
    assert has_symnode, "SymNode should be in results when resolve_parents=False"


@pytest.mark.asyncio
async def test_direct_retriever_creation():
    """Test creating VectorIndexRetriever directly."""
    embeddings = MockEmbeddings(dimension=384)
    
    documents = [Node(text="Test", embedding=[0.1] * 384)]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_direct",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_documents(documents)
    
    # Create directly
    retriever = VectorIndexRetriever(
        vector_index=index,
        top_k=1
    )
    
    results = await retriever.aretrieve("test")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_aretrieve_alias():
    """Test that aretrieve works as alias."""
    embeddings = MockEmbeddings(dimension=384)
    
    documents = [Node(text="Test", embedding=[0.1] * 384)]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_aretrieve",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_documents(documents)
    
    retriever = index.as_retriever(top_k=1)
    
    # Both should work
    results1 = await retriever.aretrieve("test")
    results2 = await retriever.aretrieve("test")
    
    assert len(results1) == len(results2)
