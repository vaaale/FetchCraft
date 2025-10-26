"""
Pytest configuration and shared fixtures for RAG Framework tests.
"""

import pytest
import asyncio
from qdrant_client import QdrantClient
from rag_framework import (
    Node,
    Chunk,
    QdrantVectorStore,
    VectorIndex
)


class MockEmbeddings:
    """Mock embeddings for testing."""
    
    def __init__(self, dimension=384):
        self._dimension = dimension
    
    @property
    def dimension(self):
        return self._dimension
    
    async def embed_query(self, text: str):
        """Return a mock embedding based on text length."""
        value = len(text) / 100.0
        return [value] * self._dimension
    
    async def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [await self.embed_query(text) for text in texts]
    
    async def aget_dimension(self):
        """Get dimension."""
        return self._dimension


@pytest.fixture
def mock_embeddings():
    """Fixture for mock embeddings."""
    return MockEmbeddings(dimension=384)


@pytest.fixture
def qdrant_client():
    """Fixture for in-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
async def vector_store(qdrant_client):
    """Fixture for vector store."""
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name="test_collection",
        vector_size=384
    )


@pytest.fixture
async def vector_index(vector_store, mock_embeddings):
    """Fixture for vector index."""
    return VectorIndex(
        vector_store=vector_store,
        embeddings=mock_embeddings
    )


@pytest.fixture
def sample_nodes():
    """Fixture for sample nodes."""
    return [
        Node(text="Python programming language", embedding=[0.1] * 384),
        Node(text="JavaScript web development", embedding=[0.2] * 384),
        Node(text="Machine learning and AI", embedding=[0.3] * 384),
    ]


@pytest.fixture
def sample_chunks():
    """Fixture for sample chunks."""
    return [
        Chunk.from_text(text="First chunk of text", chunk_index=0, embedding=[0.1] * 384),
        Chunk.from_text(text="Second chunk of text", chunk_index=1, embedding=[0.2] * 384),
        Chunk.from_text(text="Third chunk of text", chunk_index=2, embedding=[0.3] * 384),
    ]


# Configure asyncio event loop for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
