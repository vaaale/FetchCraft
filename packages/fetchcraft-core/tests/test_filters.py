"""
Tests for metadata filtering functionality.
"""

import pytest
import asyncio
from qdrant_client import QdrantClient

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.node import Chunk
from fetchcraft.filters import (
    EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS,
    AND, OR, NOT,
    FieldFilter, FilterOperator, CompositeFilter, FilterCondition
)


@pytest.fixture
def vector_store():
    """Create an in-memory vector store for testing."""
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    client = QdrantClient(":memory:")
    return QdrantVectorStore(
        client=client,
        collection_name="test_filters",
        embeddings=embeddings
    )


@pytest.fixture
def index(vector_store):
    """Create a vector index with test data."""
    return VectorIndex(vector_store=vector_store, index_id="test")


@pytest.fixture
async def populated_index(index):
    """Create and populate an index with test documents."""
    nodes = [
        Chunk.from_text(
            "Python programming tutorial for beginners",
            chunk_index=0,
            metadata={"language": "python", "year": 2023, "level": "beginner", "category": "tutorial"}
        ),
        Chunk.from_text(
            "Advanced JavaScript techniques",
            chunk_index=1,
            metadata={"language": "javascript", "year": 2024, "level": "advanced", "category": "guide"}
        ),
        Chunk.from_text(
            "Rust systems programming",
            chunk_index=2,
            metadata={"language": "rust", "year": 2022, "level": "advanced", "category": "systems"}
        ),
        Chunk.from_text(
            "Go for scalable backends",
            chunk_index=3,
            metadata={"language": "go", "year": 2023, "level": "intermediate", "category": "systems"}
        ),
        Chunk.from_text(
            "TypeScript for type safety",
            chunk_index=4,
            metadata={"language": "typescript", "year": 2024, "level": "intermediate", "category": "tutorial"}
        ),
    ]
    
    await index.add_nodes(nodes)
    return index


class TestBasicFilters:
    """Test basic field filters."""
    
    @pytest.mark.asyncio
    async def test_equality_filter(self, populated_index):
        """Test EQ operator."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=EQ("metadata.category", "tutorial")
        )
        
        assert len(results) <= 5
        for node, score in results:
            assert node.metadata["category"] == "tutorial"
    
    @pytest.mark.asyncio
    async def test_not_equal_filter(self, populated_index):
        """Test NE operator."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=NE("metadata.level", "beginner")
        )
        
        for node, score in results:
            assert node.metadata["level"] != "beginner"
    
    @pytest.mark.asyncio
    async def test_greater_than_filter(self, populated_index):
        """Test GT operator."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=GT("year", 2022)
        )
        
        for node, score in results:
            assert node.metadata["year"] > 2022
    
    @pytest.mark.asyncio
    async def test_greater_than_equal_filter(self, populated_index):
        """Test GTE operator."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=GTE("year", 2023)
        )
        
        for node, score in results:
            assert node.metadata["year"] >= 2023
    
    @pytest.mark.asyncio
    async def test_in_filter(self, populated_index):
        """Test IN operator."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=IN("language", ["python", "rust"])
        )
        
        for node, score in results:
            assert node.metadata["language"] in ["python", "rust"]


class TestCompositeFilters:
    """Test composite filters (AND, OR, NOT)."""
    
    @pytest.mark.asyncio
    async def test_and_filter(self, populated_index):
        """Test AND condition."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=AND(
                EQ("category", "tutorial"),
                GTE("year", 2023)
            )
        )
        
        for node, score in results:
            assert node.metadata["category"] == "tutorial"
            assert node.metadata["year"] >= 2023
    
    @pytest.mark.asyncio
    async def test_or_filter(self, populated_index):
        """Test OR condition."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=OR(
                EQ("level", "beginner"),
                EQ("level", "advanced")
            )
        )
        
        for node, score in results:
            assert node.metadata["level"] in ["beginner", "advanced"]
    
    @pytest.mark.asyncio
    async def test_not_filter(self, populated_index):
        """Test NOT condition."""
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=NOT(EQ("metadata.category", "systems"))
        )
        
        for node, score in results:
            assert node.metadata["category"] != "systems"
    
    @pytest.mark.asyncio
    async def test_nested_filters(self, populated_index):
        """Test nested filter combinations."""
        # (level == "beginner" OR level == "advanced") AND year >= 2023
        results = await populated_index.search_by_text(
            "programming",
            k=5,
            filters=AND(
                OR(
                    EQ("level", "beginner"),
                    EQ("level", "advanced")
                ),
                GTE("year", 2023)
            )
        )
        
        for node, score in results:
            assert node.metadata["level"] in ["beginner", "advanced"]
            assert node.metadata["year"] >= 2023


class TestRetrieverFilters:
    """Test filters with retriever interface."""
    
    @pytest.mark.asyncio
    async def test_retriever_with_per_query_filters(self, populated_index):
        """Test retriever.retrieve() with per-query filters."""
        retriever = populated_index.as_retriever(top_k=5)
        
        # Test with sync retrieve (uses aretrieve internally)
        results = retriever.retrieve(
            "programming",
            filters=EQ("metadata.language", "python")
        )
        
        assert len(results) > 0
        for result in results:
            assert result.node.metadata["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_retriever_with_default_filters_in_constructor(self, populated_index):
        """Test retriever with default filters set in constructor."""
        retriever = populated_index.as_retriever(
            top_k=5,
            filters=EQ("metadata.category", "tutorial")
        )
        
        # Retrieve without passing filters - should use default
        results = retriever.retrieve("programming")
        
        assert len(results) > 0
        for result in results:
            assert result.node.metadata["category"] == "tutorial"
    
    @pytest.mark.asyncio
    async def test_retriever_override_default_filters(self, populated_index):
        """Test overriding default filters in retrieve call."""
        retriever = populated_index.as_retriever(
            top_k=5,
            filters=EQ("metadata.category", "tutorial")  # Default filter
        )
        
        # Override with different filter
        results = retriever.retrieve(
            "programming",
            filters=EQ("metadata.category", "systems")  # Override
        )
        
        assert len(results) > 0
        for result in results:
            # Should use the override filter, not the default
            assert result.node.metadata["category"] == "systems"
    
    @pytest.mark.asyncio
    async def test_retriever_no_filters_specified(self, populated_index):
        """Test retriever without any filters."""
        retriever = populated_index.as_retriever(top_k=5)
        
        # No default filters, no per-query filters
        results = retriever.retrieve("programming")
        
        # Should return results from any category
        assert len(results) > 0
        categories = {result.node.metadata["category"] for result in results}
        assert len(categories) > 1  # Multiple categories present


class TestFilterRepresentation:
    """Test filter representation and serialization."""
    
    def test_field_filter_to_dict(self):
        """Test FieldFilter.to_dict()."""
        filter = EQ("category", "tutorial")
        
        filter_dict = filter.to_dict()
        
        assert filter_dict["type"] == "field"
        assert filter_dict["key"] == "category"
        assert filter_dict["operator"] == "=="
        assert filter_dict["value"] == "tutorial"
    
    def test_composite_filter_to_dict(self):
        """Test CompositeFilter.to_dict()."""
        filter = AND(
            EQ("category", "tutorial"),
            GTE("year", 2023)
        )
        
        filter_dict = filter.to_dict()
        
        assert filter_dict["type"] == "composite"
        assert filter_dict["condition"] == "AND"
        assert len(filter_dict["filters"]) == 2
    
    def test_filter_repr(self):
        """Test filter string representation."""
        filter = EQ("category", "tutorial")
        
        repr_str = repr(filter)
        
        assert "FieldFilter" in repr_str
        assert "category" in repr_str
        assert "tutorial" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
