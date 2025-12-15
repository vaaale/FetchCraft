"""
Tests for metadata filtering functionality with ChromaDB.
"""

import pytest
import asyncio

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Chunk, DocumentNode
from fetchcraft.filters import (
    EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS,
    AND, OR, NOT
)

# Check if ChromaDB is available
try:
    import chromadb
    from fetchcraft.vector_store import ChromaVectorStore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.fixture
def vector_store():
    """Create an in-memory ChromaDB vector store for testing."""
    if not CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB not installed")
    
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    client = chromadb.Client()
    return ChromaVectorStore(
        client=client,
        collection_name="test_filters_chroma",
        embeddings=embeddings
    )


@pytest.fixture
def index(vector_store):
    """Create a vector index with test data."""
    return VectorIndex(vector_store=vector_store, index_id="test_chroma")


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
    
    await index.add_nodes(None, nodes)
    return index


async def collect_results(async_iter, k: int = 100):
    """Helper to collect results from async iterator."""
    results = []
    async for item in async_iter:
        results.append(item)
        if len(results) >= k:
            break
    return results


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaBasicFilters:
    """Test basic field filters with ChromaDB."""
    
    @pytest.mark.asyncio
    async def test_equality_filter(self, populated_index):
        """Test EQ operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=EQ("metadata.category", "tutorial")
            ),
            k=5
        )
        
        assert len(results) <= 5
        for node, score in results:
            assert node.metadata["category"] == "tutorial"
    
    @pytest.mark.asyncio
    async def test_not_equal_filter(self, populated_index):
        """Test NE operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=NE("metadata.level", "beginner")
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["level"] != "beginner"
    
    @pytest.mark.asyncio
    async def test_greater_than_filter(self, populated_index):
        """Test GT operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=GT("year", 2022)
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["year"] > 2022
    
    @pytest.mark.asyncio
    async def test_greater_than_equal_filter(self, populated_index):
        """Test GTE operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=GTE("year", 2023)
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["year"] >= 2023
    
    @pytest.mark.asyncio
    async def test_less_than_filter(self, populated_index):
        """Test LT operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=LT("year", 2024)
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["year"] < 2024
    
    @pytest.mark.asyncio
    async def test_less_than_equal_filter(self, populated_index):
        """Test LTE operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=LTE("year", 2023)
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["year"] <= 2023
    
    @pytest.mark.asyncio
    async def test_in_filter(self, populated_index):
        """Test IN operator."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=IN("language", ["python", "rust"])
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["language"] in ["python", "rust"]


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaCompositeFilters:
    """Test composite filters (AND, OR, NOT) with ChromaDB."""
    
    @pytest.mark.asyncio
    async def test_and_filter(self, populated_index):
        """Test AND condition."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=AND(
                    EQ("category", "tutorial"),
                    GTE("year", 2023)
                )
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["category"] == "tutorial"
            assert node.metadata["year"] >= 2023
    
    @pytest.mark.asyncio
    async def test_or_filter(self, populated_index):
        """Test OR condition."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=OR(
                    EQ("level", "beginner"),
                    EQ("level", "advanced")
                )
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["level"] in ["beginner", "advanced"]
    
    @pytest.mark.asyncio
    async def test_not_filter(self, populated_index):
        """Test NOT condition."""
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=NOT(EQ("metadata.category", "systems"))
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["category"] != "systems"
    
    @pytest.mark.asyncio
    async def test_nested_filters(self, populated_index):
        """Test nested filter combinations."""
        # (level == "beginner" OR level == "advanced") AND year >= 2023
        results = await collect_results(
            populated_index.search_by_text_iter(
                "programming",
                filters=AND(
                    OR(
                        EQ("level", "beginner"),
                        EQ("level", "advanced")
                    ),
                    GTE("year", 2023)
                )
            ),
            k=5
        )
        
        for node, score in results:
            assert node.metadata["level"] in ["beginner", "advanced"]
            assert node.metadata["year"] >= 2023


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaRetrieverFilters:
    """Test filters with retriever interface and ChromaDB."""
    
    @pytest.mark.asyncio
    async def test_retriever_with_per_query_filters(self, populated_index):
        """Test retriever.retrieve() with per-query filters."""
        retriever = populated_index.as_retriever(top_k=5)
        
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


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaFilterTranslation:
    """Test ChromaDB filter translation."""
    
    def test_field_filter_translation(self, vector_store):
        """Test FieldFilter translation to ChromaDB format."""
        filter = EQ("category", "tutorial")
        
        chroma_filter = vector_store._translate_filter_to_chroma(filter)
        
        assert chroma_filter == {"category": {"$eq": "tutorial"}}
    
    def test_and_filter_translation(self, vector_store):
        """Test AND filter translation."""
        filter = AND(
            EQ("category", "tutorial"),
            GTE("year", 2023)
        )
        
        chroma_filter = vector_store._translate_filter_to_chroma(filter)
        
        assert "$and" in chroma_filter
        assert len(chroma_filter["$and"]) == 2
    
    def test_or_filter_translation(self, vector_store):
        """Test OR filter translation."""
        filter = OR(
            EQ("level", "beginner"),
            EQ("level", "advanced")
        )
        
        chroma_filter = vector_store._translate_filter_to_chroma(filter)
        
        assert "$or" in chroma_filter
        assert len(chroma_filter["$or"]) == 2
    
    def test_not_filter_translation(self, vector_store):
        """Test NOT filter translation."""
        filter = NOT(EQ("archived", True))
        
        chroma_filter = vector_store._translate_filter_to_chroma(filter)
        
        # ChromaDB uses operator negation: NOT(EQ) becomes NE
        assert "archived" in chroma_filter
        assert "$ne" in chroma_filter["archived"]
        assert chroma_filter["archived"]["$ne"] == True
    
    def test_raw_dict_passthrough(self, vector_store):
        """Test that raw dicts are passed through unchanged."""
        raw_filter = {"category": {"$eq": "tutorial"}}
        
        chroma_filter = vector_store._translate_filter_to_chroma(raw_filter)
        
        assert chroma_filter == raw_filter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
