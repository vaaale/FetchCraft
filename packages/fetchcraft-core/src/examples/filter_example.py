"""
Example: Using metadata filters for precise retrieval

This example demonstrates how to use metadata filters to refine search results.
"""

import asyncio
import os
from pathlib import Path

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.node import Chunk
from fetchcraft.filters import (
    EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS,
    AND, OR, NOT,
    FieldFilter, FilterOperator, CompositeFilter, FilterCondition
)
from qdrant_client import QdrantClient


async def example_basic_filters():
    """Demonstrate basic field filters."""
    print("=" * 80)
    print("Example 1: Basic Field Filters")
    print("=" * 80)
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY", "sk-123")
    base_url = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")
    
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key=api_key,
        base_url=base_url
    )
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="filter_demo",
        embeddings=embeddings
    )
    
    index = VectorIndex(vector_store=vector_store)
    
    # Create test data with metadata
    nodes = [
        Chunk.from_text(
            "Python is a high-level programming language.",
            chunk_index=0,
            metadata={"language": "python", "year": 2023, "category": "tutorial", "level": "beginner"}
        ),
        Chunk.from_text(
            "JavaScript is widely used for web development.",
            chunk_index=1,
            metadata={"language": "javascript", "year": 2023, "category": "tutorial", "level": "intermediate"}
        ),
        Chunk.from_text(
            "Rust provides memory safety without garbage collection.",
            chunk_index=2,
            metadata={"language": "rust", "year": 2022, "category": "systems", "level": "advanced"}
        ),
        Chunk.from_text(
            "Go is designed for building scalable systems.",
            chunk_index=3,
            metadata={"language": "go", "year": 2023, "category": "systems", "level": "intermediate"}
        ),
        Chunk.from_text(
            "TypeScript adds type safety to JavaScript.",
            chunk_index=4,
            metadata={"language": "typescript", "year": 2024, "category": "tutorial", "level": "advanced"}
        ),
    ]
    
    await index.add_nodes(DocumentNode, nodes)
    print(f"✓ Indexed {len(nodes)} documents with metadata")
    
    # Example 1: Equality filter
    print("\n1. Equality Filter (category == 'tutorial'):")
    filter_eq = EQ("category", "tutorial")
    results = await index.search_by_text("programming", k=5, filters=filter_eq)
    for node, score in results:
        print(f"  - {node.text[:50]}... (category: {node.metadata['category']})")
    
    # Example 2: Greater than filter
    print("\n2. Greater Than Filter (year > 2022):")
    filter_gt = GT("year", 2022)
    results = await index.search_by_text("programming", k=5, filters=filter_gt)
    for node, score in results:
        print(f"  - {node.text[:50]}... (year: {node.metadata['year']})")
    
    # Example 3: IN filter
    print("\n3. IN Filter (language in ['python', 'rust']):")
    filter_in = IN("language", ["python", "rust"])
    results = await index.search_by_text("programming", k=5, filters=filter_in)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']})")
    
    # Example 4: Contains filter
    print("\n4. Contains Filter (text contains 'type'):")
    filter_contains = CONTAINS("language", "type")
    results = await index.search_by_text("programming", k=5, filters=filter_contains)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']})")


async def example_composite_filters():
    """Demonstrate composite filters (AND, OR, NOT)."""
    print("\n" + "=" * 80)
    print("Example 2: Composite Filters")
    print("=" * 80)
    
    # Setup (same as above)
    api_key = os.getenv("OPENAI_API_KEY", "sk-123")
    base_url = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")
    
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key=api_key,
        base_url=base_url
    )
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="composite_filter_demo",
        embeddings=embeddings
    )
    
    index = VectorIndex(vector_store=vector_store)
    
    # Create test data
    nodes = [
        Chunk.from_text(
            "Machine learning with Python for beginners.",
            chunk_index=0,
            metadata={"topic": "ml", "language": "python", "year": 2023, "level": "beginner"}
        ),
        Chunk.from_text(
            "Advanced deep learning techniques.",
            chunk_index=1,
            metadata={"topic": "dl", "language": "python", "year": 2024, "level": "advanced"}
        ),
        Chunk.from_text(
            "Web development with JavaScript.",
            chunk_index=2,
            metadata={"topic": "webdev", "language": "javascript", "year": 2023, "level": "intermediate"}
        ),
        Chunk.from_text(
            "Data science fundamentals in Python.",
            chunk_index=3,
            metadata={"topic": "datascience", "language": "python", "year": 2024, "level": "beginner"}
        ),
        Chunk.from_text(
            "Building APIs with Rust.",
            chunk_index=4,
            metadata={"topic": "api", "language": "rust", "year": 2023, "level": "advanced"}
        ),
    ]
    
    await index.add_nodes(DocumentNode, nodes)
    print(f"✓ Indexed {len(nodes)} documents with metadata")
    
    # Example 1: AND filter
    print("\n1. AND Filter (language == 'python' AND year >= 2024):")
    filter_and = AND(
        EQ("language", "python"),
        GTE("year", 2024)
    )
    results = await index.search_by_text("learning", k=5, filters=filter_and)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']}, year: {node.metadata['year']})")
    
    # Example 2: OR filter
    print("\n2. OR Filter (level == 'beginner' OR level == 'advanced'):")
    filter_or = OR(
        EQ("level", "beginner"),
        EQ("level", "advanced")
    )
    results = await index.search_by_text("learning", k=5, filters=filter_or)
    for node, score in results:
        print(f"  - {node.text[:50]}... (level: {node.metadata['level']})")
    
    # Example 3: NOT filter
    print("\n3. NOT Filter (NOT language == 'python'):")
    filter_not = NOT(EQ("language", "python"))
    results = await index.search_by_text("programming", k=5, filters=filter_not)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']})")
    
    # Example 4: Nested filters
    print("\n4. Nested Filter ((language == 'python' AND level == 'beginner') OR year == 2024):")
    filter_nested = OR(
        AND(
            EQ("language", "python"),
            EQ("level", "beginner")
        ),
        EQ("year", 2024)
    )
    results = await index.search_by_text("learning", k=5, filters=filter_nested)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']}, level: {node.metadata['level']}, year: {node.metadata['year']})")


async def example_retriever_with_filters():
    """Demonstrate using filters with retriever."""
    print("\n" + "=" * 80)
    print("Example 3: Retriever with Filters")
    print("=" * 80)
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY", "sk-123")
    base_url = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")
    
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key=api_key,
        base_url=base_url
    )
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="retriever_filter_demo",
        embeddings=embeddings
    )
    
    index = VectorIndex(vector_store=vector_store)
    
    # Create test data
    nodes = [
        Chunk.from_text(
            "Introduction to machine learning algorithms.",
            chunk_index=0,
            metadata={"category": "ai", "difficulty": "easy", "tags": ["ml", "intro"]}
        ),
        Chunk.from_text(
            "Advanced neural network architectures.",
            chunk_index=1,
            metadata={"category": "ai", "difficulty": "hard", "tags": ["nn", "advanced"]}
        ),
        Chunk.from_text(
            "Web development basics with HTML and CSS.",
            chunk_index=2,
            metadata={"category": "webdev", "difficulty": "easy", "tags": ["html", "css"]}
        ),
        Chunk.from_text(
            "Database design principles.",
            chunk_index=3,
            metadata={"category": "database", "difficulty": "medium", "tags": ["db", "design"]}
        ),
    ]
    
    await index.add_nodes(DocumentNode, nodes)
    print(f"✓ Indexed {len(nodes)} documents with metadata")
    
    # Method 1: Pass filter to retrieve call
    print("\n1. Retriever WITHOUT default filters (pass filters per query):")
    retriever = index.as_retriever(top_k=3)
    results = retriever.retrieve("learning", filters=EQ("category", "ai"))
    for result in results:
        print(f"  - {result.text[:50]}... (category: {result.metadata['category']})")
    
    # Method 2: Retriever WITH default filters in constructor
    print("\n2. Retriever WITH default filters (category == 'ai'):")
    retriever_with_filters = index.as_retriever(
        top_k=3,
        filters=EQ("category", "ai")  # Default filter applied to all queries
    )
    results = retriever_with_filters.retrieve("learning")  # No filter param needed
    for result in results:
        print(f"  - {result.text[:50]}... (category: {result.metadata['category']})")
    
    # Method 3: Override default filters
    print("\n3. Override default filters per query:")
    results = retriever_with_filters.retrieve(
        "development",
        filters=EQ("category", "webdev")  # Override default filter
    )
    for result in results:
        print(f"  - {result.text[:50]}... (category: {result.metadata['category']})")
    
    # Method 4: Complex default filter
    print("\n4. Retriever with complex default filter (category == 'ai' AND difficulty == 'easy'):")
    retriever_complex = index.as_retriever(
        top_k=3,
        filters=AND(
            EQ("category", "ai"),
            EQ("difficulty", "easy")
        )
    )
    results = retriever_complex.retrieve("learning")
    for result in results:
        print(f"  - {result.text[:50]}... (category: {result.metadata['category']}, difficulty: {result.metadata['difficulty']})")


async def main():
    """Run all filter examples."""
    print("\n🔍 Metadata Filter Examples\n")
    
    # Example 1: Basic filters
    await example_basic_filters()
    
    # Example 2: Composite filters
    await example_composite_filters()
    
    # Example 3: Retriever with filters
    await example_retriever_with_filters()
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)
    
    print("\n📚 Filter Capabilities:")
    print("  • Operators: EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS")
    print("  • Conditions: AND, OR, NOT")
    print("  • Nested filters for complex queries")
    print("  • Works with any metadata field")
    print("  • Translated to native vector store filters")
    print("\n💡 Use Cases:")
    print("  • Filter by date ranges")
    print("  • Filter by categories or tags")
    print("  • Exclude certain content types")
    print("  • Combine multiple conditions")
    print("  • Build dynamic, user-driven queries")


if __name__ == "__main__":
    asyncio.run(main())
