"""
Example: Using metadata filters with ChromaDB

This example demonstrates how to use metadata filters with ChromaDB vector store.
"""

import asyncio
import os
from pathlib import Path

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.vector_store import ChromaVectorStore
from fetchcraft.node import Chunk, DocumentNode
from fetchcraft.filters import (
    EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS,
    AND, OR, NOT
)

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


async def collect_results(async_iter, k: int = 100):
    """Helper to collect results from async iterator."""
    results = []
    async for item in async_iter:
        results.append(item)
        if len(results) >= k:
            break
    return results


async def example_chroma_basic_filters():
    """Demonstrate basic filters with ChromaDB."""
    if not CHROMADB_AVAILABLE:
        print("⚠️  ChromaDB not installed. Install with: pip install chromadb")
        return
    
    print("=" * 80)
    print("Example: ChromaDB with Metadata Filters")
    print("=" * 80)
    
    # Setup ChromaDB
    api_key = os.getenv("OPENAI_API_KEY", "sk-123")
    base_url = os.getenv("OPENAI_BASE_URL", "http://wingman:8000/v1")
    
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        api_key=api_key,
        base_url=base_url
    )
    
    client = chromadb.Client()  # In-memory client
    vector_store = ChromaVectorStore(
        client=client,
        collection_name="filter_demo_chroma",
        embeddings=embeddings
    )
    
    index = VectorIndex(vector_store=vector_store)
    
    # Create test data
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
    print(f"✓ Indexed {len(nodes)} documents with metadata")
    
    # Example 1: Equality filter
    print("\n1. Equality Filter (category == 'tutorial'):")
    results = await collect_results(index.search_by_text_iter("programming", filters=EQ("category", "tutorial")), k=5)
    for node, score in results:
        print(f"  - {node.text[:50]}... (category: {node.metadata['category']})")
    
    # Example 2: Greater than filter
    print("\n2. Greater Than Filter (year > 2022):")
    results = await collect_results(index.search_by_text_iter("programming", filters=GT("year", 2022)), k=5)
    for node, score in results:
        print(f"  - {node.text[:50]}... (year: {node.metadata['year']})")
    
    # Example 3: IN filter
    print("\n3. IN Filter (language in ['python', 'rust']):")
    results = await collect_results(index.search_by_text_iter("programming", filters=IN("language", ["python", "rust"])), k=5)
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']})")
    
    # Example 4: AND filter
    print("\n4. AND Filter (category == 'tutorial' AND year >= 2023):")
    results = await collect_results(
        index.search_by_text_iter(
            "programming",
            filters=AND(
                EQ("category", "tutorial"),
                GTE("year", 2023)
            )
        ),
        k=5
    )
    for node, score in results:
        print(f"  - {node.text[:50]}... (category: {node.metadata['category']}, year: {node.metadata['year']})")
    
    # Example 5: OR filter
    print("\n5. OR Filter (level == 'beginner' OR level == 'advanced'):")
    results = await collect_results(
        index.search_by_text_iter(
            "programming",
            filters=OR(
                EQ("level", "beginner"),
                EQ("level", "advanced")
            )
        ),
        k=5
    )
    for node, score in results:
        print(f"  - {node.text[:50]}... (level: {node.metadata['level']})")
    
    # Example 6: NOT filter
    print("\n6. NOT Filter (NOT category == 'systems'):")
    results = await collect_results(index.search_by_text_iter("programming", filters=NOT(EQ("category", "systems"))), k=5)
    for node, score in results:
        print(f"  - {node.text[:50]}... (category: {node.metadata['category']})")
    
    # Example 7: Complex nested filter
    print("\n7. Complex Nested Filter ((language == 'python' OR language == 'typescript') AND year >= 2023):")
    results = await collect_results(
        index.search_by_text_iter(
            "programming",
            filters=AND(
                OR(
                    EQ("language", "python"),
                    EQ("language", "typescript")
                ),
                GTE("year", 2023)
            )
        ),
        k=5
    )
    for node, score in results:
        print(f"  - {node.text[:50]}... (language: {node.metadata['language']}, year: {node.metadata['year']})")
    
    # Example 8: With Retriever (default filters)
    print("\n8. Retriever with Default Filters (category == 'tutorial'):")
    retriever = index.as_retriever(
        top_k=3,
        filters=EQ("category", "tutorial")
    )
    results = retriever.retrieve("programming")
    for result in results:
        print(f"  - {result.text[:50]}... (category: {result.metadata['category']})")


async def main():
    """Run ChromaDB filter examples."""
    print("\n🔍 ChromaDB Metadata Filter Examples\n")
    
    await example_chroma_basic_filters()
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)
    
    print("\n📚 ChromaDB Filter Support:")
    print("  ✅ All operators: EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS")
    print("  ✅ All conditions: AND, OR, NOT")
    print("  ✅ Nested filters")
    print("  ✅ Works with retrievers")
    print("  ✅ Translated to native ChromaDB format")


if __name__ == "__main__":
    asyncio.run(main())
