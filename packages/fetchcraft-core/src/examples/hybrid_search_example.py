"""
Example demonstrating hybrid search with QdrantVectorStore.

Hybrid search combines:
- Dense vectors (semantic search) for understanding meaning
- Sparse vectors (BM25-style) for keyword matching

This provides the best of both worlds - semantic understanding plus precise keyword matching.
"""

import asyncio
import os
from qdrant_client import QdrantClient

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Node
from fetchcraft.vector_store import QdrantVectorStore


async def basic_hybrid_search_example():
    """Basic example of hybrid search with Qdrant."""
    
    print("="*80)
    print("Hybrid Search Example - Combining Dense + Sparse Vectors")
    print("="*80)
    
    # Sample documents about Python programming
    documents_text = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is widely used for web development and runs in browsers.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "Natural language processing enables computers to understand human language.",
        "The Python package manager pip makes it easy to install libraries.",
        "React is a JavaScript library for building user interfaces.",
        "TensorFlow is a popular framework for machine learning and deep learning."
    ]
    
    # Step 1: Initialize embeddings
    print("\n1. Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    print(f"   ✓ Embeddings ready (dimension: {embeddings.dimension})")
    
    # Step 2: Create vector store with HYBRID SEARCH enabled
    print("\n2. Creating Qdrant vector store with HYBRID SEARCH...")
    client = QdrantClient(":memory:")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hybrid_docs",
        embeddings=embeddings,
        enable_hybrid=True,  # 🔥 Enable hybrid search!
        fusion_method="rrf",  # Reciprocal Rank Fusion
        distance="Cosine"
    )
    print(f"   ✓ Vector store created with hybrid search enabled")
    print(f"   ✓ Fusion method: RRF (Reciprocal Rank Fusion)")
    
    # Step 3: Create index and add documents
    print("\n3. Indexing documents...")
    index = VectorIndex(vector_store=vector_store, index_id="hybrid_test")
    
    nodes = [Node(text=text, metadata={"index": i}) for i, text in enumerate(documents_text)]
    await index.add_nodes(DocumentNode, nodes)
    
    print(f"   ✓ Indexed {len(nodes)} documents")
    print(f"   ✓ Each document has both:")
    print(f"     - Dense vector (semantic meaning)")
    print(f"     - Sparse vector (keyword matching)")
    
    # Step 4: Perform hybrid search
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    
    queries = [
        "What is Python?",
        "pip package manager",
        "deep neural networks"
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)
        
        results = await index.search_by_text(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n  {i}. [Score: {score:.4f}]")
            print(f"     {doc.text}")
    
    print("\n" + "="*80)
    print("✅ Hybrid search completed!")
    print("="*80)


async def compare_hybrid_vs_dense():
    """Compare hybrid search vs dense-only search."""
    
    print("\n\n" + "="*80)
    print("COMPARISON: Hybrid vs Dense-Only Search")
    print("="*80)
    
    # Documents with specific keywords
    documents_text = [
        "Python pip is the package installer for Python packages.",
        "Machine learning models learn from data to make predictions.",
        "The pip command is used to install Python packages.",
        "Deep learning is a subset of machine learning using neural networks.",
        "JavaScript npm is similar to pip but for Node.js packages."
    ]
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    client = QdrantClient(":memory:")
    
    # Create two vector stores: one hybrid, one dense-only
    print("\n1. Creating two vector stores:")
    print("   - Dense-only (standard semantic search)")
    print("   - Hybrid (semantic + keyword matching)")
    
    # Dense-only store
    dense_store = QdrantVectorStore(
        client=client,
        collection_name="dense_only",
        embeddings=embeddings,
        enable_hybrid=False
    )
    dense_index = VectorIndex(vector_store=dense_store, index_id="dense")
    
    # Hybrid store
    hybrid_store = QdrantVectorStore(
        client=client,
        collection_name="hybrid",
        embeddings=embeddings,
        enable_hybrid=True,
        fusion_method="rrf"
    )
    hybrid_index = VectorIndex(vector_store=hybrid_store, index_id="hybrid")
    
    # Index same documents in both
    nodes = [Node(text=text) for text in documents_text]
    await dense_index.add_nodes(DocumentNode, nodes)
    await hybrid_index.add_nodes(DocumentNode, nodes)
    
    print(f"   ✓ Indexed {len(nodes)} documents in both stores")
    
    # Query with specific keyword
    query = "pip package installer"
    print(f"\n2. Searching for: '{query}'")
    print("   (This query has specific keywords: 'pip', 'package', 'installer')")
    
    # Search both
    print("\n" + "-"*80)
    print("Dense-Only Results (semantic search):")
    print("-"*80)
    dense_results = await dense_index.search_by_text(query, k=3)
    for i, (doc, score) in enumerate(dense_results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc.text}")
    
    print("\n" + "-"*80)
    print("Hybrid Results (semantic + keyword):")
    print("-"*80)
    hybrid_results = await hybrid_index.search_by_text(query, k=3)
    for i, (doc, score) in enumerate(hybrid_results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc.text}")
    
    print("\n" + "="*80)
    print("Analysis:")
    print("  • Dense-only: Good at finding semantically similar content")
    print("  • Hybrid: Better at finding exact keyword matches + semantic relevance")
    print("  • Hybrid often ranks keyword-matching results higher")
    print("="*80)


async def fusion_methods_comparison():
    """Compare RRF vs DBSF fusion methods."""
    
    print("\n\n" + "="*80)
    print("FUSION METHODS: RRF vs DBSF")
    print("="*80)
    
    documents_text = [
        "Python is a versatile programming language.",
        "Machine learning enables automated pattern recognition.",
        "Deep learning uses multi-layer neural networks.",
    ]
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    client = QdrantClient(":memory:")
    
    print("\n1. Testing both fusion methods:")
    print("   • RRF (Reciprocal Rank Fusion): Position-based")
    print("   • DBSF (Distribution-Based Score Fusion): Score-based")
    
    # Create stores with different fusion methods
    rrf_store = QdrantVectorStore(
        client=client,
        collection_name="hybrid_rrf",
        embeddings=embeddings,
        enable_hybrid=True,
        fusion_method="rrf"  # Reciprocal Rank Fusion
    )
    
    dbsf_store = QdrantVectorStore(
        client=client,
        collection_name="hybrid_dbsf",
        embeddings=embeddings,
        enable_hybrid=True,
        fusion_method="dbsf"  # Distribution-Based Score Fusion
    )
    
    # Index documents
    nodes = [Node(text=text) for text in documents_text]
    rrf_index = VectorIndex(vector_store=rrf_store)
    dbsf_index = VectorIndex(vector_store=dbsf_store)
    
    await rrf_index.add_nodes(DocumentNode, nodes)
    await dbsf_index.add_nodes(DocumentNode, nodes)
    
    print(f"   ✓ Indexed {len(nodes)} documents with both methods")
    
    query = "machine learning patterns"
    print(f"\n2. Query: '{query}'")
    
    print("\n" + "-"*80)
    print("RRF Results:")
    rrf_results = await rrf_index.search_by_text(query, k=3)
    for i, (doc, score) in enumerate(rrf_results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc.text}")
    
    print("\n" + "-"*80)
    print("DBSF Results:")
    dbsf_results = await dbsf_index.search_by_text(query, k=3)
    for i, (doc, score) in enumerate(dbsf_results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc.text}")
    
    print("\n" + "="*80)
    print("Fusion Method Comparison:")
    print("  • RRF: Better for balanced results, considers rank positions")
    print("  • DBSF: Better for score-based ranking, normalizes scores statistically")
    print("  • Both are effective - RRF is simpler, DBSF is more sophisticated")
    print("="*80)


async def main():
    """Run all examples."""
    try:
        await basic_hybrid_search_example()
        await compare_hybrid_vs_dense()
        await fusion_methods_comparison()
        
        print("\n" + "="*80)
        print("✅ ALL HYBRID SEARCH EXAMPLES COMPLETED")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. Hybrid search combines semantic + keyword matching")
        print("  2. Enable with: enable_hybrid=True")
        print("  3. Choose fusion method: 'rrf' or 'dbsf'")
        print("  4. Requires fastembed: pip install fastembed")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
