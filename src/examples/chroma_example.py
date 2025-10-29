"""
Example demonstrating ChromaDB vector store usage.

This example shows:
1. Creating a ChromaVectorStore
2. Indexing documents with ChromaDB
3. Performing semantic search
4. Using both in-memory and persistent modes
"""

import asyncio
import os
from pathlib import Path

# Import chromadb (will guide user if not installed)
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB is not installed. Install with: pip install chromadb")

from fetchcraft import (
    ChromaVectorStore,
    ChromaConfig,
    VectorIndex,
    OpenAIEmbeddings,
    DocumentNode,
    Chunk,
    SymNode
)
from fetchcraft.source import FilesystemDocumentSource
from fetchcraft.node_parser import SimpleNodeParser, HierarchicalNodeParser


async def example_basic_usage():
    """Basic usage example with in-memory ChromaDB."""
    
    print("="*80)
    print("Example 1: Basic ChromaDB Usage (In-Memory)")
    print("="*80)
    
    if not CHROMADB_AVAILABLE:
        print("\n❌ ChromaDB is required for this example.")
        print("Install with: pip install chromadb")
        return
    
    # Step 1: Set up embeddings
    print("\n1. Setting up embeddings...")
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    print(f"   ✓ Embeddings model: {embeddings.model}")
    print(f"   ✓ Dimension: {embeddings.dimension}")
    
    # Step 2: Create ChromaDB client and vector store
    print("\n2. Creating ChromaDB vector store...")
    
    client = chromadb.Client()  # In-memory mode
    
    vector_store = ChromaVectorStore(
        client=client,
        collection_name="demo_docs",
        embeddings=embeddings,
        distance="cosine"  # Options: "cosine", "l2", "ip"
    )
    
    print(f"   ✓ Created vector store with collection: {vector_store.collection_name}")
    print(f"   ✓ Distance metric: {vector_store.distance}")
    
    # Step 3: Create vector index
    print("\n3. Creating vector index...")
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id="demo_index"
    )
    
    print(f"   ✓ Index ID: {vector_index.index_id}")
    
    # Step 4: Prepare documents
    print("\n4. Preparing sample documents...")
    
    sample_documents = {
        "machine_learning.txt": """
Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on developing computer programs that can access data and use it 
to learn for themselves.
""",
        "deep_learning.txt": """
Deep learning is a subset of machine learning that uses neural networks with 
multiple layers. These networks can learn complex patterns in large amounts 
of data. Deep learning has revolutionized fields like computer vision and 
natural language processing.
""",
        "nlp.txt": """
Natural language processing (NLP) is a branch of artificial intelligence 
that focuses on the interaction between computers and human language. 
It enables computers to understand, interpret, and generate human language 
in valuable ways.
"""
    }
    
    # Parse and chunk documents
    parser = SimpleNodeParser(chunk_size=200, overlap=20)
    
    all_chunks = []
    for filename, content in sample_documents.items():
        # Create document node
        doc = DocumentNode.from_text(
            text=content.strip(),
            metadata={"filename": filename, "source": "demo"}
        )
        chunks = parser.get_nodes([doc])
        all_chunks.extend(chunks)
        print(f"   ✓ {filename}: {len(chunks)} chunks")
    
    print(f"\n   Total chunks: {len(all_chunks)}")
    
    # Step 5: Index documents
    print("\n5. Indexing documents...")
    
    await vector_index.add_nodes(all_chunks, show_progress=True)
    
    print(f"   ✓ Indexed {len(all_chunks)} chunks")
    
    # Step 6: Perform search
    print("\n6. Performing semantic search...")
    
    queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How does NLP work?"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = await vector_index.search_by_text(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            filename = doc.metadata.get('filename', 'unknown')
            preview = doc.text[:80].replace('\n', ' ')
            print(f"     {i}. Score: {score:.4f} | {filename}")
            print(f"        Preview: {preview}...")
    
    print("\n" + "="*80)
    print("✅ Example completed successfully!")
    print("="*80)


async def example_persistent_storage():
    """Example with persistent ChromaDB storage."""
    
    print("\n\n" + "="*80)
    print("Example 2: Persistent ChromaDB Storage")
    print("="*80)
    
    if not CHROMADB_AVAILABLE:
        print("\n❌ ChromaDB is required for this example.")
        return
    
    # Create persistent storage directory
    persist_dir = "./chroma_db"
    Path(persist_dir).mkdir(exist_ok=True)
    
    print(f"\n1. Using persistent storage at: {persist_dir}")
    
    # Set up embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    # Create persistent client
    client = chromadb.PersistentClient(path=persist_dir)
    
    vector_store = ChromaVectorStore(
        client=client,
        collection_name="persistent_docs",
        embeddings=embeddings,
        distance="cosine"
    )
    
    print(f"   ✓ Created persistent vector store")
    print(f"   ✓ Data will be saved to disk")
    
    # Use from_config method
    print("\n2. Alternative: Creating from config...")
    
    config = ChromaConfig(
        collection_name="config_collection",
        persist_directory=persist_dir,
        distance="cosine"
    )
    
    vector_store_from_config = ChromaVectorStore.from_config(config, embeddings=embeddings)
    
    print(f"   ✓ Created from config: {vector_store_from_config.collection_name}")
    
    print("\n" + "="*80)
    print("✅ Persistent storage example completed!")
    print("="*80)


async def example_hierarchical_chunking():
    """Example using hierarchical chunking with ChromaDB."""
    
    print("\n\n" + "="*80)
    print("Example 3: Hierarchical Chunking with ChromaDB")
    print("="*80)
    
    if not CHROMADB_AVAILABLE:
        print("\n❌ ChromaDB is required for this example.")
        return
    
    print("\n1. Setting up ChromaDB with hierarchical chunking...")
    
    # Set up embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    # Create ChromaDB client
    client = chromadb.Client()
    
    vector_store = ChromaVectorStore(
        client=client,
        collection_name="hierarchical_docs",
        embeddings=embeddings,
        distance="cosine"
    )
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id="hierarchical_index"
    )
    
    print("   ✓ Vector store and index created")
    
    # Use hierarchical chunking
    print("\n2. Creating hierarchical chunks...")
    
    parser = HierarchicalNodeParser(
        chunk_size=500,
        overlap=50,
        child_sizes=[150, 75],
        child_overlap=15
    )
    
    sample_text = """
Artificial intelligence (AI) is transforming the world. Machine learning, 
a subset of AI, enables systems to learn from data without explicit programming. 
Deep learning uses neural networks with multiple layers to process complex patterns.

Natural language processing allows computers to understand human language. 
Computer vision enables machines to interpret visual information from the world. 
These technologies are revolutionizing industries from healthcare to finance, 
creating new opportunities and challenges for society.

The future of AI holds immense potential. Advances in quantum computing may 
accelerate AI capabilities exponentially. Ethical considerations around AI 
development and deployment are becoming increasingly important as these systems 
become more powerful and widespread.
"""
    
    # Create document node
    doc = DocumentNode.from_text(text=sample_text.strip(), metadata={"source": "ai_overview"})
    nodes = parser.get_nodes([doc])
    
    parents = [n for n in nodes if not hasattr(n, 'is_symbolic') or not n.is_symbolic]
    children = [n for n in nodes if hasattr(n, 'is_symbolic') and n.is_symbolic]
    
    print(f"   ✓ Created {len(parents)} parent chunks")
    print(f"   ✓ Created {len(children)} child nodes")
    print(f"   ✓ Total nodes: {len(nodes)}")
    
    # Index all nodes
    print("\n3. Indexing hierarchical chunks...")
    
    await vector_index.add_nodes(nodes, show_progress=True)
    
    print(f"   ✓ Indexed all {len(nodes)} nodes")
    
    # Search with parent resolution
    print("\n4. Searching with parent resolution...")
    
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    results = await retriever.aretrieve("What is artificial intelligence?")
    
    print(f"\n   Retrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        is_parent = not (hasattr(result.node, 'is_symbolic') and result.node.is_symbolic)
        node_type = "Parent" if is_parent else "Child"
        size = len(result.node.text)
        print(f"     {i}. Type: {node_type} | Size: {size} chars | Score: {result.score:.4f}")
    
    print("\n" + "="*80)
    print("✅ Hierarchical chunking example completed!")
    print("="*80)


async def example_comparison():
    """Compare ChromaDB distance metrics."""
    
    print("\n\n" + "="*80)
    print("Example 4: Comparing Distance Metrics")
    print("="*80)
    
    if not CHROMADB_AVAILABLE:
        print("\n❌ ChromaDB is required for this example.")
        return
    
    print("\nComparing different distance metrics with the same data...\n")
    
    # Sample documents
    sample_text = "Machine learning is a subset of artificial intelligence."
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    parser = SimpleNodeParser(chunk_size=100)
    doc = DocumentNode.from_text(text=sample_text, metadata={"test": "distance_comparison"})
    chunks = parser.get_nodes([doc])
    
    # Test different distance metrics
    distance_metrics = ["cosine", "l2", "ip"]
    
    for distance in distance_metrics:
        print(f"Testing with {distance.upper()} distance:")
        
        client = chromadb.Client()
        vector_store = ChromaVectorStore(
            client=client,
            collection_name=f"test_{distance}",
            embeddings=embeddings,
            distance=distance
        )
        
        vector_index = VectorIndex(
            vector_store=vector_store,
            index_id=f"index_{distance}"
        )
        
        await vector_index.add_nodes(chunks)
        results = await vector_index.search_by_text("AI and ML", k=1)
        
        if results:
            score = results[0][1]
            print(f"  ✓ Collection created with {distance} metric")
            print(f"  ✓ Search score: {score:.4f}\n")
    
    print("="*80)
    print("✅ Distance metric comparison completed!")
    print("="*80)


async def main():
    """Run all examples."""
    
    if not CHROMADB_AVAILABLE:
        print("\n" + "="*80)
        print("⚠️  ChromaDB NOT INSTALLED")
        print("="*80)
        print("\nTo use ChromaVectorStore, install ChromaDB:")
        print("  pip install chromadb")
        print("\nFor more information: https://docs.trychroma.com/")
        print("="*80)
        return
    
    try:
        await example_basic_usage()
        await example_persistent_storage()
        await example_hierarchical_chunking()
        await example_comparison()
        
        print("\n\n" + "="*80)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features of ChromaVectorStore:")
        print("  ✓ In-memory and persistent storage modes")
        print("  ✓ Multiple distance metrics (cosine, l2, ip)")
        print("  ✓ Metadata filtering support")
        print("  ✓ Hierarchical chunking compatible")
        print("  ✓ Easy integration with VectorIndex")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
