"""
Example demonstrating the use of Retriever abstraction.

This example shows how to:
1. Create a VectorIndex with documents
2. Create a retriever using as_retriever()
3. Retrieve documents using natural language queries
4. Customize retriever behavior
"""

import asyncio
from qdrant_client import QdrantClient

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Node, Chunk
from fetchcraft.retriever import VectorIndexRetriever
from fetchcraft.vector_store import QdrantVectorStore


async def basic_retriever_example():
    """Basic example of using VectorIndexRetriever."""
    
    print("="*60)
    print("Basic Retriever Example")
    print("="*60 + "\n")
    
    # Step 1: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )
    dimension = embeddings.dimension
    print(f"✓ Initialized embeddings (dimension: {dimension})\n")
    
    # Step 2: Create and populate index
    documents_text = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is widely used for web development and runs in browsers.",
        "Machine learning algorithms can learn patterns from data.",
        "Neural networks are inspired by biological neural networks.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "RAG combines retrieval and generation for better AI responses."
    ]
    
    print(f"Creating index with {len(documents_text)} documents...")
    
    # Create nodes WITHOUT embeddings - index will auto-generate them!
    nodes = [
        Node(
            text=text,
            metadata={"parsing": "example", "index": i}
        )
        for i, text in enumerate(documents_text)
    ]
    
    # Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="documents",
        embeddings=embeddings
    )
    
    # VectorIndex uses the vector store's embeddings
    index = VectorIndex(
        vector_store=vector_store
    )
    
    # Embeddings are auto-generated when adding documents!
    await index.add_nodes(DocumentNode, nodes)
    print(f"✓ Indexed {len(nodes)} documents (auto-embedded)\n")
    
    # Step 3: Create retriever from index (no embeddings param needed!)
    retriever = index.as_retriever(top_k=3)
    
    print(f"✓ Created retriever: {retriever}\n")
    
    # Step 4: Retrieve using natural language queries
    queries = [
        "programming languages",
        "artificial intelligence and learning",
        "understanding text with computers"
    ]
    
    print("Retrieving documents:\n")
    for query in queries:
        print(f"Query: '{query}'")
        
        # Simply pass the query text - embedding is handled internally!
        results = await retriever.aretrieve(query)
        
        print("  Top results:")
        for i, node_with_score in enumerate(results, 1):
            print(f"    {i}. [Score: {node_with_score.score:.3f}] {node_with_score.text[:60]}...")
        print()


async def retriever_configuration_example():
    """Example showing retriever configuration options."""
    
    print("\n" + "="*60)
    print("Retriever Configuration Example")
    print("="*60 + "\n")
    
    # Setup (abbreviated)
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    documents_text = [
        "RAG is a technique for retrieval-augmented generation.",
        "Vector databases store embeddings for similarity search.",
        "Semantic search finds documents by meaning, not keywords.",
        "Chunking splits documents into smaller pieces.",
        "Embeddings represent text as high-dimensional vectors."
    ]
    
    nodes = [
        Node(text=text, metadata={"topic": "rag"})
        for text in documents_text
    ]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="rag_docs",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_nodes(DocumentNode, nodes)
    
    print("✓ Setup complete\n")
    
    # Create retriever with custom configuration
    retriever = index.as_retriever(
        top_k=2,  # Return only top 2 results
        resolve_parents=True  # Resolve SymNode parents
    )
    
    print("Retrieving with top_k=2:")
    results = await retriever.aretrieve("What is RAG?")
    print(f"  Retrieved {len(results)} results\n")
    
    # Override top_k for a single query
    print("Retrieving with top_k=4 (override):")
    results = await retriever.aretrieve("What is RAG?", top_k=4)
    print(f"  Retrieved {len(results)} results\n")
    
    # Update retriever configuration
    retriever.update_config(top_k=5)
    print("Updated retriever config to top_k=5")
    results = await retriever.aretrieve("What is RAG?")
    print(f"  Retrieved {len(results)} results\n")


async def retriever_with_symnode_example():
    """Example showing retriever with SymNode parent resolution."""
    
    print("\n" + "="*60)
    print("Retriever with SymNode Example")
    print("="*60 + "\n")
    
    # Setup embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    # Create parent chunk
    parent_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines information 
    retrieval with text generation. It retrieves relevant documents and uses them 
    to generate more accurate and contextual responses.
    """.strip()
    
    parent_chunk = Chunk.from_text(
        text=parent_text,
        metadata={"type": "definition", "topic": "RAG"}
    )
    # No need to manually add embeddings - VectorIndex handles it!
    
    # Create SymNodes for granular search
    sub_texts = [
        "Retrieval-Augmented Generation (RAG) is a technique",
        "combines information retrieval with text generation",
        "retrieves relevant documents and uses them to generate responses"
    ]
    
    sym_nodes = parent_chunk.create_symbolic_nodes(sub_texts)
    # No need to manually add embeddings!
    
    # Setup index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hierarchical",
        embeddings=embeddings,
        document_class=Node
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    
    # Index parent first, then SymNodes (embeddings auto-generated!)
    await index.add_nodes(DocumentNode, [parent_chunk])
    await index.add_nodes(DocumentNode, sym_nodes)
    
    print(f"✓ Indexed 1 parent chunk and {len(sym_nodes)} SymNodes (auto-embedded)\n")
    
    # Create retriever with parent resolution enabled (default)
    retriever = index.as_retriever(
        top_k=3,
        resolve_parents=True
    )
    
    query = "What does RAG combine?"
    print(f"Query: '{query}'")
    print("With parent resolution (returns full context):")
    results = await retriever.aretrieve(query)
    
    for i, node in enumerate(results, 1):
        print(f"\n  {i}. Type: {node.__class__.__name__} [Score: {node.score:.3f}]")
        print(f"     Text: {node.text[:80]}...")
    
    # Create retriever without parent resolution
    retriever_no_resolve = index.as_retriever(
        top_k=3,
        resolve_parents=False
    )
    
    print(f"\n\nSame query without parent resolution (returns SymNodes):")
    results = await retriever_no_resolve.aretrieve(query)
    
    for i, node in enumerate(results, 1):
        print(f"\n  {i}. Type: {node.__class__.__name__} [Score: {node.score:.3f}]")
        print(f"     Text: {node.text[:80]}...")


async def direct_retriever_creation():
    """Example showing direct VectorIndexRetriever instantiation."""
    
    print("\n" + "="*60)
    print("Direct Retriever Creation")
    print("="*60 + "\n")
    
    # Setup
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    documents_text = [
        "Vector embeddings encode semantic meaning.",
        "Cosine similarity measures vector similarity.",
        "Dense vectors are better than sparse for semantics."
    ]
    
    nodes = [
        Node(text=text)
        for text in documents_text
    ]
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="vectors",
        embeddings=embeddings
    )
    index = VectorIndex(
        vector_store=vector_store
    )
    await index.add_nodes(DocumentNode, nodes)
    
    # Create retriever directly (not using as_retriever)
    retriever = VectorIndexRetriever(
        vector_index=index,
        top_k=2,
        resolve_parents=True
    )
    
    print(f"Created retriever: {retriever}\n")
    
    results = await retriever.aretrieve("semantic similarity")
    print(f"Query: 'semantic similarity'")
    print(f"Retrieved {len(results)} results:")
    for i, node in enumerate(results, 1):
        print(f"\n  {i}. Type: {node.__class__.__name__} [Score: {node.score:.3f}]")
        print(f"     Text: {node.text[:80]}...")


async def main():
    """Run all examples."""
    try:
        await basic_retriever_example()
        await retriever_configuration_example()
        await retriever_with_symnode_example()
        await direct_retriever_creation()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
