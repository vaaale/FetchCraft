"""
Example demonstrating the use of SymNode for hierarchical node relationships.

This example shows how to:
1. Create parent Chunk nodes with full context
2. Create smaller SymNode instances that reference the parent
3. Index the SymNodes for granular semantic search
4. Automatically resolve to parent nodes during retrieval
"""

import asyncio
from typing import List
from qdrant_client import QdrantClient

from fetchcraft import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Chunk,
    SymNode,
    Node
)


async def basic_symnode_example():
    """Basic example of creating and using SymNode with parent resolution."""
    
    print("="*60)
    print("Basic SymNode Example")
    print("="*60 + "\n")
    
    # Step 1: Create a large chunk (parent)
    long_text = """
    Machine learning is a subset of artificial intelligence that enables computers 
    to learn from data without being explicitly programmed. It uses algorithms that 
    iteratively learn from data to improve predictions and find patterns. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers to analyze 
    various factors with a structure similar to the human neural system.
    """.strip()
    
    big_chunk = Chunk.from_text(
        text=long_text,
        chunk_index=0,
        start_char_idx=0,
        end_char_idx=len(long_text),
        metadata={"topic": "machine_learning", "source": "textbook"}
    )
    
    print(f"Created parent chunk (ID: {big_chunk.id[:8]}...):")
    print(f"  Length: {len(big_chunk.text)} chars")
    print(f"  Preview: {big_chunk.text[:60]}...\n")
    
    # Step 2: Create smaller symbolic nodes from the big chunk
    small_chunk1 = long_text[0:120]  # First part about ML
    small_chunk2 = long_text[120:]    # Second part about deep learning
    
    sym_node1 = SymNode.create(
        text=small_chunk1,
        parent_id=big_chunk.id,
        metadata=big_chunk.metadata.copy()
    )
    
    sym_node2 = SymNode.create(
        text=small_chunk2,
        parent_id=big_chunk.id,
        metadata=big_chunk.metadata.copy()
    )
    
    print(f"Created SymNode 1 (ID: {sym_node1.id[:8]}...):")
    print(f"  Length: {len(sym_node1.text)} chars")
    print(f"  Parent ID: {sym_node1.parent_id[:8]}...")
    print(f"  Text: {sym_node1.text[:60]}...")
    print()
    
    print(f"Created SymNode 2 (ID: {sym_node2.id[:8]}...):")
    print(f"  Length: {len(sym_node2.text)} chars")
    print(f"  Parent ID: {sym_node2.parent_id[:8]}...")
    print(f"  Text: {sym_node2.text[:60]}...")
    print()
    
    # Step 3: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )
    
    dimension = embeddings.dimension
    print(f"✓ Initialized embeddings (dimension: {dimension})\n")
    
    # Step 4: Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hierarchical_docs",
        embeddings=embeddings,
        document_class=Node  # Use base Node to support both Chunk and SymNode
    )
    
    # VectorIndex uses the vector store's embeddings - will auto-generate embeddings!
    index = VectorIndex(
        vector_store=vector_store
    )
    
    # Step 5: Add parent chunk first (must exist before SymNodes can reference it)
    # Embeddings auto-generated!
    parent_ids = await index.add_documents([big_chunk])
    print(f"✓ Indexed parent chunk (ID: {parent_ids[0][:8]}...)\n")
    
    # Step 6: Add symbolic nodes to the index (embeddings auto-generated!)
    sym_ids = await index.add_documents([sym_node1, sym_node2])
    print(f"✓ Indexed {len(sym_ids)} SymNodes (all embeddings auto-generated!)\n")
    
    # Step 7: Search and observe parent resolution
    query = "What is deep learning?"
    print(f"Query: '{query}'\n")
    
    # Search with parent resolution (default behavior)
    print("--- With Parent Resolution (default) ---")
    results = await index.search_by_text(query, k=2, resolve_parents=True)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.3f}]")
        print(f"   Type: {doc.__class__.__name__}")
        print(f"   ID: {doc.id[:8]}...")
        print(f"   Text: {doc.text[:80]}...")
        print()
    
    # Search without parent resolution to see the SymNodes
    print("--- Without Parent Resolution ---")
    results_no_resolve = await index.search_by_text(query, k=2, resolve_parents=False)
    
    for i, (doc, score) in enumerate(results_no_resolve, 1):
        print(f"{i}. [Score: {score:.3f}]")
        print(f"   Type: {doc.__class__.__name__}")
        print(f"   ID: {doc.id[:8]}...")
        if hasattr(doc, 'parent_id') and doc.parent_id:
            print(f"   Parent ID: {doc.parent_id[:8]}...")
        print(f"   Text: {doc.text[:80]}...")
        print()


async def hierarchical_chunking_example():
    """
    Example showing how to use SymNode for hierarchical chunking strategy.
    Large chunks for context, small SymNodes for precise semantic matching.
    """
    
    print("\n" + "="*60)
    print("Hierarchical Chunking with SymNode")
    print("="*60 + "\n")
    
    # Sample documents
    documents = [
        {
            "title": "Python Programming",
            "text": """Python is a high-level, interpreted programming language known for its 
            simplicity and readability. It supports multiple programming paradigms including 
            procedural, object-oriented, and functional programming. Python has a large standard 
            library and active community."""
        },
        {
            "title": "JavaScript Basics",
            "text": """JavaScript is a lightweight, interpreted programming language primarily 
            used for web development. It enables interactive web pages and is an essential part 
            of web applications. JavaScript can run both in browsers and on servers using 
            Node.js runtime."""
        }
    ]
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )

    # Setup vector store
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hierarchical",
        embeddings=embeddings,
        document_class=Node
    )
    # VectorIndex uses the vector store's embeddings for auto-generation
    index = VectorIndex(
        vector_store=vector_store
    )
    
    # Process each document
    all_parent_chunks: List[Chunk] = []
    all_sym_nodes: List[SymNode] = []
    
    for doc in documents:
        # Create parent chunk with full context
        parent_chunk = Chunk.from_text(
            text=doc["text"].strip(),
            metadata={"title": doc["title"]}
        )
        all_parent_chunks.append(parent_chunk)
        
        # Split into smaller pieces for SymNodes
        text = doc["text"].strip()
        sentences = text.split(". ")
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Only create SymNode for substantial sentences
                sym_node = SymNode.create(
                    text=sentence.strip() + ".",
                    parent_id=parent_chunk.id,
                    metadata={"title": doc["title"], "parent_chunk_id": parent_chunk.id}
                )
                all_sym_nodes.append(sym_node)
    
    print(f"Created {len(all_parent_chunks)} parent chunks")
    print(f"Created {len(all_sym_nodes)} SymNodes\n")
    
    # Index parent chunks first (embeddings auto-generated!)
    await index.add_documents(all_parent_chunks)
    print(f"✓ Indexed {len(all_parent_chunks)} parent chunks")
    
    # Index SymNodes (embeddings auto-generated!)
    await index.add_documents(all_sym_nodes)
    print(f"✓ Indexed {len(all_sym_nodes)} SymNodes (all embeddings auto-generated!)\n")
    
    # Search examples
    queries = [
        "programming paradigms",
        "web development and browsers",
        "server-side JavaScript"
    ]
    
    print("Searching with automatic parent resolution:\n")
    for query in queries:
        print(f"Query: '{query}'")
        results = await index.search_by_text(query, k=1)
        
        if results:
            doc, score = results[0]
            print(f"  → [Score: {score:.3f}] {doc.metadata.get('title', 'N/A')}")
            print(f"    Type: {doc.__class__.__name__}")
            print(f"    Text: {doc.text[:100]}...")
        print()


async def main():
    """Run all examples."""
    try:
        await basic_symnode_example()
        await hierarchical_chunking_example()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
