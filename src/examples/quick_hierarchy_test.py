"""
Quick test to verify hierarchical chunking returns correct chunk sizes.

This is a minimal example that:
1. Creates hierarchical chunks
2. Indexes them
3. Retrieves results
4. Verifies chunk sizes are parent chunks (not children)
"""

import asyncio
import os
from qdrant_client import QdrantClient

from fetchcraft import (
    HierarchicalChunkingStrategy,
    TextFileDocumentParser,
    QdrantVectorStore,
    VectorIndex,
    OpenAIEmbeddings,
    Chunk,
    SymNode
)


async def quick_test():
    """Quick verification that hierarchical chunking works correctly."""
    
    print("🔍 Quick Hierarchical Chunking Test\n")
    
    # Configuration
    PARENT_SIZE = 500
    CHILD_SIZES = [150, 75]
    
    # Sample document
    text = """
Machine learning is a method of data analysis that automates analytical 
model building. It is a branch of artificial intelligence based on the 
idea that systems can learn from data, identify patterns and make decisions 
with minimal human intervention.

Deep learning is a subset of machine learning in artificial intelligence 
that has networks capable of learning unsupervised from data that is 
unstructured or unlabeled. Also known as deep neural learning or deep 
neural network.

Natural language processing is a branch of artificial intelligence that 
helps computers understand, interpret and manipulate human language. 
NLP draws from many disciplines, including computer science and 
computational linguistics.
"""
    
    # Step 1: Create chunks
    print(f"1. Creating hierarchical chunks:")
    print(f"   Parent size: {PARENT_SIZE} chars")
    print(f"   Child sizes: {CHILD_SIZES}\n")
    
    chunker = HierarchicalChunkingStrategy(
        chunk_size=PARENT_SIZE,
        overlap=50,
        child_chunks=CHILD_SIZES,
        child_overlap=15
    )
    
    parser = TextFileDocumentParser(chunker=chunker)
    nodes = parser.parse(text.strip())
    
    parents = [n for n in nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    children = [n for n in nodes if isinstance(n, SymNode)]
    
    print(f"   ✓ Created {len(parents)} parents, {len(children)} children")
    print(f"   ✓ Total nodes to index: {len(nodes)}\n")
    
    # Verify parent sizes
    print(f"   Parent chunk sizes:")
    for i, p in enumerate(parents, 1):
        print(f"     Parent {i}: {len(p.text)} chars")
    print()
    
    # Step 2: Index
    print(f"2. Indexing in vector store...")
    
    client = QdrantClient(":memory:")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hierarchy_test",
        embeddings=embeddings,
        distance="Cosine"
    )
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id="test"
    )
    
    await vector_index.add_documents(nodes)
    print(f"   ✓ Indexed {len(nodes)} nodes\n")
    
    # Step 3: Retrieve
    print(f"3. Testing retrieval with resolve_parents=True...")
    
    retriever = vector_index.as_retriever(top_k=10)
    results = await retriever.aretrieve("What is machine learning?")
    
    print(f"   ✓ Retrieved {len(results)} results\n")
    
    # Step 4: Verify
    print(f"4. Verifying results:")
    
    all_parents = True
    sizes_correct = True
    
    for i, result in enumerate(results, 1):
        node = result.node
        is_parent = isinstance(node, Chunk) and not isinstance(node, SymNode)
        size = len(node.text)
        
        # Check type
        if not is_parent:
            all_parents = False
            print(f"   ✗ Result {i}: Child node (expected parent!)")
        else:
            print(f"   ✓ Result {i}: Parent chunk")
        
        # Check size
        size_ok = size >= PARENT_SIZE * 0.5  # Allow some variance
        if not size_ok:
            sizes_correct = False
            print(f"      Size: {size} chars (expected ~{PARENT_SIZE}) ✗")
        else:
            print(f"      Size: {size} chars ✓")
        
        # Show preview
        preview = node.text[:80].replace('\n', ' ')
        print(f"      Preview: {preview}...\n")
    
    # Final verdict
    print("="*70)
    if all_parents and sizes_correct:
        print("✅ SUCCESS: Hierarchical chunking works correctly!")
        print("   - All results are parent chunks (not children)")
        print("   - All sizes match parent chunk expectations")
        print("   - Child nodes were searched, but parents returned")
    else:
        print("⚠️  WARNING: Some issues detected")
        if not all_parents:
            print("   - Expected only parent chunks in results")
        if not sizes_correct:
            print("   - Some chunk sizes are unexpected")
    print("="*70)
    
    return all_parents and sizes_correct


async def main():
    try:
        success = await quick_test()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
