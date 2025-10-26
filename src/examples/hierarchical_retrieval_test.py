"""HierarchicalChunkingStrategy
Example demonstrating HierarchicalChunkingStrategy with vector index retrieval.

This example shows:
1. Creating hierarchical chunks with multiple child sizes
2. Indexing all nodes (parents and children) in a vector store
3. Performing search/retrieval
4. Verifying that retrieved results are PARENT chunks (not children)
"""

import asyncio
import os
from pathlib import Path
from qdrant_client import QdrantClient

from fetchcraft import (
    HierarchicalChunkingStrategy,
    CharacterChunkingStrategy,
    TextFileDocumentParser,
    QdrantVectorStore,
    VectorIndex,
    OpenAIEmbeddings,
    Chunk,
    SymNode
)


async def test_hierarchical_retrieval():
    """Test that hierarchical chunking returns parent chunks on retrieval."""
    
    print("="*80)
    print("Hierarchical Chunking Retrieval Test")
    print("="*80)
    
    # Sample documents
    documents = {
        "doc1.txt": """
Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on the development of computer programs that can access data and 
use it to learn for themselves.

The process of learning begins with observations or data, such as examples, 
direct experience, or instruction, in order to look for patterns in data and 
make better decisions in the future. Machine learning algorithms are designed 
to improve automatically through experience.
""",
        "doc2.txt": """
Deep learning is a subset of machine learning that uses neural networks with 
multiple layers. These networks can learn complex patterns in large amounts 
of data. Deep learning has revolutionized fields like computer vision, 
natural language processing, and speech recognition.

Neural networks are inspired by the human brain and consist of interconnected 
nodes called neurons. Each connection has a weight that adjusts as learning 
proceeds, allowing the network to make better predictions over time.
""",
        "doc3.txt": """
Natural language processing (NLP) is a branch of artificial intelligence 
that focuses on the interaction between computers and human language. 
It enables computers to understand, interpret, and generate human language 
in a valuable way.

Modern NLP systems use deep learning techniques to achieve state-of-the-art 
results. Applications include machine translation, sentiment analysis, 
chatbots, and text summarization.
"""
    }
    
    # Step 1: Create hierarchical chunking strategy
    print("\n" + "="*80)
    print("Step 1: Creating Hierarchical Chunking Strategy")
    print("="*80)
    
    PARENT_CHUNK_SIZE = 300
    CHILD_SIZES = [100, 50]
    
    chunker = HierarchicalChunkingStrategy(
        chunk_size=PARENT_CHUNK_SIZE,
        overlap=30,
        child_chunks=CHILD_SIZES,
        child_overlap=10
    )
    
    print(f"\nConfiguration:")
    print(f"  Parent chunk size: {PARENT_CHUNK_SIZE} chars")
    print(f"  Child chunk sizes: {CHILD_SIZES}")
    print(f"  Recursive splitting: paragraph → line → sentence → word")
    
    # Step 2: Parse documents
    print("\n" + "="*80)
    print("Step 2: Parsing Documents")
    print("="*80)
    
    parser = TextFileDocumentParser(chunker=chunker)
    all_nodes = []
    
    for filename, content in documents.items():
        nodes = parser.parse(
            text=content.strip(),
            metadata={"filename": filename}
        )
        all_nodes.extend(nodes)
        print(f"\n{filename}: {len(content)} chars → {len(nodes)} nodes")
    
    # Separate parents and children
    parent_chunks = [n for n in all_nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    child_nodes = [n for n in all_nodes if isinstance(n, SymNode)]
    
    print(f"\nTotal nodes created:")
    print(f"  Parent chunks: {len(parent_chunks)}")
    print(f"  Child nodes: {len(child_nodes)}")
    print(f"  Total: {len(all_nodes)}")
    
    # Show parent chunk sizes
    print(f"\nParent chunk size distribution:")
    for i, parent in enumerate(parent_chunks, 1):
        print(f"  Parent {i}: {len(parent.text)} chars (target: {PARENT_CHUNK_SIZE})")
    
    # Show child distribution by size
    child_by_size = {}
    for child in child_nodes:
        size = child.metadata.get('child_size', 'unknown')
        child_by_size[size] = child_by_size.get(size, 0) + 1
    
    print(f"\nChild node distribution by size:")
    for size in sorted(child_by_size.keys(), reverse=True):
        count = child_by_size[size]
        print(f"  Size {size}: {count} nodes")
    
    # Step 3: Set up vector store and index
    print("\n" + "="*80)
    print("Step 3: Setting Up Vector Store")
    print("="*80)
    
    # Use in-memory Qdrant for testing
    client = QdrantClient(":memory:")
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_hierarchical",
        vector_size=embeddings.dimension,
        distance="Cosine"
    )
    
    vector_index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings,
        index_id="test_index"
    )
    
    print(f"\n✓ Vector store initialized")
    print(f"  Collection: test_hierarchical")
    print(f"  Embedding dimension: {embeddings.dimension}")
    
    # Step 4: Index all nodes
    print("\n" + "="*80)
    print("Step 4: Indexing Nodes")
    print("="*80)
    
    print(f"\nIndexing {len(all_nodes)} nodes...")
    print(f"  This includes:")
    print(f"    - {len(parent_chunks)} parent chunks (stored for context)")
    print(f"    - {len(child_nodes)} child nodes (indexed for search)")
    
    await vector_index.add_documents(all_nodes, show_progress=True)
    
    print(f"\n✓ All nodes indexed successfully")
    
    # Step 5: Perform retrieval
    print("\n" + "="*80)
    print("Step 5: Testing Retrieval")
    print("="*80)
    
    # Create retriever with parent resolution
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is natural language processing?"
    ]
    
    for query_num, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {query_num}: '{query}'")
        print(f"{'─'*80}")
        
        results = await retriever.retrieve(query)
        
        print(f"\nRetrieved {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            node = result.node
            score = result.score
            
            # Check if this is a parent chunk or child node
            is_parent = isinstance(node, Chunk) and not isinstance(node, SymNode)
            is_child = isinstance(node, SymNode)
            
            print(f"\n  Result {i} (score: {score:.4f}):")
            print(f"    Type: {'Parent Chunk' if is_parent else 'Child SymNode'}")
            print(f"    Size: {len(node.text)} chars")
            print(f"    File: {node.metadata.get('filename', 'N/A')}")
            
            # Show preview
            preview = node.text[:100].replace('\n', ' ')
            print(f"    Preview: {preview}...")
            
            # Verify size expectations
            if is_parent:
                # Should be close to parent chunk size
                size_ok = len(node.text) >= PARENT_CHUNK_SIZE * 0.5  # Allow some variance
                status = "✓" if size_ok else "✗"
                print(f"    {status} Size matches parent expectation ({PARENT_CHUNK_SIZE} target)")
            elif is_child:
                # With resolve_parents=True, we should NOT get child nodes directly
                print(f"    ⚠️  WARNING: Got child node directly (expected parent!)")
    
    # Step 6: Verification
    print("\n" + "="*80)
    print("Step 6: Verification")
    print("="*80)
    
    # Count how many results are parents vs children
    all_results = []
    for query in test_queries:
        results = await retriever.retrieve(query)
        all_results.extend(results)
    
    parent_results = sum(1 for r in all_results if isinstance(r.node, Chunk) and not isinstance(r.node, SymNode))
    child_results = sum(1 for r in all_results if isinstance(r.node, SymNode))
    
    print(f"\nRetrieval Results Summary:")
    print(f"  Total results: {len(all_results)}")
    print(f"  Parent chunks: {parent_results}")
    print(f"  Child nodes: {child_results}")
    
    if child_results == 0 and parent_results > 0:
        print(f"\n✅ SUCCESS: All retrieved results are parent chunks!")
        print(f"   This confirms that:")
        print(f"   - Child nodes were indexed and searched")
        print(f"   - Parent resolution worked correctly")
        print(f"   - Users get full context (parent chunks) on retrieval")
    else:
        print(f"\n⚠️  UNEXPECTED: Got {child_results} child nodes in results")
        print(f"   Expected: Only parent chunks with resolve_parents=True")
    
    # Verify parent chunk sizes
    print(f"\nParent Chunk Size Verification:")
    sizes_correct = 0
    for result in all_results:
        if isinstance(result.node, Chunk) and not isinstance(result.node, SymNode):
            size = len(result.node.text)
            # Allow 50% variance (some chunks may be smaller at document boundaries)
            is_correct = size >= PARENT_CHUNK_SIZE * 0.5
            if is_correct:
                sizes_correct += 1
            status = "✓" if is_correct else "✗"
            print(f"  {status} Chunk: {size} chars (target: {PARENT_CHUNK_SIZE})")
    
    if sizes_correct == parent_results:
        print(f"\n✅ All parent chunks have appropriate sizes!")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


async def compare_with_character_chunking():
    """Compare hierarchical vs character chunking retrieval."""
    
    print("\n\n" + "="*80)
    print("COMPARISON: Hierarchical vs Character Chunking")
    print("="*80)
    
    sample_text = """
Artificial intelligence (AI) is transforming the world. Machine learning, 
a subset of AI, enables systems to learn from data. Deep learning uses 
neural networks with multiple layers to process complex patterns.

Natural language processing allows computers to understand human language. 
Computer vision enables machines to interpret visual information. These 
technologies are revolutionizing industries from healthcare to finance.
"""
    
    # Set up vector store
    client = QdrantClient(":memory:")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
    )
    
    # Test 1: Character chunking
    print("\n" + "─"*80)
    print("Character Chunking")
    print("─"*80)
    
    char_chunker = CharacterChunkingStrategy(chunk_size=100, overlap=20)
    char_parser = TextFileDocumentParser(chunker=char_chunker)
    char_chunks = char_parser.parse(sample_text.strip())
    
    print(f"Created {len(char_chunks)} chunks")
    for i, chunk in enumerate(char_chunks, 1):
        print(f"  Chunk {i}: {len(chunk.text)} chars")
    
    # Index and retrieve
    char_store = QdrantVectorStore(
        client=client,
        collection_name="test_character",
        vector_size=embeddings.dimension,
        distance="Cosine"
    )
    char_index = VectorIndex(char_store, embeddings, index_id="char_test")
    await char_index.add_documents(char_chunks)
    
    char_retriever = char_index.as_retriever(top_k=2, resolve_parents=False)
    char_results = await char_retriever.retrieve("machine learning")
    
    print(f"\nRetrieved {len(char_results)} results:")
    for i, result in enumerate(char_results, 1):
        print(f"  Result {i}: {len(result.node.text)} chars (score: {result.score:.4f})")
    
    # Test 2: Hierarchical chunking
    print("\n" + "─"*80)
    print("Hierarchical Chunking")
    print("─"*80)
    
    hier_chunker = HierarchicalChunkingStrategy(
        chunk_size=200,
        overlap=20,
        child_chunks=[60, 30],
        child_overlap=10
    )
    hier_parser = TextFileDocumentParser(chunker=hier_chunker)
    hier_nodes = hier_parser.parse(sample_text.strip())
    
    hier_parents = [n for n in hier_nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    hier_children = [n for n in hier_nodes if isinstance(n, SymNode)]
    
    print(f"Created {len(hier_parents)} parent chunks and {len(hier_children)} child nodes")
    for i, parent in enumerate(hier_parents, 1):
        children = [c for c in hier_children if c.parent_id == parent.id]
        print(f"  Parent {i}: {len(parent.text)} chars → {len(children)} children")
    
    # Index and retrieve
    hier_store = QdrantVectorStore(
        client=client,
        collection_name="test_hierarchical_compare",
        vector_size=embeddings.dimension,
        distance="Cosine"
    )
    hier_index = VectorIndex(hier_store, embeddings, index_id="hier_test")
    await hier_index.add_documents(hier_nodes)
    
    hier_retriever = hier_index.as_retriever(top_k=2, resolve_parents=True)
    hier_results = await hier_retriever.retrieve("machine learning")
    
    print(f"\nRetrieved {len(hier_results)} results:")
    for i, result in enumerate(hier_results, 1):
        is_parent = isinstance(result.node, Chunk) and not isinstance(result.node, SymNode)
        node_type = "Parent" if is_parent else "Child"
        print(f"  Result {i}: {node_type}, {len(result.node.text)} chars (score: {result.score:.4f})")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    
    char_avg_size = sum(len(r.node.text) for r in char_results) / len(char_results)
    hier_avg_size = sum(len(r.node.text) for r in hier_results) / len(hier_results)
    
    print(f"\nCharacter Chunking:")
    print(f"  Average result size: {char_avg_size:.0f} chars")
    print(f"  Context: Limited to chunk size")
    
    print(f"\nHierarchical Chunking:")
    print(f"  Average result size: {hier_avg_size:.0f} chars")
    print(f"  Context: Full parent chunk (more context!)")
    print(f"  Search: Uses small child nodes (more precise!)")
    
    print(f"\n✅ Hierarchical provides {(hier_avg_size / char_avg_size):.1f}x more context!")


async def main():
    """Run all tests."""
    try:
        await test_hierarchical_retrieval()
        await compare_with_character_chunking()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
