"""
Example: Complete Document Processing Pipeline

This example demonstrates the full pipeline:
1. Load documents from filesystem using FilesystemDocumentSource
2. Parse documents using SimpleNodeParser or HierarchicalNodeParser
3. Index and search the parsed nodes
"""

import asyncio
from pathlib import Path
import tempfile
from qdrant_client import QdrantClient

from fetchcraft import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Chunk,
    SymNode
)
from fetchcraft.source import FilesystemDocumentSource
from fetchcraft.node_parser import SimpleNodeParser, HierarchicalNodeParser


async def example_simple_parser():
    """Example using SimpleNodeParser"""
    print("="*80)
    print("Example 1: SimpleNodeParser")
    print("="*80)
    
    # Create temp directory with sample documents
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create sample documents
        (tmp_path / "doc1.txt").write_text(
            "Artificial intelligence is transforming technology. "
            "Machine learning enables computers to learn from data. "
            "Deep learning uses neural networks for complex tasks."
        )
        (tmp_path / "doc2.txt").write_text(
            "Python is a popular programming language. "
            "It is used for web development, data science, and AI. "
            "Python has a simple and readable syntax."
        )
        (tmp_path / "doc3.txt").write_text(
            "Vector databases store embeddings efficiently. "
            "They enable semantic search over large datasets. "
            "Popular vector databases include Qdrant and Milvus."
        )
        
        # Step 1: Load documents
        print("\n1. Loading documents...")
        source = FilesystemDocumentSource.from_directory(
            directory=tmp_path,
            pattern="*.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        print(f"   ✓ Loaded {len(documents)} documents")
        
        # Step 2: Parse with SimpleNodeParser
        print("\n2. Parsing with SimpleNodeParser...")
        parser = SimpleNodeParser(
            chunk_size=100,
            overlap=20
        )
        
        nodes = parser.get_nodes(documents)
        print(f"   ✓ Created {len(nodes)} chunks")
        
        # Show sample
        print(f"\n   Sample chunk:")
        print(f"   - Text: {nodes[0].text[:80]}...")
        print(f"   - Length: {len(nodes[0].text)} chars")
        print(f"   - Doc ID: {nodes[0].doc_id[:8]}...")
        
        # Step 3: Index
        print("\n3. Indexing chunks...")
        embeddings = OpenAIEmbeddings(
            model="qwen3-embedding-0.6b",
            api_key="sk-123",
            base_url="http://wingman:8000/v1"
        )
        
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="simple_example",
            embeddings=embeddings
        )
        
        index = VectorIndex(vector_store=vector_store)
        await index.add_nodes(nodes)
        print(f"   ✓ Indexed {len(nodes)} chunks")
        
        # Step 4: Search
        print("\n4. Searching...")
        query = "What is machine learning?"
        results = await index.search_by_text(query, k=2)
        
        print(f"   Query: '{query}'")
        print(f"   Results:")
        for i, (node, score) in enumerate(results, 1):
            print(f"   {i}. [Score: {score:.3f}] {node.text[:60]}...")


async def example_hierarchical_parser():
    """Example using HierarchicalNodeParser"""
    print("\n" + "="*80)
    print("Example 2: HierarchicalNodeParser")
    print("="*80)
    
    # Create temp directory with sample documents
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a longer document
        long_text = """
        Introduction to Neural Networks
        
        Neural networks are computational models inspired by biological neurons.
        They consist of interconnected nodes organized in layers. Each connection
        has a weight that adjusts during training.
        
        Architecture
        
        A typical neural network has three types of layers: input, hidden, and output.
        The input layer receives data, hidden layers process it, and the output layer
        produces predictions. Deep networks have multiple hidden layers.
        
        Training Process
        
        Training uses backpropagation and gradient descent. The network makes predictions,
        calculates errors, and adjusts weights to minimize loss. This iterative process
        continues until the model converges.
        
        Applications
        
        Neural networks excel at image recognition, natural language processing, and
        game playing. They power modern AI systems like chatbots and recommendation
        engines. Recent advances include transformers and attention mechanisms.
        """ * 2  # Repeat for longer content
        
        (tmp_path / "neural_networks.txt").write_text(long_text)
        
        # Step 1: Load documents
        print("\n1. Loading documents...")
        source = FilesystemDocumentSource.from_directory(
            directory=tmp_path,
            pattern="*.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        print(f"   ✓ Loaded {len(documents)} documents")
        print(f"   ✓ Total length: {len(documents[0].text)} chars")
        
        # Step 2: Parse with HierarchicalNodeParser
        print("\n2. Parsing with HierarchicalNodeParser...")
        parser = HierarchicalNodeParser(
            chunk_size=400,
            overlap=50,
            child_sizes=[150, 75],
            child_overlap=15
        )
        
        nodes = parser.get_nodes(documents)
        
        parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
        child_nodes = [n for n in nodes if isinstance(n, SymNode)]
        
        print(f"   ✓ Created {len(nodes)} total nodes")
        print(f"     - Parent chunks: {len(parent_chunks)}")
        print(f"     - Child SymNodes: {len(child_nodes)}")
        
        # Show hierarchy
        print(f"\n   First parent chunk:")
        parent = parent_chunks[0]
        print(f"   - Text: {parent.text[:80]}...")
        print(f"   - Length: {len(parent.text)} chars")
        print(f"   - Children: {len(parent.children_ids)}")
        
        print(f"\n   Sample child SymNode:")
        child = child_nodes[0]
        print(f"   - Text: {child.text[:60]}...")
        print(f"   - Size: {child.metadata['child_size']} chars")
        print(f"   - Parent ID: {child.parent_id[:8]}...")
        
        # Step 3: Index only children for search
        print("\n3. Indexing child SymNodes...")
        embeddings = OpenAIEmbeddings(
            model="qwen3-embedding-0.6b",
            api_key="sk-123",
            base_url="http://wingman:8000/v1"
        )
        
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="hierarchical_example",
            embeddings=embeddings
        )
        
        index = VectorIndex(vector_store=vector_store)
        await index.add_nodes(child_nodes)
        print(f"   ✓ Indexed {len(child_nodes)} child SymNodes")
        
        # Step 4: Search (returns children, resolve to parents)
        print("\n4. Searching...")
        query = "How does training work?"
        results = await index.search_by_text(query, k=3)
        
        print(f"   Query: '{query}'")
        print(f"   Results (child SymNodes):")
        for i, (node, score) in enumerate(results, 1):
            print(f"   {i}. [Score: {score:.3f}] {node.text[:50]}...")
            if isinstance(node, SymNode):
                print(f"      → Would resolve to parent: {node.parent_id[:8]}...")


async def example_comparison():
    """Compare SimpleNodeParser vs HierarchicalNodeParser"""
    print("\n" + "="*80)
    print("Example 3: Comparison")
    print("="*80)
    
    sample_text = "Sample text. " * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "sample.txt").write_text(sample_text)
        
        source = FilesystemDocumentSource.from_directory(tmp_path)
        documents = [doc async for doc in source.get_documents()]
        
        # Simple parser
        simple = SimpleNodeParser(chunk_size=200, overlap=20)
        simple_nodes = simple.get_nodes(documents)
        
        # Hierarchical parser
        hierarchical = HierarchicalNodeParser(
            chunk_size=200,
            overlap=20,
            child_sizes=[100, 50]
        )
        hierarchical_nodes = hierarchical.get_nodes(documents)
        hierarchical_children = [n for n in hierarchical_nodes if isinstance(n, SymNode)]
        
        print(f"\nSimpleNodeParser:")
        print(f"  - Total nodes: {len(simple_nodes)}")
        print(f"  - Storage: 1x")
        
        print(f"\nHierarchicalNodeParser:")
        print(f"  - Total nodes: {len(hierarchical_nodes)}")
        print(f"  - Child SymNodes: {len(hierarchical_children)}")
        print(f"  - Storage: ~{len(hierarchical_nodes)/len(simple_nodes):.1f}x")
        print(f"  - Better search precision (smaller children)")
        print(f"  - Better context retrieval (large parents)")


async def main():
    """Run all examples"""
    await example_simple_parser()
    await example_hierarchical_parser()
    await example_comparison()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
