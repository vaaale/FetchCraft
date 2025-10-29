"""
Simple example demonstrating the streamlined VectorIndex API.

This example shows how easy it is to build a RAG system with the new API.
"""

import asyncio
from qdrant_client import QdrantClient

from fetchcraft import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Node
)


async def main():
    """Minimal example of the new VectorIndex API."""
    
    print("="*60)
    print("Simple VectorIndex Usage Example")
    print("="*60 + "\n")
    
    # 1. Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )
    dimension = embeddings.dimension
    print(f"✓ Embeddings model ready (dimension: {dimension})\n")
    
    # 2. Create documents WITHOUT embeddings
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
        "Machine learning models learn from data.",
        "Natural language processing enables text understanding.",
        "Vector databases store and search embeddings efficiently."
    ]
    
    nodes = [Node(text=text) for text in documents]
    print(f"✓ Created {len(nodes)} documents\n")
    
    # 3. Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="simple_example",
        embeddings=embeddings
    )
    
    # VectorIndex uses the vector store's embeddings
    index = VectorIndex(
        vector_store=vector_store
    )
    print("✓ Vector index created\n")
    
    # 4. Add documents - embeddings generated automatically!
    print("Adding documents to index...")
    await index.add_nodes(nodes)
    print("✓ Documents indexed (embeddings auto-generated!)\n")
    
    # 5. Search with text query
    print("="*60)
    print("Searching")
    print("="*60 + "\n")
    
    query = "programming and code"
    print(f"Query: '{query}'\n")
    
    results = await index.search_by_text(query, k=3)
    
    print("Top results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.3f}] {doc.text}")
    print()
    
    # 6. Create a retriever for convenience
    print("="*60)
    print("Using Retriever")
    print("="*60 + "\n")
    
    retriever = index.as_retriever(top_k=2)
    print(f"✓ Created retriever\n")
    
    queries = [
        "What is machine learning?",
        "Tell me about data storage"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        results = await retriever.aretrieve(query)
        
        for i, node in enumerate(results, 1):
            print(f"  {i}. {node.text}")
        print()
    
    print("="*60)
    print("Done! ✓")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
