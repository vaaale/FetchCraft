"""
Example demonstrating the use of OpenAI embeddings with the RAG framework.
"""

import asyncio
from pathlib import Path
from qdrant_client import QdrantClient

from rag_framework import (
    OpenAIEmbeddings,
    TextFileDocumentParser,
    QdrantVectorStore,
    VectorIndex,
    Node,
    Chunk
)


async def basic_embeddings_example():
    """Basic example of using OpenAI embeddings."""
    
    print("="*60)
    print("Basic OpenAI Embeddings Example")
    print("="*60 + "\n")
    
    # Initialize OpenAI embeddings
    # API key will be read from OPENAI_API_KEY environment variable
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",  # 1536 dimensions
        api_key="sk-124",  # Optional: specify key directly
        base_url="http://wingman:8000/v1"  # Optional: custom endpoint
    )
    
    print(f"Initialized: {embeddings}")
    
    # Determine dimension asynchronously
    dimension = await embeddings.aget_dimension()
    print(f"Embedding dimension: {dimension}\n")
    
    # Embed a single query
    query_text = "What is machine learning?"
    try:
        query_embedding = await embeddings.embed_query(query_text)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return
    
    print(f"Query: '{query_text}'")
    print(f"Embedding length: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}\n")
    
    # Embed multiple documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text."
    ]
    
    doc_embeddings = await embeddings.embed_documents(documents)
    
    print(f"Embedded {len(doc_embeddings)} documents")
    for i, (doc, emb) in enumerate(zip(documents, doc_embeddings)):
        print(f"  {i+1}. {doc[:50]}... (dim: {len(emb)})")
    print()


async def rag_pipeline_example():
    """Complete RAG pipeline using OpenAI embeddings."""
    
    print("="*60)
    print("Complete RAG Pipeline with OpenAI Embeddings")
    print("="*60 + "\n")
    
    # Step 1: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",  # 1536 dimensions
        api_key="sk-124",  # Optional: specify key directly
        base_url="http://wingman:8000/v1"  # Optional: custom endpoint
    )
    
    # Determine dimension asynchronously
    dimension = await embeddings.aget_dimension()
    print(f"✓ Initialized embeddings (dimension: {dimension})")
    
    # Step 2: Create sample documents
    documents_text = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is widely used for web development and runs in browsers.",
        "Machine learning algorithms can learn patterns from data.",
        "Neural networks are inspired by biological neural networks.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Step 3: Create Node objects WITHOUT embeddings
    nodes = []
    for text in documents_text:
        node = Node(
            text=text,
            metadata={"source": "example", "type": "document"}
        )
        nodes.append(node)
    
    print(f"✓ Created {len(nodes)} nodes\n")
    
    # Step 4: Setup vector store and index
    client = QdrantClient(":memory:")  # Use in-memory for demo
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="documents",
        vector_size=dimension
    )
    
    # VectorIndex now takes embeddings at initialization
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings
    )
    
    # Step 5: Add documents to index (embeddings auto-generated!)
    doc_ids = await index.add_documents(nodes)
    print(f"✓ Indexed {len(doc_ids)} documents (embeddings auto-generated!)\n")
    
    # Step 6: Perform semantic search
    queries = [
        "programming languages",
        "artificial intelligence and learning",
        "understanding text with computers"
    ]
    
    print("Searching for similar documents:\n")
    for query in queries:
        print(f"Query: '{query}'")
        
        # Search with text query directly (no manual embedding needed!)
        results = await index.search_by_text(query, k=2)
        
        print("  Top results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"    {i}. [Score: {score:.3f}] {doc.text}")
        print()


async def document_parsing_with_embeddings():
    """Example of parsing documents and adding embeddings."""
    
    print("="*60)
    print("Document Parsing with Embeddings")
    print("="*60 + "\n")
    
    # Create a sample text file
    sample_text = """
    Artificial Intelligence has revolutionized many industries. Machine learning, 
    a subset of AI, enables computers to learn from data without explicit programming.
    Deep learning, using neural networks, has achieved remarkable results in image 
    recognition, natural language processing, and game playing. The field continues 
    to evolve rapidly with new architectures and techniques being developed regularly.
    """.strip()
    
    sample_file = Path("temp_document.txt")
    sample_file.write_text(sample_text)
    
    # Parse the document into chunks
    chunks = TextFileDocumentParser.from_file(
        file_path=sample_file,
        chunk_size=100,
        overlap=20
    )
    
    print(f"✓ Parsed document into {len(chunks)} chunks\n")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        api_key="sk-124",
        base_url="http://wingman:8000/v1"
    )
    
    # Get dimension
    dimension = await embeddings.aget_dimension()
    print(f"✓ Initialized embeddings (dimension: {dimension})\n")
    
    # Create vector store with Chunk as document class
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="chunks",
        document_class=Chunk,  # Use Chunk to preserve chunk-specific properties
        vector_size=dimension
    )
    
    # VectorIndex with embeddings - will auto-generate embeddings for chunks!
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings
    )
    
    # Index the chunks (embeddings auto-generated!)
    chunk_ids = await index.add_documents(chunks)
    print(f"✓ Indexed {len(chunk_ids)} chunks (embeddings auto-generated!)\n")
    
    # Search within chunks using text query
    query = "What is deep learning?"
    print(f"Query: '{query}'")
    
    results = await index.search_by_text(query, k=2)
    print("\nTop matching chunks:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}]")
        print(f"     Chunk {chunk.chunk_index}: {chunk.text[:80]}...")
        print(f"     Position: chars {chunk.start_char_idx}-{chunk.end_char_idx}")
    
    # Cleanup
    sample_file.unlink()
    print("\n✓ Cleaned up temporary file")


async def custom_endpoint_example():
    """Example using a custom OpenAI-compatible endpoint."""
    
    print("\n" + "="*60)
    print("Custom Endpoint Example (Azure OpenAI / Local Model)")
    print("="*60 + "\n")
    
    # Example with custom base_url (e.g., for Azure OpenAI or local model)
    try:
        embeddings = OpenAIEmbeddings(
            model="qwen3-embedding-0.6b",  # 1536 dimensions
            api_key="sk-124",  # Optional: specify key directly
            base_url="http://wingman:8000/v1"  # Optional: custom endpoint
        )

        print(f"Initialized custom endpoint: {embeddings}")
        print(f"Base URL: {embeddings.base_url}")
        print(f"Model: {embeddings.model}")
        
        # Determine dimension asynchronously
        dimension = await embeddings.aget_dimension()
        print(f"Dimension: {dimension}")
        
    except Exception as e:
        print(f"Note: Custom endpoint example (requires valid credentials)")
        print(f"Configuration shown for reference only\n")
        print("You can use this pattern for:")
        print("  - Azure OpenAI endpoints")
        print("  - Local embedding models (e.g., via LiteLLM)")
        print("  - Other OpenAI-compatible APIs")


async def main():
    """Run all examples."""
    
    # Check if API key is available

    try:
        await custom_endpoint_example()
        await basic_embeddings_example()
        await rag_pipeline_example()
        await document_parsing_with_embeddings()
        await custom_endpoint_example()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is valid and you have API credits.")


if __name__ == "__main__":
    asyncio.run(main())
