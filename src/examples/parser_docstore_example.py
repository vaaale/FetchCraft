"""
Example: Using Parser with DocumentStore Integration

This example demonstrates how to use the TextFileDocumentParser with automatic
DocumentStore integration for storing full documents during parsing.
"""

import asyncio
from pathlib import Path
from fetchcraft import (
    TextFileDocumentParser,
    MongoDBDocumentStore,
    QdrantVectorStore,
    VectorIndex,
    OpenAIEmbeddings
)
from qdrant_client import QdrantClient


async def main():
    """
    Example workflow:
    1. Parse documents with automatic storage in MongoDB
    2. Index chunks in Qdrant for vector search
    3. Search and retrieve full documents
    """
    
    # ========================================
    # 1. Initialize Stores
    # ========================================
    
    # MongoDB for full document storage
    doc_store = MongoDBDocumentStore(
        connection_string="mongodb://localhost:27017",
        database_name="rag_app",
        collection_name="documents"
    )
    
    # Qdrant for vector search
    vector_store = QdrantVectorStore(
        client=QdrantClient(":memory:"),
        collection_name="embeddings",
        embeddings=OpenAIEmbeddings()
    )
    
    index = VectorIndex(vector_store=vector_store, index_id="main")
    
    # ========================================
    # 2. Parse with Automatic Document Storage
    # ========================================
    
    # Option A: Initialize parser with doc_store
    parser = TextFileDocumentParser(doc_store=doc_store)
    
    # Parse a file - DocumentNode automatically stored in MongoDB
    file_path = Path("example_document.txt")
    nodes = await parser.from_file(file_path)
    
    print(f"Parsed {len(nodes)} nodes from {file_path}")
    print(f"DocumentNode: {nodes[0].id}")
    
    # Option B: Override doc_store per call
    nodes2 = await parser.from_file(
        Path("another_document.txt"),
        doc_store=doc_store  # Can override per call
    )
    
    # ========================================
    # 3. Parse Multiple Files (Directory)
    # ========================================
    
    directory = Path("documents/")
    results = await parser.parse_directory(
        directory,
        pattern="*.txt",
        recursive=True
    )
    
    print(f"\nParsed {len(results)} files from directory")
    
    # All DocumentNodes are now in MongoDB
    total_docs = await doc_store.count_documents()
    print(f"Total documents in MongoDB: {total_docs}")
    
    # ========================================
    # 4. Index Chunks in Vector Store
    # ========================================
    
    # Collect all nodes (DocumentNodes + Chunks) from all files
    all_nodes = []
    for file_path, nodes in results.items():
        all_nodes.extend(nodes)
    
    # Add to vector index for search
    await index.add_documents(all_nodes)
    print(f"\nIndexed {len(all_nodes)} nodes for vector search")
    
    # ========================================
    # 5. Search and Retrieve Full Documents
    # ========================================
    
    # Perform vector search
    query = "What is machine learning?"
    search_results = await index.search_by_text(query, k=5)
    
    print(f"\nSearch results for: '{query}'")
    print("-" * 60)
    
    for chunk, score in search_results:
        print(f"\nScore: {score:.4f}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Chunk Text: {chunk.text[:100]}...")
        
        # Retrieve full document from MongoDB using doc_id
        if chunk.doc_id:
            full_doc = await doc_store.get_document(chunk.doc_id)
            if full_doc:
                print(f"Full Document ID: {full_doc.id}")
                print(f"Full Document (first 200 chars): {full_doc.text[:200]}...")
                print(f"Source: {full_doc.metadata.get('source', 'N/A')}")
    
    # ========================================
    # 6. Additional Document Store Operations
    # ========================================
    
    # Get all nodes belonging to a document
    doc_id = nodes[0].id  # First DocumentNode
    all_doc_nodes = await doc_store.get_documents_by_doc_id(doc_id)
    print(f"\n\nDocument {doc_id} has {len(all_doc_nodes)} total nodes")
    
    # List documents with pagination
    recent_docs = await doc_store.list_documents(limit=10, offset=0)
    print(f"Recent documents: {len(recent_docs)}")
    
    # Filter documents by metadata
    filtered_docs = await doc_store.list_documents(
        filters={"metadata.source": {"$regex": ".*example.*"}}
    )
    print(f"Filtered documents: {len(filtered_docs)}")
    
    # ========================================
    # Cleanup
    # ========================================
    await doc_store.close()


async def simple_example():
    """
    Simplified example showing the basics.
    """
    
    # Initialize with doc_store
    doc_store = MongoDBDocumentStore(
        connection_string="mongodb://localhost:27017",
        database_name="simple_example"
    )
    
    parser = TextFileDocumentParser(doc_store=doc_store)
    
    # Parse file - DocumentNode automatically stored
    nodes = await parser.from_file(Path("document.txt"))
    
    # DocumentNode is the first element
    doc_node = nodes[0]
    print(f"Parsed document: {doc_node.id}")
    
    # Verify it's in MongoDB
    retrieved = await doc_store.get_document(doc_node.id)
    assert retrieved is not None
    print(f"Verified in MongoDB: {retrieved.id}")
    
    await doc_store.close()


async def without_docstore_example():
    """
    Example without doc_store - manual storage.
    """
    
    # Parser without doc_store
    parser = TextFileDocumentParser()
    
    # Parse without automatic storage
    nodes = await parser.from_file(Path("document.txt"))
    
    print(f"Parsed {len(nodes)} nodes without doc_store")
    
    # You can still manually store if needed
    doc_store = MongoDBDocumentStore(
        connection_string="mongodb://localhost:27017"
    )
    
    doc_node = nodes[0]
    await doc_store.add_document(doc_node)
    print(f"Manually stored document: {doc_node.id}")
    
    await doc_store.close()


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Or run the simple example
    # asyncio.run(simple_example())
    
    # Or run without doc_store
    # asyncio.run(without_docstore_example())
