# Document Store Usage Guide

## Overview

The Document Store provides persistent storage for full documents (DocumentNodes) separately from vector embeddings. This architecture allows for:

- **Efficient vector search** in vector stores (Qdrant, Chroma)
- **Full document retrieval** from document stores (MongoDB)
- **Independent scaling** of storage layers
- **Clean separation of concerns**

## Architecture

```
┌─────────────────────┐
│   Application       │
└──────┬──────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐    ┌─────────────────┐
│ Vector Store │    │ Document Store  │
│  (Qdrant)    │    │   (MongoDB)     │
├──────────────┤    ├─────────────────┤
│ Embeddings   │    │ Full Documents  │
│ + Metadata   │    │ + All Metadata  │
│              │    │                 │
│ Fast Search  │    │ Full Retrieval  │
└──────────────┘    └─────────────────┘
```

## Installation

```bash
# Install MongoDB support
pip install motor  # Async MongoDB driver

# Optional: For testing with mock MongoDB
pip install mongomock-motor
```

## Basic Usage

### 1. Initialize MongoDB Document Store

```python
from fetchcraft import MongoDBDocumentStore, DocumentNode

# Create store
store = MongoDBDocumentStore(
    connection_string="mongodb://localhost:27017",
    database_name="my_rag_app",
    collection_name="documents"
)
```

### 2. Store Documents

```python
# Create a document
doc = DocumentNode.from_text(
    text="This is the full document text...",
    metadata={
        "parsing": "example.txt",
        "author": "John Doe",
        "date": "2025-01-15"
    }
)

# Store it
doc_id = await store.add_document(doc)
print(f"Stored document: {doc_id}")
```

### 3. Store Multiple Documents

```python
# Create multiple documents
docs = [
    DocumentNode.from_text(f"Document {i}", metadata={"index": i})
    for i in range(100)
]

# Bulk insert
doc_ids = await store.insert_nodes(docs)
print(f"Stored {len(doc_ids)} documents")
```

### 4. Retrieve Documents

```python
# Get single document
doc = await store.get_node(doc_id)
print(f"Retrieved: {doc.text[:50]}...")

# Get multiple documents
docs = await store.get_documents([id1, id2, id3])
print(f"Retrieved {len(docs)} documents")
```

### 5. Update Documents

```python
# Modify document
doc.text = "Updated text"
doc.metadata["version"] = 2

# Save changes
success = await store.update_document(doc)
print(f"Updated: {success}")
```

### 6. Delete Documents

```python
# Delete single document
success = await store.delete_document(doc_id)

# Delete multiple documents
success = await store.delete_documents([id1, id2, id3])
```

### 7. List and Search

```python
# List with pagination
docs = await store.list_documents(limit=10, offset=0)

# Count total documents
count = await store.count_documents()
print(f"Total documents: {count}")

# Check if document exists
exists = await store.document_exists(doc_id)
```

## Integration with Vector Stores

### Combined RAG Pipeline

```python
from fetchcraft import (
    MongoDBDocumentStore,
    QdrantVectorStore,
    VectorIndex,
    TextFileDocumentParser,
    OpenAIEmbeddings
)
from pathlib import Path

# 1. Initialize stores
doc_store = MongoDBDocumentStore(
    connection_string="mongodb://localhost:27017",
    database_name="rag_app",
    collection_name="documents"
)

vector_store = QdrantVectorStore(
    client=QdrantClient(":memory:"),
    collection_name="embeddings",
    embeddings=OpenAIEmbeddings()
)

index = VectorIndex(vector_store=vector_store, index_id="main")

# 2. Parse and process documents
parser = TextFileDocumentParser()
nodes = parser.from_file(Path("document.txt"))

# nodes[0] is DocumentNode, rest are Chunks/SymNodes
doc_node = nodes[0]
chunk_nodes = nodes[1:]

# 3. Store DocumentNode in document store
await doc_store.add_document(doc_node)

# 4. Store all nodes (including chunks) in vector store for search
await index.add_nodes(DocumentNode, nodes)

# 5. Search using vectors
results = await index.search_by_text("What is the main topic?", k=5)

# 6. Retrieve full document using doc_id
for node, score in results:
    # Get the original document
    full_doc = await doc_store.get_document(node.doc_id)
    print(f"Score: {score:.3f}")
    print(f"Chunk: {node.text[:100]}...")
    print(f"Full Document: {full_doc.text[:200]}...")
    print("---")
```

## Advanced Features

### Retrieve All Nodes from a Document

```python
# Get DocumentNode + all chunks by doc_id
all_nodes = await store.get_documents_by_doc_id(doc.id)

doc_nodes = [n for n in all_nodes if isinstance(n, DocumentNode)]
chunk_nodes = [n for n in all_nodes if isinstance(n, Chunk)]

print(f"Document: {doc_nodes[0].text[:50]}...")
print(f"Chunks: {len(chunk_nodes)}")
```

### Filtering Documents

```python
# MongoDB query filters
docs = await store.list_documents(
    limit=100,
    filters={
        "metadata.parsing": "example.txt",
        "metadata.date": {"$gte": "2025-01-01"}
    }
)

# Count with filters
count = await store.count_documents(
    filters={"metadata.author": "John Doe"}
)
```

### Connection Management

```python
# Close connection when done
await store.close()

# Or use async context manager (if implemented)
async with MongoDBDocumentStore(...) as store:
    await store.add_document(doc)
    # Connection automatically closed
```

## Configuration Options

```python
from fetchcraft import MongoDBConfig

config = MongoDBConfig(
    connection_string="mongodb://user:password@host:port/",
    database_name="my_database",
    collection_name="my_documents"
)

store = MongoDBDocumentStore(
    connection_string=config.connection_string,
    database_name=config.database_name,
    collection_name=config.collection_name
)
```

## Best Practices

### 1. Separate Storage for Different Purposes

- **DocumentStore**: Store full DocumentNodes with complete metadata
- **VectorStore**: Store all searchable nodes (chunks, symnodes) with embeddings

### 2. Use doc_id for Cross-Store References

```python
# All nodes from same document share doc_id
doc = DocumentNode.from_text("...")
chunks = parser.parse(doc.text, parent_node=doc)

# All chunks have doc_id == doc.id
assert all(chunk.doc_id == doc.id for chunk in chunks)

# Search vectors, retrieve full document
results = await vector_index.search_by_text("query")
for chunk, score in results:
    full_doc = await doc_store.get_node(chunk.doc_id)
```

### 3. Batch Operations for Performance

```python
# Good: Batch insert
await store.insert_nodes(docs)

# Bad: Individual inserts
for doc in docs:
    await store.add_document(doc)  # Slow!
```

### 4. Index Important Fields

```python
# MongoDB automatically creates indexes on:
# - id (unique)
# - doc_id
# - metadata.parsing

# Add custom indexes if needed
await store.collection.create_index([("metadata.custom_field", 1)])
```

## Error Handling

```python
try:
    doc = await store.get_node(doc_id)
    if doc is None:
        print("Document not found")
except Exception as e:
    print(f"Error retrieving document: {e}")
```

## Performance Considerations

- **Bulk operations**: Use `add_documents()` instead of multiple `add_document()` calls
- **Pagination**: Use `limit` and `offset` for large result sets
- **Indexes**: MongoDB automatically indexes `id` and `doc_id`
- **Connection pooling**: Motor handles this automatically
- **Async operations**: Always use `await` for non-blocking I/O

## Testing

```python
import pytest
from mongomock_motor import AsyncMongoMockClient


@pytest.mark.asyncio
async def test_document_store():
    # Use mock client for testing
    client = AsyncMongoMockClient()

    store = MongoDBDocumentStore(
        database_name="test_db",
        collection_name="test_collection",
        client=client
    )

    doc = DocumentNode.from_text("Test")
    await store.add_document(doc)

    retrieved = await store.get_node(doc.id)
    assert retrieved.text == "Test"

    await store.close()
```

## Summary

The Document Store provides:

✅ **Persistent storage** for full documents  
✅ **Async operations** via Motor (async MongoDB driver)  
✅ **CRUD operations** (Create, Read, Update, Delete)  
✅ **Bulk operations** for performance  
✅ **Pagination** for large datasets  
✅ **Filtering** via MongoDB queries  
✅ **doc_id integration** for cross-store references  
✅ **Type preservation** (DocumentNode, Chunk, SymNode)  
✅ **Automatic indexing** for fast lookups  

Use it alongside Vector Stores for a complete RAG solution! 🚀
