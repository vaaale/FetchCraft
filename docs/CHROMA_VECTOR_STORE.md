# ChromaDB Vector Store

Complete guide for using ChromaDB as a vector store backend in fetchcraft.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Distance Metrics](#distance-metrics)
- [Persistence](#persistence)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)

## Overview

`ChromaVectorStore` is a vector store implementation that uses [ChromaDB](https://www.trychroma.com/) as the backend. ChromaDB is an open-source embedding database designed for AI applications.

**Features:**
- ✅ **In-memory and persistent storage** modes
- ✅ **Multiple distance metrics** (cosine, L2, inner product)
- ✅ **Metadata filtering** for advanced queries
- ✅ **Automatic collection management**
- ✅ **Compatible with all fetchcraft features** (hierarchical chunking, etc.)
- ✅ **Easy to use** - similar API to other vector stores

## Installation

Install ChromaDB:

```bash
pip install chromadb
```

Or add to your `requirements.txt`:

```txt
chromadb>=0.4.0
```

## Quick Start

### Basic Usage (In-Memory)

```python
import chromadb
from fetchcraft import ChromaVectorStore, VectorIndex, OpenAIEmbeddings

# Create ChromaDB client
client = chromadb.Client()

# Create vector store
vector_store = ChromaVectorStore(
    client=client,
    collection_name="my_documents",
    distance="cosine"
)

# Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key"
)

# Create index
vector_index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings,
    index_id="my_index"
)

# Add documents
await vector_index.add_nodes(chunks)

# Search
results = await vector_index.search_by_text("your query", k=5)
```

### Using Configuration

```python
from fetchcraft import ChromaVectorStore, ChromaConfig

# Create config
config = ChromaConfig(
    collection_name="my_docs",
    persist_directory="./chroma_db",  # Optional: for persistence
    distance="cosine"
)

# Create vector store from config
vector_store = ChromaVectorStore.from_config(config)
```

## Configuration

### ChromaConfig

Configuration class for ChromaDB vector store.

```python
class ChromaConfig(BaseModel):
    collection_name: str = "documents"
    persist_directory: Optional[str] = None  # None = in-memory
    distance: str = "cosine"  # "cosine", "l2", or "ip"
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | `"documents"` | Name of the ChromaDB collection |
| `persist_directory` | `Optional[str]` | `None` | Directory for persistent storage (None = in-memory) |
| `distance` | `str` | `"cosine"` | Distance metric to use |

### ChromaVectorStore

Main vector store class.

```python
vector_store = ChromaVectorStore(
    client=client,              # ChromaDB client instance
    collection_name="docs",     # Collection name
    document_class=Node,        # Document class (optional)
    distance="cosine"           # Distance metric
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `chromadb.Client` | Required | ChromaDB client instance |
| `collection_name` | `str` | Required | Name of the collection |
| `document_class` | `Type[D]` | `Node` | Document model class |
| `distance` | `str` | `"cosine"` | Distance metric |

## Usage Examples

### Example 1: In-Memory Storage

Perfect for development, testing, or temporary data:

```python
import chromadb
from fetchcraft import ChromaVectorStore, VectorIndex, OpenAIEmbeddings

# Create in-memory client
client = chromadb.Client()

vector_store = ChromaVectorStore(
    client=client,
    collection_name="temp_docs"
)

# Data is stored in RAM, lost when program exits
```

**Use cases:**
- Development and testing
- Temporary data processing
- Quick experiments
- CI/CD pipelines

### Example 2: Persistent Storage

For production use with data that persists across restarts:

```python
import chromadb
from fetchcraft import ChromaVectorStore

# Create persistent client
client = chromadb.PersistentClient(path="./chroma_db")

vector_store = ChromaVectorStore(
    client=client,
    collection_name="production_docs"
)

# Data is saved to disk at ./chroma_db
```

**Use cases:**
- Production applications
- Long-term data storage
- Resumable indexing
- Backup and recovery

### Example 3: With Hierarchical Chunking

Combine ChromaDB with hierarchical chunking for optimal retrieval:

```python
from fetchcraft import (
    ChromaVectorStore,
    VectorIndex,
    OpenAIEmbeddings,
    HierarchicalChunkingStrategy,
    TextFileDocumentParser
)
import chromadb

# Setup
client = chromadb.Client()
vector_store = ChromaVectorStore(client=client, collection_name="hierarchical")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="key")
vector_index = VectorIndex(vector_store, embeddings, "idx")

# Create hierarchical chunks
chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,
    child_chunks=[1024, 512, 256],
    child_overlap=50
)

parser = TextFileDocumentParser(chunker=chunker)
nodes = parser.parse_directory("docs/", pattern="*.md", recursive=True)

# Index all nodes (parents + children)
await vector_index.add_nodes(nodes)

# Search with parent resolution
retriever = vector_index.as_retriever(top_k=5, resolve_parents=True)
results = await retriever.aretrieve("your query")

# Returns parent chunks for full context!
```

### Example 4: Metadata Filtering

Use metadata to filter search results:

```python
# Add documents with metadata
chunks = parser.parse(text, metadata={
    "parsing": "user_manual",
    "version": "2.0",
    "category": "installation"
})

await vector_index.insert_nodes(chunks)

# Search with metadata filter
results = await vector_store.similarity_search(
    query_embedding=embedding,
    k=5,
    where={"category": "installation"}  # Filter by metadata
)
```

### Example 5: Multiple Collections

Use different collections for different types of data:

```python
import chromadb

client = chromadb.Client()

# Collection for user documentation
docs_store = ChromaVectorStore(
    client=client,
    collection_name="user_docs"
)

# Collection for API references
api_store = ChromaVectorStore(
    client=client,
    collection_name="api_docs"
)

# Collection for FAQs
faq_store = ChromaVectorStore(
    client=client,
    collection_name="faqs"
)

# Each collection is independent
```

## Distance Metrics

ChromaDB supports three distance metrics:

### 1. Cosine Distance (Default)

**Formula:** `1 - cosine_similarity(a, b)`

```python
vector_store = ChromaVectorStore(
    client=client,
    collection_name="docs",
    distance="cosine"
)
```

**Best for:**
- Text embeddings (most common)
- When vector magnitude doesn't matter
- General-purpose semantic search

**Score range:** 0.0 (identical) to 2.0 (opposite)  
**Similarity conversion:** `similarity = 1 - distance`

### 2. L2 Distance (Euclidean)

**Formula:** `sqrt(sum((a - b)^2))`

```python
vector_store = ChromaVectorStore(
    client=client,
    collection_name="docs",
    distance="l2"
)
```

**Best for:**
- When absolute distances matter
- Image embeddings
- Spatial data

**Score range:** 0.0 (identical) to infinity  
**Similarity conversion:** `similarity = 1 / (1 + distance)`

### 3. Inner Product (IP)

**Formula:** `sum(a * b)`

```python
vector_store = ChromaVectorStore(
    client=client,
    collection_name="docs",
    distance="ip"
)
```

**Best for:**
- Normalized vectors
- Maximum dot product search
- Specific ML models

**Score range:** -1.0 to 1.0 (for normalized vectors)  
**Note:** Higher is better (already a similarity score)

### Choosing a Distance Metric

| Use Case | Recommended Metric | Reason |
|----------|-------------------|---------|
| Text embeddings (OpenAI, etc.) | **Cosine** | Standard for semantic similarity |
| Image embeddings | **L2** or **Cosine** | Depends on embedding model |
| Normalized vectors | **IP** | Most efficient |
| Custom embeddings | **Test all** | Benchmark on your data |

## Persistence

### In-Memory Mode

Data is stored in RAM and lost when the program exits:

```python
import chromadb

client = chromadb.Client()  # In-memory
```

**Advantages:**
- Fast
- No disk I/O
- Good for testing

**Disadvantages:**
- Data is lost on exit
- Limited by available RAM

### Persistent Mode

Data is saved to disk:

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
```

**Advantages:**
- Data persists across restarts
- Can handle large datasets
- Production-ready

**Disadvantages:**
- Slower than in-memory
- Requires disk space

### Best Practices

1. **Development:** Use in-memory for fast iteration
2. **Testing:** Use temporary directories
3. **Production:** Use persistent storage with backups
4. **Large datasets:** Use persistent storage with SSD

## Advanced Features

### Index Isolation

Use `index_id` to isolate different indices in the same collection:

```python
# Index 1
index1 = VectorIndex(vector_store, embeddings, index_id="user_docs")
await index1.insert_nodes(user_chunks)

# Index 2
index2 = VectorIndex(vector_store, embeddings, index_id="admin_docs")
await index2.insert_nodes(admin_chunks)

# Searches are isolated by index_id
results1 = await index1.search_by_text("query")  # Only searches user_docs
results2 = await index2.search_by_text("query")  # Only searches admin_docs
```

### Document Management

```python
# Add documents
doc_ids = await vector_store.insert_nodes(chunks, index_id="my_index")

# Retrieve a single document
doc = await vector_store.get_node("doc_id", index_id="my_index")

# Delete documents
await vector_store.delete(["doc_id1", "doc_id2"], index_id="my_index")
```

### Custom Document Classes

Use custom document types that inherit from `Node`:

```python
from fetchcraft import Node
from pydantic import Field

class CustomDocument(Node):
    author: str = Field(description="Document author")
    date: str = Field(description="Publication date")
    category: str = Field(description="Document category")

# Use with ChromaVectorStore
vector_store = ChromaVectorStore(
    client=client,
    collection_name="custom_docs",
    document_class=CustomDocument
)
```

## API Reference

### ChromaVectorStore Methods

#### `add_documents(documents, index_id=None)`

Add documents to the collection.

```python
async def add_documents(
    self, 
    documents: List[D], 
    index_id: Optional[str] = None
) -> List[str]:
    """Returns list of document IDs"""
```

#### `similarity_search(query_embedding, k=4, index_id=None, **kwargs)`

Search for similar documents.

```python
async def similarity_search(
    self,
    query_embedding: List[float],
    k: int = 4,
    index_id: Optional[str] = None,
    **kwargs
) -> List[tuple[D, float]]:
    """Returns list of (document, score) tuples"""
```

#### `get_document(document_id, index_id=None)`

Retrieve a single document by ID.

```python
async def get_document(
    self, 
    document_id: str, 
    index_id: Optional[str] = None
) -> Optional[D]:
    """Returns document or None"""
```

#### `delete(ids, index_id=None)`

Delete documents by IDs.

```python
async def delete(
    self, 
    ids: List[str], 
    index_id: Optional[str] = None
) -> bool:
    """Returns True if successful"""
```

#### `from_config(config)`

Create vector store from configuration.

```python
@classmethod
def from_config(
    cls, 
    config: Union[Dict[str, Any], ChromaConfig]
) -> 'ChromaVectorStore':
    """Returns ChromaVectorStore instance"""
```

## Comparison with Other Vector Stores

| Feature | ChromaDB | Qdrant | Pinecone |
|---------|----------|---------|----------|
| In-memory mode | ✅ | ✅ | ❌ |
| Persistent storage | ✅ | ✅ | ✅ (cloud) |
| Self-hosted | ✅ | ✅ | ❌ |
| Cloud option | ✅ | ✅ | ✅ |
| Distance metrics | 3 | 3+ | Multiple |
| Metadata filtering | ✅ | ✅ | ✅ |
| Easy setup | ✅✅ | ✅ | ✅ |
| Cost | Free | Free/Paid | Paid |

**ChromaDB is best for:**
- Local development
- Self-hosted applications
- Projects requiring both in-memory and persistent modes
- Getting started quickly

## Troubleshooting

### ChromaDB Not Found

```python
ImportError: cannot import name 'chromadb'
```

**Solution:** Install ChromaDB
```bash
pip install chromadb
```

### Collection Already Exists

ChromaDB will automatically use existing collections. To start fresh:

```python
# Delete existing collection
client.delete_collection(name="my_collection")

# Create new one
vector_store = ChromaVectorStore(client=client, collection_name="my_collection")
```

### Dimension Mismatch

```python
ValueError: Dimension of embeddings does not match collection
```

**Solution:** Ensure all embeddings have the same dimension, or create a new collection.

### Persistence Issues

If data isn't persisting:

```python
# Make sure you're using PersistentClient
client = chromadb.PersistentClient(path="./chroma_db")  # ✅

# Not this:
# client = chromadb.Client()  # ❌ In-memory only
```

## Examples

See `/src/examples/chroma_example.py` for complete working examples:

```bash
python -m examples.chroma_example
```

## Resources

- **ChromaDB Documentation:** https://docs.trychroma.com/
- **ChromaDB GitHub:** https://github.com/chroma-core/chroma
- **fetchcraft Examples:** `/src/examples/chroma_example.py`

## Next Steps

1. Try the basic example: `python -m examples.chroma_example`
2. Read about [hierarchical chunking](CHUNKING_STRATEGIES.md)
3. Explore [retriever tools](RETRIEVER_TOOLS.md)
4. Check out [agent integration](AGENTS.md)
