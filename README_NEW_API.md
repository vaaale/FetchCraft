# RAG Framework - Simplified API

## Quick Start

```python
import asyncio
from qdrant_client import QdrantClient
from rag_framework import OpenAIEmbeddings, QdrantVectorStore, VectorIndex, Node

async def main():
    # 1. Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = await embeddings.aget_dimension()
    
    # 2. Create documents (NO embeddings needed!)
    nodes = [
        Node(text="Python is a programming language."),
        Node(text="Machine learning uses algorithms.")
    ]
    
    # 3. Create index with embeddings
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="docs",
        vector_size=dimension
    )
    
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings  # Embeddings are part of the index!
    )
    
    # 4. Add documents (embeddings auto-generated!)
    await index.add_documents(nodes)
    
    # 5. Search with text
    results = await index.search_by_text("What is Python?", k=2)
    
    # 6. Or use a retriever
    retriever = index.as_retriever(top_k=2)
    results = await retriever.retrieve("What is Python?")
    
    for doc, score in results:
        print(f"[{score:.3f}] {doc.text}")

asyncio.run(main())
```

## Key Features

### ✅ Automatic Embedding Generation
No need to manually generate embeddings - the index handles it!

```python
# Old way
embeddings_list = await embeddings.embed_documents(texts)
nodes = [Node(text=t, embedding=e) for t, e in zip(texts, embeddings_list)]

# New way
nodes = [Node(text=t) for t in texts]
await index.add_documents(nodes)  # Embeddings auto-generated!
```

### ✅ Text-Based Search
Search directly with text queries:

```python
results = await index.search_by_text("your query", k=5)
```

### ✅ Simplified Retriever
Create retrievers without passing embeddings:

```python
retriever = index.as_retriever(top_k=5)
results = await retriever.retrieve("your query")
```

### ✅ Hierarchical Nodes with SymNode
Automatically resolve parent nodes:

```python
from rag_framework import Chunk, SymNode

# Create parent with full context
parent = Chunk.from_text(text="Long parent text...")

# Create smaller SymNodes for precise matching
sym_nodes = parent.create_symbolic_nodes([
    "fragment 1",
    "fragment 2"
])

# Add to index (embeddings auto-generated!)
await index.add_documents([parent])
await index.add_documents(sym_nodes)

# Search returns parent chunks, not SymNodes!
results = await index.search_by_text("query", k=3)
```

## Complete Example

See `src/examples/simple_usage.py` for a complete working example.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  VectorIndex                    │
│  - Manages vector store                         │
│  - Owns embeddings model                        │
│  - Auto-generates embeddings                    │
│  - Resolves SymNode parents                     │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌──────────────┐
│  VectorStore  │       │  Embeddings  │
│  (Qdrant)     │       │  (OpenAI)    │
└───────────────┘       └──────────────┘
```

## API Overview

### VectorIndex

**Constructor:**
```python
VectorIndex(
    vector_store: VectorStore,
    embeddings: Embeddings,  # Required!
    index_id: Optional[str] = None
)
```

**Methods:**
- `add_documents(docs, auto_embed=True)` - Add documents with auto-embedding
- `search_by_text(query, k=4)` - Search with text query
- `search(query_embedding, k=4)` - Search with embedding vector
- `as_retriever(top_k=4)` - Create a retriever
- `get_document(doc_id)` - Get document by ID
- `delete_documents(doc_ids)` - Delete documents

### Retriever

**Creation:**
```python
retriever = index.as_retriever(
    top_k=4,
    resolve_parents=True
)
```

**Methods:**
- `retrieve(query)` - Retrieve documents by text query
- `aretrieve(query)` - Async alias for retrieve
- `update_config(top_k=...)` - Update retriever config

### SymNode

**Creation:**
```python
parent = Chunk.from_text(text="Parent text...")
sym_node = SymNode.create(
    text="Child text",
    parent_id=parent.id
)

# Or use helper
sym_nodes = parent.create_symbolic_nodes(["text1", "text2"])
```

## Examples

- **`src/examples/simple_usage.py`** - Basic usage
- **`src/examples/retriever_example.py`** - Retriever patterns
- **`src/examples/symnode_example.py`** - Hierarchical nodes
- **`src/examples/embeddings_example.py`** - Embedding integration

## Documentation

- **`VECTORINDEX_API_CHANGES.md`** - Complete API changes and migration guide
- **`RETRIEVER_USAGE.md`** - Retriever abstraction guide
- **`SYMNODE_USAGE.md`** - SymNode hierarchical nodes guide

## Testing

```bash
# Run all tests
python -m pytest src/tests/ -v

# Run specific test
python -m pytest src/tests/test_retriever.py -v
```

## Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "qdrant-client>=1.15.1",
    "openai>=1.0.0",
    "pytest>=8.4.2",
]
```

## What's New

### 🎯 Simplified API
- VectorIndex requires embeddings at initialization
- Automatic embedding generation when adding documents
- No manual embedding management needed

### 🔍 Text-Based Search
- New `search_by_text()` method for direct text queries
- No need to generate query embeddings manually

### 🎨 Cleaner Retriever
- `as_retriever()` uses index's embeddings model
- No need to pass embeddings separately

### 🌲 SymNode Hierarchical Nodes
- Create parent-child relationships
- Automatic parent resolution during retrieval
- Better context with precise matching

## Upgrade Path

1. **Add embeddings to VectorIndex initialization:**
   ```python
   index = VectorIndex(vector_store, embeddings)  # Add embeddings!
   ```

2. **Remove manual embedding generation:**
   ```python
   # Before: doc_embeddings = await embeddings.embed_documents(texts)
   # After: Just create nodes without embeddings
   nodes = [Node(text=t) for t in texts]
   ```

3. **Update retriever creation:**
   ```python
   # Before: retriever = index.as_retriever(embeddings=embeddings, top_k=5)
   # After: retriever = index.as_retriever(top_k=5)
   ```

See `VECTORINDEX_API_CHANGES.md` for detailed migration guide.
