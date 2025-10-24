# VectorIndex API Changes

## Overview

The `VectorIndex` has been refactored to handle embedding generation internally. This significantly simplifies the API and reduces boilerplate code.

## Key Changes

### 1. VectorIndex Requires Embeddings at Initialization

**Before:**
```python
index = VectorIndex(vector_store=vector_store)
```

**After:**
```python
index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings  # Required!
)
```

### 2. Automatic Embedding Generation

The index now automatically generates embeddings when adding documents.

**Before:**
```python
# Manual embedding generation required
embeddings_model = OpenAIEmbeddings(...)
doc_embeddings = await embeddings_model.embed_documents(texts)

nodes = [
    Node(text=text, embedding=embedding)
    for text, embedding in zip(texts, doc_embeddings)
]

index = VectorIndex(vector_store=vector_store)
await index.add_documents(nodes)
```

**After:**
```python
# Embeddings handled automatically!
embeddings_model = OpenAIEmbeddings(...)

nodes = [
    Node(text=text)  # No embedding needed!
    for text in texts
]

index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings_model
)
await index.add_documents(nodes)  # Embeddings auto-generated!
```

### 3. New `search_by_text()` Method

A convenience method for searching with text queries (auto-generates query embedding).

**New:**
```python
# Search with text directly
results = await index.search_by_text(
    query="What is machine learning?",
    k=5
)
```

**Equivalent to:**
```python
# Manual approach
query_embedding = await embeddings.embed_query("What is machine learning?")
results = await index.search(query_embedding, k=5)
```

### 4. Simplified Retriever Creation

The `as_retriever()` method no longer requires an embeddings parameter.

**Before:**
```python
retriever = index.as_retriever(
    embeddings=embeddings,  # Had to pass embeddings
    top_k=5
)
```

**After:**
```python
retriever = index.as_retriever(
    top_k=5  # Embeddings from index!
)
```

## Complete Example: Before vs After

### Before (Old API)

```python
import asyncio
from qdrant_client import QdrantClient
from rag_framework import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Node
)

async def main():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = await embeddings.aget_dimension()
    
    # Create documents
    texts = [
        "Python is a programming language.",
        "Machine learning uses algorithms."
    ]
    
    # MANUAL EMBEDDING GENERATION
    doc_embeddings = await embeddings.embed_documents(texts)
    
    nodes = [
        Node(text=text, embedding=emb)
        for text, emb in zip(texts, doc_embeddings)
    ]
    
    # Setup index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="docs",
        vector_size=dimension
    )
    
    index = VectorIndex(vector_store=vector_store)
    await index.add_documents(nodes)
    
    # Search
    query_emb = await embeddings.embed_query("What is Python?")
    results = await index.search(query_emb, k=2)
    
    # Create retriever
    retriever = index.as_retriever(
        embeddings=embeddings,
        top_k=2
    )
    results = await retriever.retrieve("What is Python?")

asyncio.run(main())
```

### After (New API)

```python
import asyncio
from qdrant_client import QdrantClient
from rag_framework import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Node
)

async def main():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = await embeddings.aget_dimension()
    
    # Create documents
    texts = [
        "Python is a programming language.",
        "Machine learning uses algorithms."
    ]
    
    # NO MANUAL EMBEDDING GENERATION!
    nodes = [
        Node(text=text)  # No embedding field!
        for text in texts
    ]
    
    # Setup index WITH embeddings
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="docs",
        vector_size=dimension
    )
    
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings  # Embeddings part of index!
    )
    
    # Auto-generates embeddings!
    await index.add_documents(nodes)
    
    # Search with text directly
    results = await index.search_by_text("What is Python?", k=2)
    
    # Create retriever (no embeddings param!)
    retriever = index.as_retriever(top_k=2)
    results = await retriever.retrieve("What is Python?")

asyncio.run(main())
```

## Benefits

### 1. Less Boilerplate
- No need to manually call `embed_documents()` before indexing
- No need to manage embeddings separately from documents

### 2. Cleaner Code
- Documents can be created without embeddings
- Embedding generation is handled transparently

### 3. More Intuitive
- VectorIndex owns the embedding model
- Single source of truth for embeddings

### 4. Flexibility
- Can still provide pre-computed embeddings if needed
- Auto-embedding can be disabled: `add_documents(nodes, auto_embed=False)`

## API Reference

### VectorIndex Constructor

```python
def __init__(
    self,
    vector_store: VectorStore[D],
    embeddings: Embeddings,  # NEW: Required
    index_id: Optional[str] = None
):
    """
    Initialize VectorIndex with a vector store and embeddings model.
    
    Args:
        vector_store: Vector store implementation
        embeddings: Embeddings model for generating document embeddings
        index_id: Unique identifier for this index
    """
```

### add_documents()

```python
async def add_documents(
    self,
    documents: List[D],
    auto_embed: bool = True  # NEW: Auto-generate embeddings
) -> List[str]:
    """
    Add documents to the index.
    
    Automatically generates embeddings for documents that don't have them.
    
    Args:
        documents: List of document objects to add
        auto_embed: If True, automatically generate embeddings for
                   documents without them (default: True)
    
    Returns:
        List of document IDs that were added
    """
```

### search_by_text() (NEW)

```python
async def search_by_text(
    self,
    query: str,  # Text query, not embedding!
    k: int = 4,
    resolve_parents: bool = True,
    **kwargs
) -> List[tuple[D, float]]:
    """
    Search for similar documents using a text query.
    Automatically generates the query embedding.
    
    Args:
        query: The query text
        k: Number of results to return
        resolve_parents: Resolve SymNode parents
        **kwargs: Additional search parameters
    
    Returns:
        List of (document, score) tuples
    """
```

### as_retriever()

```python
def as_retriever(
    self,
    top_k: int = 4,
    resolve_parents: bool = True,
    **search_kwargs
) -> VectorIndexRetriever:
    """
    Create a retriever from this index.
    
    Uses the index's embeddings model for query encoding.
    
    Args:
        top_k: Number of results to return
        resolve_parents: Resolve SymNode parents
        **search_kwargs: Additional search parameters
    
    Returns:
        A VectorIndexRetriever instance
    """
```

## Migration Guide

### Step 1: Update VectorIndex Initialization

Add the `embeddings` parameter to all `VectorIndex` constructors:

```python
# Old
index = VectorIndex(vector_store=vector_store)

# New
index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings
)
```

### Step 2: Remove Manual Embedding Generation

Remove calls to `embed_documents()` before adding to index:

```python
# Old
doc_embeddings = await embeddings.embed_documents(texts)
nodes = [Node(text=t, embedding=e) for t, e in zip(texts, doc_embeddings)]

# New
nodes = [Node(text=t) for t in texts]
```

### Step 3: Update Retriever Creation

Remove the `embeddings` parameter from `as_retriever()`:

```python
# Old
retriever = index.as_retriever(embeddings=embeddings, top_k=5)

# New
retriever = index.as_retriever(top_k=5)
```

### Step 4: (Optional) Use search_by_text()

Replace manual query embedding + search with `search_by_text()`:

```python
# Old
query_emb = await embeddings.embed_query("query")
results = await index.search(query_emb, k=5)

# New
results = await index.search_by_text("query", k=5)
```

## Advanced Usage

### Mixed Embedding Sources

You can mix documents with and without pre-computed embeddings:

```python
node1 = Node(text="Text 1")  # Will be auto-embedded
node2 = Node(text="Text 2", embedding=[0.1] * 1536)  # Pre-computed

await index.add_documents([node1, node2])
# Only node1 gets auto-embedded!
```

### Disable Auto-Embedding

```python
# All documents must have embeddings
await index.add_documents(nodes, auto_embed=False)
```

### Access the Embeddings Model

```python
# The embeddings model is accessible
dimension = await index.embeddings.aget_dimension()
custom_emb = await index.embeddings.embed_query("test")
```

## Backward Compatibility

The old `search()` method still works with query embeddings:

```python
# Still supported for low-level control
query_embedding = await embeddings.embed_query("query")
results = await index.search(query_embedding, k=5)
```

## Examples

See updated examples in:
- `src/examples/retriever_example.py` - Updated for new API
- `src/examples/symnode_example.py` - Works with new API
- `src/examples/embeddings_example.py` - Shows embedding integration

## Testing

All tests have been updated to use the new API. Run:

```bash
python -m pytest src/tests/ -v
```
