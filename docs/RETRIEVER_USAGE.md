# Retriever - High-Level Retrieval Abstraction

## Overview

The `Retriever` abstraction provides a simple, high-level interface for retrieving documents using natural language queries. Unlike working directly with embeddings and vector stores, retrievers handle all the embedding generation internally.

## Key Benefits

1. **Simple API**: Just pass text queries, no need to generate embeddings manually
2. **Configurable**: Set defaults for `top_k`, parent resolution, and other parameters
3. **Flexible**: Override parameters on a per-query basis
4. **Consistent**: Works seamlessly with SymNode parent resolution

## Basic Usage

### Example Matching Your Request

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
    # Setup embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create and populate index (abbreviated)
    index = VectorIndex(...)
    # ... add documents to index ...
    
    # Create retriever from index
    retriever = index.as_retriever(
        embeddings=embeddings,
        top_k=2
    )
    
    # Retrieve with natural language query
    results = await retriever.retrieve("an interesting book about RAG")
    
    for doc, score in results:
        print(f"[{score:.3f}] {doc.text}")

asyncio.run(main())
```

### Complete Working Example

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
    # 1. Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="your-api-key"
    )
    dimension = await embeddings.aget_dimension()
    
    # 2. Create sample documents
    documents_text = [
        "RAG combines retrieval and generation.",
        "Vector databases store embeddings.",
        "Semantic search finds meaning, not keywords."
    ]
    
    # 3. Generate embeddings and create nodes
    doc_embeddings = await embeddings.embed_documents(documents_text)
    nodes = [
        Node(text=text, embedding=emb)
        for text, emb in zip(documents_text, doc_embeddings)
    ]
    
    # 4. Setup vector store and index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="docs",
        vector_size=dimension
    )
    index = VectorIndex(vector_store=vector_store)
    await index.add_documents(nodes)
    
    # 5. Create retriever
    retriever = index.as_retriever(
        embeddings=embeddings,
        top_k=2
    )
    
    # 6. Retrieve!
    results = await retriever.retrieve("What is RAG?")
    
    for doc, score in results:
        print(f"[Score: {score:.3f}] {doc.text}")

asyncio.run(main())
```

## API Reference

### VectorIndex.as_retriever()

```python
def as_retriever(
    self,
    embeddings: Embeddings,
    top_k: int = 4,
    resolve_parents: bool = True,
    **search_kwargs
) -> VectorIndexRetriever:
    """
    Create a retriever from this index.
    
    Args:
        embeddings: The embeddings model for query encoding
        top_k: Number of results to return (default: 4)
        resolve_parents: Resolve SymNode parents (default: True)
        **search_kwargs: Additional search parameters
        
    Returns:
        A VectorIndexRetriever instance
    """
```

### VectorIndexRetriever.retrieve()

```python
async def retrieve(
    self, 
    query: str,
    top_k: Optional[int] = None,
    **kwargs
) -> List[tuple[Document, float]]:
    """
    Retrieve documents based on a text query.
    
    Args:
        query: The query text
        top_k: Override default top_k for this query
        **kwargs: Additional search parameters
        
    Returns:
        List of (document, score) tuples
    """
```

### VectorIndexRetriever.update_config()

```python
def update_config(
    self,
    top_k: Optional[int] = None,
    resolve_parents: Optional[bool] = None,
    **search_kwargs
) -> None:
    """
    Update retriever configuration.
    
    Args:
        top_k: New default for top_k
        resolve_parents: New default for resolve_parents
        **search_kwargs: Additional search parameters to update
    """
```

## Configuration Options

### Setting Defaults

```python
# Create retriever with defaults
retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=5,                    # Return top 5 results
    resolve_parents=True        # Resolve SymNode parents
)
```

### Overriding Per Query

```python
# Use defaults
results = await retriever.retrieve("query")  # top_k=5

# Override for this query only
results = await retriever.retrieve("query", top_k=10)  # top_k=10
```

### Updating Configuration

```python
# Update the retriever's defaults
retriever.update_config(top_k=3)

# All subsequent queries use new default
results = await retriever.retrieve("query")  # top_k=3
```

## Working with SymNode

The retriever automatically handles SymNode parent resolution:

```python
from rag_framework import Chunk, SymNode

# Create parent and SymNodes
parent = Chunk.from_text(text="Long parent text with full context...")
parent.embedding = await embeddings.embed_query(parent.text)

sym_nodes = parent.create_symbolic_nodes([
    "Short fragment 1",
    "Short fragment 2"
])

# Add embeddings to SymNodes
for sym in sym_nodes:
    sym.embedding = await embeddings.embed_query(sym.text)

# Index parent first, then SymNodes
await index.add_documents([parent])
await index.add_documents(sym_nodes)

# Create retriever with parent resolution (default)
retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=3,
    resolve_parents=True  # Returns parent chunks, not SymNodes
)

# Retrieve - gets parent chunks with full context
results = await retriever.retrieve("query about fragments")

for doc, score in results:
    print(type(doc))  # Chunk (parent), not SymNode
```

### Disable Parent Resolution

```python
# Get SymNodes instead of parents
retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=3,
    resolve_parents=False  # Returns actual SymNodes
)

results = await retriever.retrieve("query")

for doc, score in results:
    if isinstance(doc, SymNode):
        print(f"SymNode pointing to parent: {doc.parent_id}")
```

## Direct Instantiation

You can also create a retriever directly:

```python
from rag_framework import VectorIndexRetriever

retriever = VectorIndexRetriever(
    vector_index=index,
    embeddings=embeddings,
    top_k=5,
    resolve_parents=True
)

results = await retriever.retrieve("query")
```

## Synchronous Usage

While not recommended, you can use the synchronous wrapper:

```python
# ⚠️ Only works when no async loop is running
results = retriever.retrieve_sync("query")
```

**Note**: This will raise an error if called from within an async context.

## Comparison: Before and After

### Before (Manual Embedding Generation)

```python
# Generate embedding manually
query_embedding = await embeddings.embed_query("What is RAG?")

# Search with embedding
results = await index.search(
    query_embedding=query_embedding,
    k=5,
    resolve_parents=True
)
```

### After (Using Retriever)

```python
# Just pass the text!
results = await retriever.retrieve("What is RAG?")
```

## Advanced Usage

### Custom Search Parameters

```python
# Pass additional parameters to the underlying search
retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=5,
    score_threshold=0.7  # Custom parameter
)

# Or override per query
results = await retriever.retrieve(
    "query",
    score_threshold=0.8
)
```

### Multiple Retrievers

Create different retrievers for different use cases:

```python
# High-precision retriever (fewer results)
precise_retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=2
)

# Broad retriever (more results)
broad_retriever = index.as_retriever(
    embeddings=embeddings,
    top_k=10
)

# Use appropriate retriever
if need_precision:
    results = await precise_retriever.retrieve(query)
else:
    results = await broad_retriever.retrieve(query)
```

## Examples

See the following files for complete working examples:

- **`src/examples/retriever_example.py`**: Comprehensive examples covering:
  - Basic retriever usage
  - Configuration options
  - SymNode integration
  - Direct instantiation

## Testing

Run the test suite:

```bash
# With pytest (if installed)
pytest src/tests/test_retriever.py -v
```

## Best Practices

1. **Reuse retrievers**: Create once, use many times
2. **Set sensible defaults**: Configure common parameters in `as_retriever()`
3. **Override when needed**: Use per-query overrides for special cases
4. **Use parent resolution**: Keep `resolve_parents=True` for better context
5. **Handle embeddings externally**: The retriever uses your embeddings model

## Performance Considerations

- **Embedding generation**: Each `retrieve()` call generates one query embedding
- **Caching**: Consider caching query embeddings for repeated queries
- **Batch queries**: For multiple queries, consider batch embedding generation
- **Parent resolution**: Adds minimal overhead (one lookup per unique parent)

## Future Enhancements

Potential future retriever types:
- **HybridRetriever**: Combines dense and sparse retrieval
- **RerankingRetriever**: Adds reranking step after initial retrieval
- **MultiIndexRetriever**: Searches across multiple indices
- **FilteredRetriever**: Adds metadata filtering
