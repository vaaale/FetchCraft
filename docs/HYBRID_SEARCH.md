# Hybrid Search with QdrantVectorStore

## Overview

Hybrid search combines **dense vectors** (semantic search) with **sparse vectors** (keyword/BM25-style search) to provide the best of both worlds:

- **Dense vectors**: Understand semantic meaning and context
- **Sparse vectors**: Match specific keywords and terms precisely

This results in better search quality, especially for queries with specific terminology or when both semantic understanding and keyword precision are important.

## Installation

Hybrid search requires the `fastembed` library for generating sparse embeddings:

```bash
pip install fastembed
```

## Basic Usage

### Enable Hybrid Search

Simply set `enable_hybrid=True` when creating your QdrantVectorStore:

```python
from qdrant_client import QdrantClient
from fetchcraft import QdrantVectorStore, VectorIndex, OpenAIEmbeddings, Node

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create Qdrant client
client = QdrantClient(":memory:")

# Create vector store with hybrid search enabled
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embeddings=embeddings,
    enable_hybrid=True,  # 🔥 Enable hybrid search
    fusion_method="rrf"  # Choose fusion method
)

# Create index
index = VectorIndex(vector_store=vector_store)

# Index documents (automatically generates both dense and sparse vectors)
nodes = [Node(text="Your document text here")]
await index.add_nodes(nodes)

# Search (automatically uses hybrid search)
results = await index.search_by_text("your query", k=5)
```

## Configuration Options

### Fusion Methods

Qdrant supports two fusion methods for combining dense and sparse results:

#### 1. RRF (Reciprocal Rank Fusion) - Default

```python
vector_store = QdrantVectorStore(
    client=client,
    collection_name="collection",
    embeddings=embeddings,
    enable_hybrid=True,
    fusion_method="rrf"  # Position-based fusion
)
```

- **Best for**: Balanced results
- **How it works**: Considers the rank positions of results
- **Characteristics**: Simple, effective, well-tested

#### 2. DBSF (Distribution-Based Score Fusion)

```python
vector_store = QdrantVectorStore(
    client=client,
    collection_name="collection",
    embeddings=embeddings,
    enable_hybrid=True,
    fusion_method="dbsf"  # Score-based fusion
)
```

- **Best for**: Score-sensitive ranking
- **How it works**: Normalizes scores statistically (mean ± 3σ)
- **Characteristics**: More sophisticated, score-aware

### Sparse Model Selection

You can customize the sparse embedding model:

```python
vector_store = QdrantVectorStore(
    client=client,
    collection_name="collection",
    embeddings=embeddings,
    enable_hybrid=True,
    sparse_model="Qdrant/bm25"  # Default BM25-style model
)
```

Available sparse models from fastembed:
- `"Qdrant/bm25"` (default) - BM25-style sparse embeddings
- Other models supported by fastembed

## How It Works

### 1. Indexing

When you add documents with hybrid search enabled:

```python
await index.insert_nodes([Node(text="Python is a programming language")])
```

The system automatically:
1. Generates **dense embeddings** using your embeddings model (e.g., OpenAI)
2. Generates **sparse embeddings** using fastembed's BM25 model
3. Stores both in Qdrant with named vectors: `"dense"` and `"sparse"`

### 2. Searching

When you search with hybrid mode:

```python
results = await index.search_by_text("Python programming", k=5)
```

The system:
1. Generates dense embedding for the query
2. Generates sparse embedding for the query
3. Uses Qdrant's **prefetch + fusion** query to:
   - Search sparse vectors (keyword matching)
   - Search dense vectors (semantic matching)
   - Combine results using chosen fusion method
4. Returns unified, ranked results

### 3. Query Flow

```
User Query: "Python pip package"
           ↓
    ┌──────┴──────┐
    ↓             ↓
Dense Vector   Sparse Vector
(Semantic)     (Keywords)
    ↓             ↓
Dense Search   Sparse Search
  Results        Results
    ↓             ↓
    └──────┬──────┘
           ↓
    Fusion (RRF/DBSF)
           ↓
    Combined Results
```

## Collection Structure

Hybrid search creates a collection with both dense and sparse vectors:

```python
{
    "vectors": {
        "dense": {
            "size": 1536,  # Based on your embeddings model
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {
            "modifier": "IDF"  # BM25-style scoring
        }
    }
}
```

## Use Cases

### When to Use Hybrid Search

✅ **Use hybrid search when:**
- Queries contain specific keywords or terminology
- You need both semantic understanding AND keyword precision
- Domain-specific terms are important
- Technical documentation or code search
- Product names, model numbers, or identifiers matter

### When Dense-Only is Sufficient

❌ **Stick with dense-only when:**
- Queries are natural language questions
- Semantic understanding is more important than keywords
- Performance is critical (hybrid is slightly slower)
- Your documents don't have important specific keywords

## Performance Considerations

### Speed
- **Hybrid search** is slightly slower than dense-only (needs to search both vectors)
- Typical overhead: 1.5-2x query time
- Still very fast for most use cases

### Storage
- **Hybrid search** requires more storage (stores both dense and sparse vectors)
- Sparse vectors are typically smaller than dense
- Total increase: ~20-30% more storage

### Quality
- **Hybrid search** often provides better results for keyword-heavy queries
- Especially effective for technical content, documentation, and product catalogs

## Migration

### From Dense-Only to Hybrid

To add hybrid search to an existing collection:

1. **Create a new collection** with hybrid enabled
2. Re-index your documents
3. Update your application code

```python
# Old (dense-only)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="old_collection",
    embeddings=embeddings
)

# New (hybrid)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="new_hybrid_collection",
    embeddings=embeddings,
    enable_hybrid=True
)

# Re-index documents
await index.insert_nodes(documents)
```

## Configuration via QdrantConfig

You can also configure hybrid search using QdrantConfig:

```python
from fetchcraft import QdrantConfig, QdrantVectorStore

config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="my_collection",
    enable_hybrid=True,
    fusion_method="rrf"
)

vector_store = QdrantVectorStore.from_config(config, embeddings=embeddings)
```

## Troubleshooting

### ImportError: fastembed not found

```bash
pip install fastembed
```

### "query_text is required for hybrid search"

When using hybrid search, you must use `search_by_text()` or pass `query_text` to `search()`:

```python
# ✅ Correct
results = await index.search_by_text("your query", k=5)

# ✅ Also correct
query_embedding = await embeddings.embed_query("your query")
results = await index.search(query_embedding, k=5, query_text="your query")

# ❌ Wrong (missing query_text for hybrid)
results = await index.search(query_embedding, k=5)  # Error!
```

### Collection already exists without hybrid

If you try to enable hybrid on an existing dense-only collection, you'll need to:
1. Delete the old collection
2. Create a new one with hybrid enabled
3. Re-index your documents

## Examples

See `src/examples/hybrid_search_example.py` for complete working examples including:
- Basic hybrid search
- Comparison with dense-only search
- Fusion method comparison
- Real-world use cases

## References

- [Qdrant Hybrid Queries Documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [FastEmbed GitHub](https://github.com/qdrant/fastembed)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Distribution-Based Score Fusion](https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18)
