# Hybrid Search - Quick Start Guide

## 30-Second Setup

```python
from qdrant_client import QdrantClient
from fetchcraft import QdrantVectorStore, VectorIndex, OpenAIEmbeddings, Node

# 1. Install fastembed
# pip install fastembed

# 2. Create hybrid-enabled store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore(
    client=QdrantClient(":memory:"),
    collection_name="docs",
    embeddings=embeddings,
    enable_hybrid=True  # 🔥 That's it!
)

# 3. Use normally
index = VectorIndex(vector_store=vector_store)
await index.add_documents([Node(text="Your docs")])
results = await index.search_by_text("query", k=5)
```

## What You Get

✅ **Better search quality** - Combines semantic + keyword matching  
✅ **No code changes** - Same API as before  
✅ **Automatic** - Handles embeddings transparently  

## Configuration Options

### Fusion Method

```python
# RRF (default) - Position-based
fusion_method="rrf"

# DBSF - Score-based
fusion_method="dbsf"
```

### Sparse Model

```python
sparse_model="Qdrant/bm25"  # Default BM25
```

## When to Use

✅ Use hybrid when:
- Technical docs, code, product names
- Keywords matter (model numbers, IDs)
- Domain-specific terminology

❌ Stick with dense when:
- Natural language Q&A
- Speed is critical
- Simple queries

## Requirements

```bash
pip install fastembed
```

## Performance

- **Speed**: ~1.5-2x slower (still fast)
- **Storage**: ~20-30% more
- **Quality**: Better for keyword queries

## Full Examples

See:
- `src/examples/hybrid_search_example.py`
- `HYBRID_SEARCH.md` for detailed guide
- `HYBRID_SEARCH_IMPLEMENTATION.md` for technical details

## Troubleshooting

**ImportError: fastembed**
```bash
pip install fastembed
```

**"query_text is required"**
```python
# Use search_by_text (recommended)
results = await index.search_by_text("query", k=5)

# Or pass query_text manually
results = await index.search(embedding, k=5, query_text="query")
```

**Collection already exists**
- Create new collection with hybrid enabled
- Re-index your documents

## That's It!

Enable with `enable_hybrid=True` and enjoy better search results! 🚀
