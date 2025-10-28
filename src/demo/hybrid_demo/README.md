# Hybrid Search RAG Framework Demo

This demo showcases the **HYBRID SEARCH** capabilities of the RAG framework, combining dense (semantic) and sparse (keyword) vectors for superior search quality.

## 🔥 What is Hybrid Search?

Hybrid search combines two complementary approaches:

- **Dense Vectors (Semantic)**: Understand meaning and context via embeddings (e.g., OpenAI)
- **Sparse Vectors (Keyword)**: Match specific terms and keywords via BM25-style embeddings

**Result**: Best of both worlds - semantic understanding PLUS precise keyword matching!

## Key Features

- 🎯 **Better Results**: Especially for technical content, product catalogs, code documentation
- 🔍 **Keyword Precision**: Finds exact terms, model numbers, IDs, technical jargon
- 💡 **Semantic Understanding**: Still understands meaning and context
- ⚡ **RRF Fusion**: Reciprocal Rank Fusion combines both search types optimally
- 📊 **Production Ready**: Fully tested, documented, and optimized

## Prerequisites

### 1. Qdrant Vector Database
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. FastEmbed (Required for Hybrid Search)
```bash
pip install fastembed
```

### 3. OpenAI API or Compatible Endpoint
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 4. Python Dependencies
```bash
pip install qdrant-client pydantic-ai openai fastembed
```

## Quick Start

### Basic Usage

```bash
# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# Run the hybrid search demo
python -m demo.hybrid_demo.run_demo
```

### With Custom Configuration

```bash
# Use different models
export EMBEDDING_MODEL="text-embedding-3-small"
export LLM_MODEL="gpt-4-turbo"

# Configure hybrid search
export ENABLE_HYBRID="true"
export FUSION_METHOD="rrf"  # or "dbsf"

# Run the demo
python -m demo.hybrid_demo.run_demo
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | `fetchcraft_hybrid` | Qdrant collection name (different from standard demo) |
| `DOCUMENTS_PATH` | `Documents` | Path to text files |
| `EMBEDDING_MODEL` | `bge-m3` | Dense embedding model |
| `LLM_MODEL` | `gpt-4-turbo` | LLM model for the agent |
| `ENABLE_HYBRID` | `true` | Enable hybrid search (set to `false` for dense-only) |
| `FUSION_METHOD` | `rrf` | Fusion method: `rrf` or `dbsf` |
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |

## How It Works

### Indexing Phase

When you index documents with hybrid search:

```
📄 Document
    ↓
 Chunker
    ↓
 ┌──────┴──────┐
 ↓             ↓
Dense Emb.   Sparse Emb.
(OpenAI)     (FastEmbed)
 ↓             ↓
 └──────┬──────┘
        ↓
   Qdrant Store
```

Each chunk gets **TWO** vectors:
1. **Dense**: `[0.1, 0.2, ..., 0.9]` (1536 dims for OpenAI)
2. **Sparse**: `{indices: [1, 42, 100], values: [0.8, 0.5, 0.3]}` (BM25-style)

### Search Phase

When you query with hybrid search:

```
🔍 Query: "Python pip package"
    ↓
 ┌──────┴──────┐
 ↓             ↓
Dense Search  Sparse Search
(semantic)    (keywords)
 ↓             ↓
Results A     Results B
 ↓             ↓
 └──────┬──────┘
        ↓
   RRF Fusion
        ↓
Combined Results
(best of both!)
```

The system:
1. Generates dense embedding for query (semantic)
2. Generates sparse embedding for query (keywords)
3. Searches both vector types in parallel
4. Fuses results using RRF or DBSF
5. Returns top-k unified, ranked results

## Fusion Methods

### RRF (Reciprocal Rank Fusion) - Default

```bash
export FUSION_METHOD="rrf"
```

- **Best for**: Balanced results
- **How it works**: Considers rank positions of results
- **Characteristics**: Simple, effective, well-tested

### DBSF (Distribution-Based Score Fusion)

```bash
export FUSION_METHOD="dbsf"
```

- **Best for**: Score-sensitive ranking
- **How it works**: Normalizes scores statistically (mean ± 3σ)
- **Characteristics**: More sophisticated, score-aware

## Example Session

```bash
$ python -m demo.hybrid_demo.run_demo

================================================================================
🚀 RAG Framework Demo - HYBRID SEARCH MODE
================================================================================

💡 Hybrid Search = Dense (semantic) + Sparse (keyword) vectors
   Benefits: Better results for technical terms and specific keywords
================================================================================

1️⃣  Initializing embeddings...
   ✓ Dense embeddings initialized: bge-m3 (dimension: 1024)

2️⃣  Connecting to Qdrant at localhost:6333...
   ✓ Connected to Qdrant

3️⃣  Checking collection 'fetchcraft_hybrid'...
   ⚠️  Collection 'fetchcraft_hybrid' does not exist - will create and index

🔥 Creating vector store with HYBRID SEARCH...
   • Enable Hybrid: True
   • Fusion Method: RRF
   ✓ Vector store created with hybrid search enabled!

4️⃣  Indexing documents with hybrid search...
   Each chunk will have:
     🎯 Dense vector (semantic understanding)
     🔍 Sparse vector (keyword matching)
   ✓ Indexed 150 chunks with hybrid search!

================================================================================
✅ HYBRID SEARCH RAG System Ready!
================================================================================

❓ Your Question: What is pip?

🔍 Searching with hybrid search (dense + sparse vectors)...

──────────────────────────────────────────────────────────────────────
💬 Answer:
pip is the package installer for Python. It allows you to install and 
manage additional libraries and dependencies that are not part of the 
Python standard library.
──────────────────────────────────────────────────────────────────────
📚 Citations (ranked by hybrid search):
   [1] python_guide.txt (score: 0.892)
   [2] package_managers.txt (score: 0.834)
   [3] installation.txt (score: 0.801)
──────────────────────────────────────────────────────────────────────
```

## When to Use Hybrid Search

### ✅ Use Hybrid Search For:

- **Technical Documentation** - API docs, SDKs, programming guides
- **Code Search** - Function names, class names, specific code patterns
- **Product Catalogs** - SKUs, model numbers, product IDs
- **Scientific Papers** - Citations, formula names, specific terminology
- **Legal Documents** - Case numbers, statute references, specific clauses
- **Medical Records** - Drug names, procedure codes, diagnoses
- **Any content with important specific keywords or identifiers**

### ❌ Dense-Only May Be Sufficient For:

- Natural language Q&A
- Creative writing or literature
- General conversational content
- When speed is more critical than precision

## Performance Comparison

| Metric | Dense-Only | Hybrid | Notes |
|--------|------------|--------|-------|
| Query Speed | 1.0x | ~1.5-2x | Slightly slower but acceptable |
| Storage | 1.0x | ~1.2-1.3x | 20-30% more storage |
| Quality (keywords) | Good | **Excellent** | Much better for specific terms |
| Quality (semantic) | Excellent | **Excellent** | Maintains semantic understanding |

## Troubleshooting

### "fastembed is required for hybrid search"

**Solution**: Install fastembed:
```bash
pip install fastembed
```

### "query_text is required for hybrid search"

This shouldn't happen in the demo (we use `search_by_text`), but if you see it:

**Solution**: The demo automatically handles this. If you're writing custom code, use:
```python
results = await index.search_by_text("your query", k=5)
```

### Collection Already Exists

To start fresh and re-index with hybrid search:

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection("fetchcraft_hybrid")
```

Then run the demo again.

### Slow Indexing

First-time indexing with hybrid search downloads the sparse embedding model (~50MB):
- This happens once
- Subsequent runs are faster
- Model is cached locally

## Architecture

```
┌─────────────────────┐
│  Text Files (.txt)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TextFileParser     │
│  (chunking)         │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│ OpenAI  │  │FastEmbed│
│  Dense  │  │ Sparse  │
└────┬────┘  └────┬────┘
     │            │
     └──────┬─────┘
            ▼
┌─────────────────────┐
│ Qdrant Vector Store │
│  - Dense vectors    │
│  - Sparse vectors   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Vector Index      │
│   + Retriever       │
│  (Hybrid Search)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ReAct Agent       │
│   (pydantic-ai)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Interactive REPL    │
│ (Hybrid Queries)    │
└─────────────────────┘
```

## Comparison with Simple Demo

| Feature | Simple Demo | Hybrid Demo |
|---------|-------------|-------------|
| Search Type | Dense-only | Dense + Sparse |
| Collection Name | `fetchcraft` | `fetchcraft_hybrid` |
| Extra Dependency | - | `fastembed` |
| Query Speed | Faster | ~1.5-2x slower |
| Keyword Matching | Good | Excellent |
| Semantic Understanding | Excellent | Excellent |
| Best For | General Q&A | Technical content |

## Customization

### Disable Hybrid Search (Use Dense-Only)

```bash
export ENABLE_HYBRID="false"
python -m demo.hybrid_demo.run_demo
```

### Try Different Fusion Methods

```bash
# RRF (default) - position-based
export FUSION_METHOD="rrf"

# DBSF - score-based
export FUSION_METHOD="dbsf"

python -m demo.hybrid_demo.run_demo
```

### Adjust Retrieval Parameters

Edit `run_demo.py`:

```python
retriever = vector_index.as_retriever(
    top_k=5,              # Retrieve more documents
    resolve_parents=True  # Return full parent documents
)
```

## Next Steps

After running this demo:

1. **Compare Results**: Run the same queries in both `simple_demo` and `hybrid_demo` to see the difference
2. **Experiment with Keywords**: Try queries with specific technical terms, model numbers, or product IDs
3. **Measure Performance**: Use `time` to measure query speeds
4. **Build Your Own**: Use this demo as a template for hybrid search in your applications

## Clean Up

To remove the hybrid search collection:

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection("fetchcraft_hybrid")
```

Or restart Qdrant:
```bash
docker restart <qdrant-container-id>
```

## Resources

- **Framework Documentation**: See main README
- **Hybrid Search Guide**: See examples/hybrid_search_example.py
- **Qdrant Docs**: https://qdrant.tech/documentation/concepts/hybrid-queries/
- **FastEmbed**: https://github.com/qdrant/fastembed

## License

This demo is part of the RAG Framework project.
