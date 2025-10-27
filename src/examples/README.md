# Examples

This directory contains examples demonstrating the fetchcraft framework features.

## Chunking Strategy Examples

### 1. `chunking_example.py`

Basic demonstration of different chunking strategies.

**Shows:**
- Character chunking (simple, single-level)
- Hierarchical chunking (multi-level with parent-child)
- Comparison between strategies

**Run:**
```bash
python -m examples.chunking_example
```

**No dependencies:** Works without external services.

---

### 2. `test_recursive_splitting.py`

Tests and demonstrates recursive splitting with semantic boundaries.

**Shows:**
- Recursive splitting algorithm
- Semantic boundary detection (paragraph → line → sentence → word)
- Parent-child relationship validation
- Multiple child size levels

**Run:**
```bash
python -m examples.test_recursive_splitting
```

**No dependencies:** Works without external services.

---

### 3. `quick_hierarchy_test.py` ⭐

Quick verification that hierarchical chunking returns correct chunk sizes.

**Shows:**
- Creating hierarchical chunks with multiple child sizes
- Indexing in vector store
- Retrieval with parent resolution
- **Verification that retrieved chunks are parent chunks (correct size)**

**Run:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key"

python -m examples.quick_hierarchy_test
```

**Dependencies:**
- Qdrant (uses in-memory mode, no installation needed)
- OpenAI API key (for embeddings)

**Expected output:**
```
✅ SUCCESS: Hierarchical chunking works correctly!
   - All results are parent chunks (not children)
   - All sizes match parent chunk expectations
   - Child nodes were searched, but parents returned
```

---

### 4. `hierarchical_retrieval_test.py`

Comprehensive test with detailed validation and comparison.

**Shows:**
- Complete end-to-end workflow
- Multiple documents with hierarchical chunking
- Detailed retrieval verification
- Size distribution analysis
- Comparison with character chunking

**Run:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key"

python -m examples.hierarchical_retrieval_test
```

**Dependencies:**
- Qdrant (uses in-memory mode)
- OpenAI API key (for embeddings)

**What it tests:**
1. ✅ Parent chunks are created at specified size
2. ✅ Multiple child sizes are created (e.g., [1024, 512, 256])
3. ✅ Child nodes reference their parents correctly
4. ✅ Retrieval returns parent chunks (not children)
5. ✅ Retrieved chunk sizes match parent size expectations
6. ✅ Comparison shows hierarchical provides more context

---

### 5. `chroma_example.py` 🆕

Comprehensive ChromaDB vector store examples.

**Shows:**
- Using ChromaDB as a vector store backend
- In-memory and persistent storage modes
- Different distance metrics (cosine, L2, inner product)
- Hierarchical chunking with ChromaDB
- Metadata filtering

**Run:**
```bash
# Install ChromaDB first
pip install chromadb

# Set your OpenAI API key
export OPENAI_API_KEY="your-key"

python -m examples.chroma_example
```

**Dependencies:**
- ChromaDB (`pip install chromadb`)
- OpenAI API key (for embeddings)

**What it demonstrates:**
1. ✅ Basic ChromaDB usage (in-memory)
2. ✅ Persistent storage configuration
3. ✅ Hierarchical chunking integration
4. ✅ Distance metric comparison
5. ✅ Configuration-based setup

---

## Vector Store Examples

### ChromaDB vs Qdrant

Both vector stores are supported. Choose based on your needs:

| Feature | ChromaDB | Qdrant |
|---------|----------|--------|
| Setup | `pip install chromadb` | `pip install qdrant-client` |
| In-memory | ✅ Easy | ✅ Easy |
| Persistent | ✅ Simple | ✅ Advanced |
| Self-hosted | ✅ Yes | ✅ Yes |
| Cloud option | ✅ Yes | ✅ Yes |
| Best for | Quick start, local dev | Production, scalability |

**Try both:**
```bash
# ChromaDB example
python -m examples.chroma_example

# Qdrant examples (already in quick_hierarchy_test, etc.)
python -m examples.quick_hierarchy_test
```

---

## Running the Examples

### Without External Services

These examples work without any setup:
```bash
python -m examples.chunking_example
python -m examples.test_recursive_splitting
```

### With Vector Store (OpenAI API Key Required)

These examples need an OpenAI API key for embeddings:

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Quick test (recommended)
python -m examples.quick_hierarchy_test

# Comprehensive test
python -m examples.hierarchical_retrieval_test
```

### With Custom Embeddings

Use custom embedding endpoint:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export EMBEDDING_MODEL="bge-m3"

python -m examples.quick_hierarchy_test
```

---

## What to Expect

### Quick Hierarchy Test

```
🔍 Quick Hierarchical Chunking Test

1. Creating hierarchical chunks:
   Parent size: 500 chars
   Child sizes: [150, 75]

   ✓ Created 2 parents, 18 children
   ✓ Total nodes to index: 20

2. Indexing in vector store...
   ✓ Indexed 20 nodes

3. Testing retrieval with resolve_parents=True...
   ✓ Retrieved 3 results

4. Verifying results:
   ✓ Result 1: Parent chunk
      Size: 487 chars ✓
      Preview: Machine learning is a method of data analysis...

   ✓ Result 2: Parent chunk
      Size: 512 chars ✓
      Preview: Deep learning is a subset of machine learning...

======================================================================
✅ SUCCESS: Hierarchical chunking works correctly!
   - All results are parent chunks (not children)
   - All sizes match parent chunk expectations
   - Child nodes were searched, but parents returned
======================================================================
```

### Key Verification Points

The tests verify:

1. **Chunk Creation**
   - Parent chunks created at specified size
   - Multiple child sizes created per parent
   - All nodes linked correctly

2. **Indexing**
   - All nodes (parents + children) indexed
   - No errors during indexing

3. **Retrieval** ⭐ **Most Important**
   - Results are **parent chunks** (not child nodes)
   - Sizes match parent chunk size (e.g., 500 chars, not 150 or 75)
   - This proves child nodes are searched, but parents returned

4. **Parent Resolution**
   - `resolve_parents=True` works correctly
   - Child → Parent mapping is preserved
   - Full context provided on retrieval

---

## Troubleshooting

### "No API key found"

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Connection refused" (Qdrant)

These examples use in-memory Qdrant (`:memory:`), so no Qdrant server needed!

### "Module not found"

Run from the project root:
```bash
cd /home/alex/PycharmProjects/MyIndex
python -m examples.quick_hierarchy_test
```

### Verify installation

```bash
pip install -e .
python -c "from fetchcraft import HierarchicalChunkingStrategy; print('OK')"
```

---

## Next Steps

After running the examples:

1. **Read the documentation**: `/docs/CHUNKING_STRATEGIES.md`
2. **Try the demo**: `/demo/simple_demo/run_demo.py`
3. **Build your application**: Use `HierarchicalChunkingStrategy` in your code

```python
from fetchcraft import HierarchicalChunkingStrategy, TextFileDocumentParser

# Multi-level hierarchy for best results
chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,
    child_chunks=[1024, 512, 256],
    child_overlap=50
)

parser = TextFileDocumentParser(chunker=chunker)
chunks = parser.parse_directory("docs/", pattern="*", recursive=True)
```
