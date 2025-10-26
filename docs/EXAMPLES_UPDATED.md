# Examples Updated for New VectorIndex API

All examples have been updated to use the new VectorIndex API where embeddings are handled automatically.

## Updated Examples

### ✅ `src/examples/simple_usage.py` 
- **NEW**: Clean demonstration of the simplified API
- Shows automatic embedding generation
- Demonstrates both search_by_text() and retriever usage

### ✅ `src/examples/retriever_example.py`
- Updated to use VectorIndex with embeddings at initialization
- No manual embedding generation in examples
- Simplified as_retriever() calls (no embeddings parameter)

### ✅ `src/examples/embeddings_example.py`
- Updated rag_pipeline_example() - auto-embeddings
- Updated document_parsing_with_embeddings() - auto-embeddings
- Uses search_by_text() for queries

### ✅ `src/examples/symnode_example.py`
- Updated basic_symnode_example() - auto-embeddings
- Updated hierarchical_chunking_example() - auto-embeddings
- Demonstrates parent resolution with new API

## Key Changes Across All Examples

### Before (Old API):
```python
# Manual embedding generation
embeddings = OpenAIEmbeddings(...)
doc_embeddings = await embeddings.embed_documents(texts)

nodes = [
    Node(text=text, embedding=emb)
    for text, emb in zip(texts, doc_embeddings)
]

index = VectorIndex(vector_store=vector_store)
await index.add_documents(nodes)

# Manual query embedding
query_emb = await embeddings.embed_query("query")
results = await index.search(query_emb, k=5)
```

### After (New API):
```python
# No manual embedding generation!
embeddings = OpenAIEmbeddings(...)

nodes = [
    Node(text=text)  # No embeddings needed!
    for text in texts
]

index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings  # Embeddings are part of index
)
await index.add_documents(nodes)  # Auto-generates embeddings!

# Direct text search
results = await index.search_by_text("query", k=5)
```

## Test Files

### ✅ All Test Files Updated!

- ✅ `src/tests/test_retriever.py` - Updated with MockEmbeddings
- ✅ `src/tests/test_symnode.py` - Updated with MockEmbeddings
- ✅ `src/tests/examples/multiple_indices_example.py` - Updated with MockEmbeddings
- ✅ `src/tests/examples/test_node_persistence.py` - Updated with MockEmbeddings

All tests passing and using the new VectorIndex API!

## Running Updated Examples

```bash
# Simple usage - best starting point
python src/examples/simple_usage.py

# Retriever examples
python src/examples/retriever_example.py

# Embeddings integration
python src/examples/embeddings_example.py

# SymNode hierarchical nodes
python src/examples/symnode_example.py
```

## Migration Checklist for Your Code

When updating your own code to the new API:

- [ ] Add `embeddings` parameter to `VectorIndex()` constructor
- [ ] Remove manual `embed_documents()` calls before adding documents
- [ ] Remove `embedding` field from Node creation (optional - can keep pre-computed)
- [ ] Update `as_retriever()` calls - remove `embeddings` parameter
- [ ] Replace manual embedding + search with `search_by_text()`
- [ ] Update from_vector_store() calls to include embeddings

## Benefits of New API

1. **Less code** - No manual embedding management
2. **Clearer intent** - Index owns the embedding model
3. **Fewer errors** - Can't forget to generate embeddings
4. **Better ergonomics** - Natural text-based search
5. **Flexible** - Still supports pre-computed embeddings

## Documentation

See these guides for more details:
- `VECTORINDEX_API_CHANGES.md` - Complete API changes
- `README_NEW_API.md` - Quick start guide
- `RETRIEVER_USAGE.md` - Retriever abstraction
- `SYMNODE_USAGE.md` - Hierarchical nodes
