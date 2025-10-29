# SymNode - Hierarchical Node Relationships

## Overview

`SymNode` (Symbolic Node) is a powerful feature that enables hierarchical node relationships in the RAG framework. It allows you to:

1. **Create smaller, granular chunks** for precise semantic matching
2. **Automatically resolve to parent nodes** during retrieval for better context
3. **Maintain parent-child relationships** in your document hierarchy

## Key Concepts

### The Problem

When chunking documents, you often face a trade-off:
- **Small chunks**: Better semantic matching, but less context when retrieved
- **Large chunks**: More context, but less precise semantic matching

### The Solution

SymNode solves this by letting you:
1. Create large parent chunks (`Chunk`) with full context
2. Create smaller `SymNode` instances that reference the parent
3. Index the smaller SymNodes for precise matching
4. Automatically retrieve the parent chunks for full context

## Usage

### Basic Example

```python
import asyncio
from qdrant_client import QdrantClient
from rag_framework import (
    Chunk, SymNode, QdrantVectorStore, VectorIndex, OpenAIEmbeddings
)


async def main():
    # 1. Create a parent chunk with full context
    long_text = """
    Machine learning is a subset of artificial intelligence that enables 
    computers to learn from data without being explicitly programmed.
    """

    parent_chunk = Chunk.from_text(
        text=long_text.strip(),
        chunk_index=0,
        metadata={"topic": "ml", "source": "textbook"}
    )

    # 2. Create smaller SymNodes that reference the parent
    small_chunk1 = long_text[0:60]
    small_chunk2 = long_text[60:]

    sym_node1 = SymNode.create(
        text=small_chunk1,
        parent_id=parent_chunk.id,
        metadata=parent_chunk.metadata.copy()
    )

    sym_node2 = SymNode.create(
        text=small_chunk2,
        parent_id=parent_chunk.id,
        metadata=parent_chunk.metadata.copy()
    )

    # 3. Generate embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    parent_chunk.embedding = await embeddings.embed_query(parent_chunk.text)
    sym_node1.embedding = await embeddings.embed_query(sym_node1.text)
    sym_node2.embedding = await embeddings.embed_query(sym_node2.text)

    # 4. Setup vector store
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="hierarchical",
        vector_size=1536
    )
    index = VectorIndex(vector_store=vector_store)

    # 5. Index documents (parent first, then SymNodes)
    await index.insert_nodes([parent_chunk])
    await index.insert_nodes([sym_node1, sym_node2])

    # 6. Search with automatic parent resolution
    query_embedding = await embeddings.embed_query("What is machine learning?")
    results = await index.search(query_embedding, k=2)

    # Results will contain the parent Chunk, not the SymNodes!
    for doc, score in results:
        print(f"Type: {doc.__class__.__name__}")  # Chunk
        print(f"Text: {doc.text}")  # Full parent text
        print(f"Score: {score}")


asyncio.run(main())
```

### Using the Helper Method

The `Chunk` class provides a convenient helper method:

```python
# Create parent chunk
parent = Chunk.from_text(
    text="Python is a programming language. It is widely used.",
    metadata={"source": "docs"}
)

# Create SymNodes from sub-texts
sub_texts = [
    "Python is a programming language.",
    "It is widely used."
]
sym_nodes = parent.create_symbolic_nodes(sub_texts)

# All SymNodes automatically have:
# - parent_id set to parent.id
# - metadata copied from parent
# - is_symbolic flag set to True
```

## API Reference

### SymNode Class

```python
class SymNode(Node):
    """A symbolic node that references a parent node."""
    
    is_symbolic: bool = True
    
    @classmethod
    def create(
        cls,
        text: str,
        parent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'SymNode':
        """Create a SymNode with a parent reference."""
        
    def requires_parent_resolution(self) -> bool:
        """Check if parent should be resolved during retrieval."""
```

### VectorIndex.search()

```python
async def search(
    self,
    query_embedding: List[float],
    k: int = 4,
    resolve_parents: bool = True,  # ← Controls parent resolution
    **kwargs
) -> List[tuple[D, float]]:
    """
    Search for similar documents.
    
    Args:
        query_embedding: The query vector
        k: Number of results
        resolve_parents: If True, automatically resolve parent nodes 
                        for SymNode results (default: True)
    
    Returns:
        List of (document, score) tuples
    """
```

### Chunk.create_symbolic_nodes()

```python
def create_symbolic_nodes(
    self,
    sub_texts: List[str],
    preserve_metadata: bool = True
) -> List['SymNode']:
    """
    Create multiple SymNode instances that reference this chunk as parent.
    
    Args:
        sub_texts: List of text strings for the SymNodes
        preserve_metadata: Whether to copy this chunk's metadata
        
    Returns:
        List of SymNode instances
    """
```

## Important Notes

### 1. Parent Must Be Indexed First

The parent node **must** be added to the index before the SymNodes:

```python
# ✓ Correct order
await index.insert_nodes([parent_chunk])
await index.insert_nodes([sym_node1, sym_node2])

# ✗ Wrong - parent won't be found during resolution
await index.insert_nodes([sym_node1, sym_node2])
await index.insert_nodes([parent_chunk])
```

### 2. Deduplication

If multiple SymNodes point to the same parent, the parent will only appear **once** in the results:

```python
# Both SymNodes have same parent_id
sym1 = SymNode.create(text="...", parent_id=parent.id)
sym2 = SymNode.create(text="...", parent_id=parent.id)

# Search might match both SymNodes
results = await index.search(query_embedding, k=5)

# But results will contain parent only once
# (automatic deduplication)
```

### 3. Disabling Parent Resolution

You can disable parent resolution to see the actual SymNodes:

```python
# Get SymNodes instead of parents
results = await index.search(
    query_embedding, 
    k=5, 
    resolve_parents=False
)

for doc, score in results:
    if isinstance(doc, SymNode):
        print(f"Found SymNode: {doc.parent_id}")
```

## Use Cases

### 1. Hierarchical Document Chunking

Split documents into:
- **Parent chunks**: Paragraphs or sections (200-500 words)
- **SymNodes**: Sentences or sub-sections (20-50 words)

Index SymNodes for precise matching, retrieve parents for context.

### 2. Multi-Granularity Search

Create multiple levels:
- Document → Section (Chunk) → Sentence (SymNode)
- Section → Paragraph (Chunk) → Sentence (SymNode)

### 3. Code Documentation

- **Parent**: Full function documentation
- **SymNodes**: Individual parameter descriptions, return values, examples

Search matches specific details, retrieval returns full documentation.

## Performance Considerations

- **Storage**: SymNodes increase storage (more vectors indexed)
- **Search**: SymNodes improve precision (smaller, more focused chunks)
- **Retrieval**: Parent resolution adds one extra lookup per unique parent
- **Deduplication**: Built-in to prevent duplicate parents in results

## Example: Complete Workflow

See `src/examples/symnode_example.py` for a complete working example demonstrating:
- Basic SymNode creation and usage
- Hierarchical chunking strategy
- Parent resolution behavior
- Comparison with and without resolution

## Testing

Run the test suite to verify SymNode functionality:

```bash
python test_symnode_basic.py
```

Or with pytest (if installed):

```bash
pytest src/tests/test_symnode.py -v
```
