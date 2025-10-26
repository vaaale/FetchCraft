# Chunking Strategies

The framework provides flexible chunking strategies for splitting documents into searchable chunks.

## Overview

### ChunkingStrategy (Abstract Base)

All chunking strategies inherit from the `ChunkingStrategy` abstract base class and implement:

```python
def chunk_text(text: str, metadata: dict, parent_node: Node) -> List[Chunk | SymNode]
```

## Available Strategies

### 1. CharacterChunkingStrategy

Simple character-based chunking with overlap. Creates independent chunks at a single level.

**Use case**: Simple applications, small documents, when context is less important.

```python
from fetchcraft import CharacterChunkingStrategy, TextFileDocumentParser

chunker = CharacterChunkingStrategy(
    chunk_size=4096,
    overlap=200,
    separator=" ",
    keep_separator=True
)

parser = TextFileDocumentParser(chunker=chunker)
chunks = parser.parse_directory(path)
```

**Pros:**
- Simple and fast
- Predictable chunk sizes
- Easy to understand

**Cons:**
- May split at arbitrary positions
- No semantic awareness
- Less context on retrieval

### 2. HierarchicalChunkingStrategy ⭐ (Default & Recommended)

Advanced multi-level chunking with parent-child relationships and recursive splitting.

**Use case**: Most applications, especially when you need both precision and context.

```python
from fetchcraft import HierarchicalChunkingStrategy, TextFileDocumentParser

chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,           # Parent chunk size (for context)
    overlap=200,
    child_chunks=[1024, 512, 256],  # Multiple child sizes for search
    child_overlap=50
)

parser = TextFileDocumentParser(chunker=chunker)
nodes = parser.parse_directory(path, pattern="*", recursive=True)
```

## Hierarchical Chunking: How It Works

### Architecture

```
Document (4096 chars)
├── Parent Chunk 1 (4096 chars)  ← Returned on retrieval
│   ├── Child 1A (1024 chars)   ← Indexed
│   ├── Child 1B (1024 chars)   ← Indexed
│   ├── Child 1C (512 chars)    ← Indexed
│   ├── Child 1D (512 chars)    ← Indexed
│   ├── Child 1E (256 chars)    ← Indexed
│   └── Child 1F (256 chars)    ← Indexed
└── Parent Chunk 2 (4096 chars)
    └── ... (similar children)
```

### Recursive Splitting

The strategy uses semantic boundaries in order of priority:

1. **Paragraph** (`\n\n`) - Preserves document structure
2. **Line break** (`\n`) - Respects formatting
3. **Sentence** (`. `, `? `, `! `) - Maintains meaning
4. **Clause** (`; `) - Keeps related ideas together
5. **Phrase** (`, `) - Preserves phrases
6. **Word** (` `) - Fall back to word boundaries
7. **Character** - Last resort

**Example:**

```python
text = """
Paragraph one. Multiple sentences!

Paragraph two.
With line breaks.

Final paragraph.
"""

chunker = HierarchicalChunkingStrategy(chunk_size=100, overlap=10)
# Will first try to split at \n\n (paragraphs)
# Then at \n (lines) if chunks are still too large
# Then at . ! ? (sentences) if needed
# And so on...
```

### Benefits

**1. Multi-Granularity Search**

Different child sizes capture different levels of detail:
- **Large children (1024)**: Good for broader context
- **Medium children (512)**: Balanced precision/context
- **Small children (256)**: Precise, specific matches

**2. Context Preservation**

When any child node matches a query, the **entire parent chunk** is returned, providing:
- Full context around the match
- Related information before/after the match
- Better understanding for the LLM

**3. Semantic Boundaries**

Recursive splitting ensures chunks don't break:
- Mid-paragraph
- Mid-sentence  
- Mid-word

**4. Better Retrieval Quality**

- Small child nodes → precise semantic search
- Large parent chunks → rich context for generation
- Best of both worlds!

## Usage Examples

### Basic Usage (Defaults)

```python
from fetchcraft import TextFileDocumentParser

# Uses HierarchicalChunkingStrategy by default
# Parent: 4096, Child: [512]
parser = TextFileDocumentParser()
chunks = parser.parse_directory(path)
```

### Custom Hierarchy

```python
from fetchcraft import HierarchicalChunkingStrategy, TextFileDocumentParser

# Three levels of child chunks
chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,
    overlap=200,
    child_chunks=[1024, 512, 256],  # 3 levels
    child_overlap=50
)

parser = TextFileDocumentParser(chunker=chunker)
chunks = parser.parse_directory(path, pattern="*.txt", recursive=True)
```

### Custom Separators

```python
# Use custom separators for domain-specific text
chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,
    overlap=200,
    child_chunks=[512],
    child_overlap=50,
    separators=["\n\n\n", "\n\n", "\n", ". ", " "]  # Custom order
)
```

### Character Chunking (Simple)

```python
from fetchcraft import CharacterChunkingStrategy, TextFileDocumentParser

chunker = CharacterChunkingStrategy(chunk_size=4096, overlap=200)
parser = TextFileDocumentParser(chunker=chunker)
chunks = parser.parse(text)
```

## Indexing & Retrieval

### What Gets Indexed

**HierarchicalChunkingStrategy:**
- ✅ Child SymNodes (all sizes) → indexed for search
- ✅ Parent Chunks → stored but NOT directly searched
- When a child matches, parent is automatically resolved and returned

**CharacterChunkingStrategy:**
- ✅ All chunks → indexed and searchable

### Example Flow

```python
# 1. Create chunks
chunker = HierarchicalChunkingStrategy(
    chunk_size=4096,
    child_chunks=[1024, 512, 256]
)
parser = TextFileDocumentParser(chunker=chunker)
chunks = parser.parse_directory("docs/")

# 2. Index (includes parents and children)
await vector_index.add_documents(chunks)

# 3. Search (searches child nodes)
results = await vector_index.search_by_text("machine learning")

# 4. Results contain parent chunks (full context!)
for doc, score in results:
    print(doc.text)  # This is the parent chunk (4096 chars)
```

## Choosing a Strategy

| Feature | Character | Hierarchical |
|---------|-----------|--------------|
| **Simplicity** | ✅ Very simple | ⚠️ More complex |
| **Search Precision** | ⚠️ Depends on chunk size | ✅ Excellent (small children) |
| **Context on Retrieval** | ⚠️ Limited to chunk size | ✅ Excellent (large parents) |
| **Semantic Boundaries** | ❌ No | ✅ Yes (recursive) |
| **Multi-granularity** | ❌ No | ✅ Yes (multiple child sizes) |
| **Index Size** | ⚠️ Moderate | ⚠️ Larger (more nodes) |
| **Recommended For** | Simple apps | Most applications |

## Performance Considerations

### Index Size

**CharacterChunkingStrategy:**
- Document (10,000 chars) → ~3 chunks (4096 size) → **3 vectors**

**HierarchicalChunkingStrategy:**
- Document (10,000 chars) → 3 parent chunks
  - Each parent → ~4 children per size × 3 sizes = 12 children
  - Total: 3 parents + 36 children = **39 vectors**

Trade-off: More vectors = better retrieval quality but larger index.

### Chunking Speed

- **Character**: Very fast (simple splitting)
- **Hierarchical**: Slower (recursive algorithm), but still fast enough for most use cases

## Best Practices

1. **Start with defaults**: `TextFileDocumentParser()` uses sensible hierarchical defaults

2. **Adjust parent size** based on your documents:
   - Short documents (articles): 2048-4096
   - Long documents (books): 4096-8192
   - Very long documents: 8192-16384

3. **Child sizes** should provide coverage:
   - Include at least one small size (256-512) for precision
   - Include at least one medium size (512-1024) for balance
   - Optionally include larger sizes for broader matches

4. **Overlap** prevents information loss:
   - Parent overlap: 10-20% of chunk_size
   - Child overlap: 5-10% of child_chunk_size

5. **Custom separators** for domain-specific text:
   - Code: Add `\n\n\n` (3 newlines) for function boundaries
   - Legal: Add section markers
   - Scientific: Add figure/table markers

## Examples

See:
- `/examples/chunking_example.py` - Basic demonstrations
- `/examples/test_recursive_splitting.py` - Verify recursive splitting
- `/demo/simple_demo/run_demo.py` - Full RAG pipeline with hierarchical chunking

## API Reference

### CharacterChunkingStrategy

```python
CharacterChunkingStrategy(
    chunk_size: int = 4096,
    overlap: int = 200,
    separator: str = " ",
    keep_separator: bool = True
)
```

### HierarchicalChunkingStrategy

```python
HierarchicalChunkingStrategy(
    chunk_size: int = 4096,              # Parent chunk size
    overlap: int = 200,                   # Parent overlap
    child_chunks: List[int] = None,      # Child sizes (default: [chunk_size // 8])
    child_overlap: int = 50,              # Child overlap
    separators: List[str] = None,        # Custom separators (optional)
    keep_separator: bool = True           # Keep separators in chunks
)
```

**Default separators:** `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`

### TextFileDocumentParser

```python
TextFileDocumentParser(
    chunker: ChunkingStrategy = None,    # Default: HierarchicalChunkingStrategy
    chunk_size: int = 4096,              # Used if chunker is None
    overlap: int = 200,                   # Used if chunker is None
    separator: str = " ",                 # Used if chunker is None
    keep_separator: bool = True           # Used if chunker is None
)
```
