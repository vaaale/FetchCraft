# RAG Framework

A flexible and extensible framework for building Retrieval-Augmented Generation (RAG) applications with support for multiple vector store backends.

## Features

- 🎯 **Modular Architecture**: Abstract base classes for easy extension
- 🗂️ **Multiple Vector Store Support**: Currently supports Qdrant (more backends can be added)
- 🔢 **Multiple Indices**: Support multiple isolated indices in the same vector store
- 🤖 **Embedding Models**: Built-in support for OpenAI embeddings with extensible architecture
- 🔍 **Hybrid Search**: Combine dense (semantic) + sparse (keyword) vectors for better results
- 📄 **Document Parsing**: Built-in text file parser with chunking strategies
- 🔗 **Relationship Management**: Node-based system with parent/child and sequential relationships
- 🧩 **Smart Chunking**: Configurable chunk size with overlap support
- ⚡ **Async-First API**: Built for high-performance applications
- 🔒 **Type-Safe**: Full type hints with Pydantic validation

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- pydantic >= 2.0.0
- qdrant-client >= 1.15.1
- openai >= 1.0.0

## Quick Start

### 1. Generate Embeddings

```python
from rag_framework import OpenAIEmbeddings

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dimensions
    # api_key="your-key",  # Optional: reads from OPENAI_API_KEY env var
)

# Embed a query
query_embedding = await embeddings.embed_query("What is machine learning?")

# Embed multiple documents
texts = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = await embeddings.embed_documents(texts)

print(f"Embedding dimension: {embeddings.dimension}")
```

### 2. Parse Documents into Chunks

```python
from pathlib import Path
from rag_framework import TextFileDocumentParser

# Parse a text file into chunks
chunks = TextFileDocumentParser.from_file(
    file_path=Path("documents/my_document.txt"),
    chunk_size=200,
    overlap=20
)

print(f"Created {len(chunks)} chunks")

# Add embeddings to chunks
chunk_texts = [chunk.text for chunk in chunks]
chunk_embeddings = await embeddings.embed_documents(chunk_texts)

for chunk, embedding in zip(chunks, chunk_embeddings):
    chunk.embedding = embedding
```

### 3. Create a Vector Store with Qdrant

```python
from qdrant_client import QdrantClient
from rag_framework import QdrantVectorStore, Node, Chunk

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Create vector store (uses Node as document class by default)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    vector_size=384
)

# Or specify Chunk if you want to store Chunk-specific properties
chunk_vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_chunks",
    document_class=Chunk,
    vector_size=384
)
```

### 4. Build a Vector Index

```python
from rag_framework import VectorIndex

# Create index with the vector store
index = VectorIndex(vector_store=vector_store)

# Add your chunks (with embeddings)
document_ids = await index.insert_nodes(chunks)
print(f"Indexed {len(document_ids)} chunks")
```

### 5. Search for Similar Documents

```python
# Search with a query embedding
query_embedding = [0.1, 0.2, ...]  # Your embedding vector

results = await index.search(
    query_embedding=query_embedding,
    k=5
)

for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")
    print()
```

## Core Components

### Node and Chunk

The framework uses `Node` as the base persistent document type with full support for relationships and metadata. All Node properties are automatically stored in the vector store.

```python
from rag_framework import Node, Chunk

# Create a document node with relationships
document = Node(
    text="Full document text...",
    metadata={"title": "My Document", "author": "John Doe"},
    embedding=[0.1, 0.2, ...]  # Your embedding vector
)

# Nodes preserve relationship IDs when stored
child_node = Node(
    text="Child document",
    metadata={"type": "child"},
    embedding=[0.3, 0.4, ...],
    parent_id=document.id,  # Relationship preserved in storage
    next_id="some-next-id"
)

# Create chunks with Chunk-specific properties
chunk1 = Chunk.from_text(
    text="First chunk of text",
    chunk_index=0,
    start_char_idx=0,
    end_char_idx=20,
    metadata={"source": "document.txt"}
)
chunk1.parent = document
chunk1.embedding = [0.5, 0.6, ...]

chunk2 = Chunk.from_text(
    text="Second chunk of text",
    chunk_index=1,
    start_char_idx=20,
    end_char_idx=40,
    metadata={"source": "document.txt"}
)
chunk2.link_to_previous(chunk1)  # Creates bidirectional link
chunk2.embedding = [0.7, 0.8, ...]

# All properties are preserved: text, metadata, embedding, and relationships
# Navigate relationships
print(chunk2.has_previous())  # True
print(chunk2.previous.text)   # "First chunk of text" (if cached in memory)
print(chunk2.previous_id)     # chunk1.id (always available from storage)
```

### Document Parser

The `TextFileDocumentParser` handles text file parsing with intelligent chunking.

```python
from pathlib import Path
from rag_framework import TextFileDocumentParser

# Parse a single file
chunks = TextFileDocumentParser.from_file(
    file_path=Path("document.txt"),
    chunk_size=500,      # Max characters per chunk
    overlap=50,          # Overlap between chunks
    separator=" ",       # Split on spaces
    keep_separator=True  # Keep spaces in chunks
)

# Parse an entire directory
results = TextFileDocumentParser.parse_directory(
    directory_path=Path("documents/"),
    chunk_size=500,
    overlap=50,
    pattern="*.txt",     # File pattern
    recursive=True       # Search subdirectories
)

for file_path, chunks in results.items():
    print(f"{file_path}: {len(chunks)} chunks")
```

### Embedding Models

The framework provides built-in support for generating embeddings with an extensible architecture.

#### OpenAI Embeddings

```python
from rag_framework import OpenAIEmbeddings

# Basic usage (reads API key from OPENAI_API_KEY env var)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 1536 dimensions
)

# With custom configuration
embeddings = OpenAIEmbeddings(
    api_key="your-api-key-here",
    model="text-embedding-3-large",  # 3072 dimensions
    dimensions=1024,  # Optional: reduce dimensions
    base_url="https://api.openai.com/v1",  # Custom endpoint
    timeout=60.0
)

# Embed documents
texts = ["Document 1", "Document 2", "Document 3"]
embeddings_list = await embeddings.embed_documents(texts)

# Embed a query
query_embedding = await embeddings.embed_query("search query")

# Get embedding dimension (determined lazily on first API call)
print(embeddings.dimension)  # e.g., 1536

# Or explicitly determine dimension in async context
dimension = await embeddings.aget_dimension()
```

**Supported Models:**
- `text-embedding-3-small` - 1536 dimensions, efficient and cost-effective
- `text-embedding-3-large` - 3072 dimensions, highest quality
- `text-embedding-ada-002` - 1536 dimensions, previous generation

**Dimension Determination:**
Embedding dimensions are determined lazily - the actual dimension is discovered on the first API call or when explicitly requested. This allows the framework to work with any embedding model without hardcoded dimension mappings.

```python
# Dimension determined on first embed call
embeddings = OpenAIEmbeddings(model="custom-model")
result = await embeddings.embed_query("test")  # Dimension determined here
print(embeddings.dimension)  # Returns actual dimension from API

# Or explicitly determine in async context
dimension = await embeddings.aget_dimension()

# Or provide dimension explicitly to skip API call
embeddings = OpenAIEmbeddings(model="custom-model", dimensions=1024)
```

**Custom Endpoints:**
The OpenAI embeddings class supports any OpenAI-compatible API endpoint, including:
- Azure OpenAI
- Local embedding models (via LiteLLM, Ollama, etc.)
- Other OpenAI-compatible services

#### Extending with Custom Embeddings

Create your own embedding implementation by inheriting from the `Embeddings` base class:

```python
from rag_framework import Embeddings
from typing import List

class MyCustomEmbeddings(Embeddings):
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Your implementation
        pass
    
    async def embed_query(self, text: str) -> List[float]:
        # Your implementation
        pass
    
    @property
    def dimension(self) -> int:
        return 768  # Your embedding dimension
```

### Choosing Between Node and Chunk

**Use `Node` when:**
- Storing general documents or content
- You only need basic properties (text, metadata, embedding, relationships)
- You want maximum flexibility

**Use `Chunk` when:**
- Storing document fragments with position information
- You need `chunk_index`, `start_char_idx`, `end_char_idx` properties
- Working with parsed documents from `TextFileDocumentParser`

**Important:** Specify `document_class=Chunk` when creating the vector store if you want Chunk-specific properties to be preserved upon retrieval.

```python
# For Nodes (default)
node_store = QdrantVectorStore(client, "nodes", vector_size=384)

# For Chunks
chunk_store = QdrantVectorStore(client, "chunks", document_class=Chunk, vector_size=384)
```

### Vector Store Abstraction

The framework provides a flexible abstraction for vector stores, making it easy to switch backends.

```python
from rag_framework import VectorStore, QdrantVectorStore, QdrantConfig

# Option 1: Direct initialization
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="documents",
    document_class=Chunk,
    vector_size=384,
    distance="Cosine"
)

# Option 2: From configuration
config = QdrantConfig(
    url="http://localhost:6333",
    collection_name="documents",
    vector_size=384,
    distance="Cosine"
)
vector_store = QdrantVectorStore.from_config(config)
```

### Vector Index

The `VectorIndex` provides a high-level interface for working with vector stores.

```python
from rag_framework import VectorIndex

index = VectorIndex(vector_store=vector_store)

# Add documents
ids = await index.insert_nodes(chunks)

# Search
results = await index.search(query_embedding, k=5)

# Get specific document
doc = await index.get_node(node_id="123")

# Delete documents
success = await index.delete_documents(["id1", "id2"])
```

## Advanced Usage

### Custom Chunking Strategy

```python
from rag_framework import TextFileDocumentParser

parser = TextFileDocumentParser(
    chunk_size=1000,
    overlap=100,
    separator="\n\n",    # Split on paragraphs
    keep_separator=False
)

text = Path("document.txt").read_text()
chunks = parser.parse(
    text=text,
    metadata={"category": "technical"}
)
```

### Working with Context

Chunks can retrieve surrounding context for better understanding.

```python
# Get surrounding chunks
chunk = chunks[5]
context = chunk.get_surrounding_context(
    num_chunks_before=2,
    num_chunks_after=2
)
print(context)

# Get text with parent document context
text_with_context = chunk.get_text_with_context(include_parent=True)
```

### Multiple Indices in the Same Vector Store

You can create multiple isolated indices within the same vector store, which is useful for multi-tenancy, environment separation, or organizing different types of content.

```python
from qdrant_client import QdrantClient
from rag_framework import VectorIndex, QdrantVectorStore, Chunk

# Create a shared vector store
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="shared_collection",
    document_class=Chunk,
    vector_size=384
)

# Create multiple indices with unique identifiers
tech_docs_index = VectorIndex(
    vector_store=vector_store,
    index_id="tech_docs"
)

marketing_index = VectorIndex(
    vector_store=vector_store,
    index_id="marketing_content"
)

support_index = VectorIndex(
    vector_store=vector_store,
    index_id="customer_support"
)

# Each index operates independently
await tech_docs_index.insert_nodes(tech_chunks)
await marketing_index.insert_nodes(marketing_chunks)

# Searches are automatically isolated to each index
tech_results = await tech_docs_index.search(query_embedding, k=5)
marketing_results = await marketing_index.search(query_embedding, k=5)

# Documents from one index are not accessible from another
doc = await tech_docs_index.get_node(doc_id)  # ✓ Found
doc = await marketing_index.get_node(doc_id)  # ✗ Returns None (isolated)
```

**Use Cases for Multiple Indices:**
- **Multi-tenant applications**: Each tenant has their own isolated index
- **Environment separation**: Separate dev, staging, and production data
- **Content organization**: Different indices for different document types
- **Language-specific indices**: Separate indices per language
- **Version control**: Maintain multiple document versions

If you don't specify an `index_id`, a UUID will be automatically generated, ensuring uniqueness.

### Complete RAG Pipeline Example

```python
import asyncio
from pathlib import Path
from qdrant_client import QdrantClient
from rag_framework import (
    TextFileDocumentParser,
    QdrantVectorStore,
    VectorIndex,
    Chunk
)


async def build_rag_index():
    # Step 1: Parse documents
    chunks = TextFileDocumentParser.from_file(
        file_path=Path("documents/knowledge_base.txt"),
        chunk_size=500,
        overlap=50
    )

    # Step 2: Generate embeddings (using your embedding model)
    for chunk in chunks:
        # Replace with your actual embedding generation
        chunk.embedding = generate_embedding(chunk.text)

    # Step 3: Create vector store
    client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="knowledge_base",
        document_class=Chunk,
        vector_size=384
    )

    # Step 4: Build index
    index = VectorIndex(vector_store=vector_store)
    document_ids = await index.insert_nodes(chunks)

    print(f"✓ Indexed {len(document_ids)} chunks")
    return index


async def search_knowledge_base(index, query: str):
    # Generate query embedding
    query_embedding = generate_embedding(query)

    # Search
    results = await index.search(query_embedding, k=5)

    # Display results
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Text: {chunk.text[:100]}...")
        if chunk.has_parent():
            print(f"   Source: {chunk.metadata.get('source', 'N/A')}")


# Run the pipeline
async def main():
    index = await build_rag_index()
    await search_knowledge_base(index, "What is machine learning?")


if __name__ == "__main__":
    asyncio.run(main())
```

## Extending the Framework

### Adding a New Vector Store Backend

To add support for a new vector store (e.g., Pinecone, Weaviate):

```python
from rag_framework import VectorStore
from typing import List, Optional, Dict, Any

class MyVectorStore(VectorStore[D]):
    def __init__(self, client, collection_name: str, document_class):
        self.client = client
        self.collection_name = collection_name
        self.document_class = document_class
    
    async def add_documents(self, documents: List[D]) -> List[str]:
        # Implement adding documents to your backend
        pass
    
    async def similarity_search(
        self, 
        query_embedding: List[float],
        k: int = 4,
        **kwargs
    ) -> List[tuple[D, float]]:
        # Implement similarity search
        pass
    
    async def delete(self, ids: List[str]) -> bool:
        # Implement deletion
        pass
    
    async def get_document(self, document_id: str) -> Optional[D]:
        # Implement retrieval
        pass
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MyVectorStore':
        # Implement configuration-based initialization
        pass
```

### Creating a Custom Document Parser

```python
from rag_framework import DocumentParser, Chunk
from pathlib import Path
from typing import List

class MarkdownDocumentParser(DocumentParser):
    def parse(self, text: str, **kwargs) -> List[Chunk]:
        # Implement markdown-specific parsing logic
        pass
    
    @classmethod
    def from_file(cls, file_path: Path, **kwargs) -> List[Chunk]:
        text = file_path.read_text()
        parser = cls()
        return parser.parse(text, **kwargs)
```

## API Reference

### Node
- `id`: Unique identifier
- `text`: Content text
- `metadata`: Additional metadata
- `embedding`: Optional embedding vector
- `parent`: Parent node reference
- `next`: Next node in sequence
- `previous`: Previous node in sequence

### Chunk (inherits from Node)
- `chunk_index`: Position in sequence
- `start_char_idx`: Start position in parent
- `end_char_idx`: End position in parent
- `link_to_previous(chunk)`: Link to previous chunk
- `get_surrounding_context(num_before, num_after)`: Get context

### VectorStore (Abstract)
- `add_documents(documents, index_id=None)`: Add documents to a specific index
- `similarity_search(query_embedding, k, index_id=None)`: Search within a specific index
- `delete(ids, index_id=None)`: Delete documents from a specific index
- `get_document(document_id, index_id=None)`: Get document from a specific index
- `from_config(config)`: Create from config

### VectorIndex
- `__init__(vector_store, index_id=None)`: Initialize with optional index identifier
- `index_id`: Unique identifier for this index
- `add_documents(documents)`: Add to this index
- `search(query_embedding, k)`: Search within this index
- `get_document(document_id)`: Retrieve document from this index
- `delete_documents(document_ids)`: Remove documents from this index
- `store`: Access the underlying vector store

### DocumentParser (Abstract)
- `parse(**kwargs)`: Parse document
- `from_file(file_path, **kwargs)`: Parse from file

### TextFileDocumentParser
- `from_file(file_path, chunk_size, overlap, ...)`: Parse file
- `parse_directory(directory_path, pattern, ...)`: Parse directory

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [ ] Add more vector store backends (Pinecone, Weaviate, ChromaDB)
- [ ] Support for PDF and HTML document parsing
- [ ] Built-in embedding generation
- [ ] Query rewriting and expansion
- [ ] Hybrid search (dense + sparse)
- [ ] Document versioning and updates
- [ ] Batch processing optimizations
- [ ] Metrics and monitoring
