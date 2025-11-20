# Fetchcraft

A flexible and extensible framework for building Retrieval-Augmented Generation (RAG) applications with support for multiple vector stores, document stores, and advanced chunking strategies.

## Features

- 🎯 **Modular Architecture**: Abstract base classes for easy extension
- 🗂️ **Multiple Vector Stores**: Qdrant and ChromaDB support
- 🔢 **Multiple Indices**: Support multiple isolated indices in the same vector store
- 🤖 **Embedding Models**: Built-in OpenAI embeddings with extensible architecture
- 🔍 **Hybrid Search**: Combine dense (semantic) + sparse (keyword) vectors with RRF/DBSF fusion
- 📄 **Document Parsing**: Filesystem source with multiple chunking strategies
- 🧩 **Hierarchical Chunking**: Parent-child node relationships with SymNode support
- 🗄️ **Document Store**: MongoDB backend for full document persistence
- 🤖 **AI Agents**: ReAct agents with retriever and file search tools (powered by Pydantic AI)
- 📊 **Evaluation Framework**: Comprehensive retriever evaluation with metrics (MRR, NDCG, Hit Rate)
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
- pydantic-ai >= 0.0.14
- qdrant-client >= 1.15.1
- openai >= 1.0.0

### Optional Dependencies

- `chromadb` - For ChromaDB vector store
- `motor` - For MongoDB document store
- `fastembed` - For hybrid search support
- `mongomock-motor` - For testing MongoDB store

## Quick Start

### 1. Setup Embeddings and Vector Store

```python
import asyncio
from qdrant_client import QdrantClient
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import Node

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dimensions
    api_key="your-api-key"  # Optional: reads from OPENAI_API_KEY env var
)

# Create Qdrant client and vector store
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embeddings=embeddings,
    distance="Cosine"
)

# Create vector index
index = VectorIndex(vector_store=vector_store, index_id="my-index")
```

### 2. Load and Parse Documents

```python
from pathlib import Path
from fetchcraft.parsing import FilesystemDocumentParser
from fetchcraft.node_parser import SimpleNodeParser, HierarchicalNodeParser

# Load documents from directory
source = FilesystemDocumentParser.from_directory(
    directory=Path("documents"),
    pattern="*.txt",
    recursive=True
)

documents = []
async for doc in source.get_documents():
    documents.append(doc)

# Simple chunking
parser = SimpleNodeParser(chunk_size=512, overlap=50)
chunks = parser.get_nodes(documents)

# Or hierarchical chunking (creates parent-child relationships)
h_parser = HierarchicalNodeParser(
    chunk_size=2048,  # Parent size
    overlap=100,
    child_sizes=[512, 128],  # Child sizes
    child_overlap=20
)
nodes = h_parser.get_nodes(documents)
```

### 3. Index Documents

```python
# Add nodes to index (embeddings are auto-generated!)
doc_ids = await index.add_nodes(DocumentNode, chunks, show_progress=True)
print(f"Indexed {len(doc_ids)} chunks")
```

### 4. Search for Similar Documents

```python
# Search with text query (no manual embedding needed!)
results = await index.search_by_text("What is machine learning?", k=5)

for node, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {node.text[:100]}...")
    print(f"Metadata: {node.metadata}")
    print()

# Or use a retriever
retriever = index.as_retriever(top_k=5, resolve_parents=True)
results = await retriever.aretrieve("machine learning")
```

## Core Components

### Node and Chunk

The framework uses `Node` as the base persistent document type with full support for relationships and metadata. All Node properties are automatically stored in the vector store.

```python
from fetchcraft.node import Node, Chunk, SymNode

# Create a document node with relationships
document = Node(
    text="Full document text...",
    metadata={"title": "My Document", "author": "John Doe"}
    # Note: embedding is auto-generated when added to vector store
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
    metadata={"parsing": "document.txt"}
)
chunk1.parent = document
chunk1.embedding = [0.5, 0.6, ...]

chunk2 = Chunk.from_text(
    text="Second chunk of text",
    chunk_index=1,
    start_char_idx=20,
    end_char_idx=40,
    metadata={"parsing": "document.txt"}
)
chunk2.link_to_previous(chunk1)  # Creates bidirectional link
chunk2.embedding = [0.7, 0.8, ...]

# All properties are preserved: text, metadata, embedding, and relationships
# Navigate relationships
print(chunk2.has_previous())  # True
print(chunk2.previous.text)   # "First chunk of text" (if cached in memory)
print(chunk2.previous_id)     # chunk1.id (always available from storage)
```

### Document Source and Parsers

Load documents from filesystem and parse with various strategies:

```python
from pathlib import Path
from fetchcraft.parsing import FilesystemDocumentParser
from fetchcraft.node_parser import SimpleNodeParser, HierarchicalNodeParser

# Load from a single file
source = FilesystemDocumentParser.from_file(Path("document.txt"))
documents = [doc async for doc in source.get_documents()]

# Load from directory
source = FilesystemDocumentParser.from_directory(
    directory=Path("documents/"),
    pattern="*.txt",  # File pattern
    recursive=True  # Search subdirectories
)

documents = []
async for doc in source.get_documents():
    documents.append(doc)

# Parse with simple chunking
parser = SimpleNodeParser(chunk_size=500, overlap=50)
chunks = parser.get_nodes(documents)

# Or use hierarchical chunking for better context
h_parser = HierarchicalNodeParser(
    chunk_size=2048,
    overlap=100,
    child_sizes=[512, 128]
)
nodes = h_parser.get_nodes(documents)
```

### Embedding Models

The framework provides built-in support for generating embeddings with an extensible architecture.

#### OpenAI Embeddings

```python
from fetchcraft.embeddings import OpenAIEmbeddings

# Basic usage (reads API key from OPENAI_API_KEY env var)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 1536 dimensions
)

# With custom configuration
embeddings = OpenAIEmbeddings(
    api_key="your-api-key-here",
    model="text-embedding-3-large",  # 3072 dimensions
    base_url="https://api.openai.com/v1"  # Custom endpoint (Azure, local models, etc.)
)

# Embed documents
texts = ["Document 1", "Document 2", "Document 3"]
embeddings_list = await embeddings.embed_documents(texts)

# Embed a query
query_embedding = await embeddings.embed_query("search query")

# Get embedding dimension (determined lazily)
print(embeddings.dimension)  # e.g., 1536
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
from fetchcraft.embeddings import Embeddings
from typing import List

class MyCustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._dimension = 768
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Your implementation
        pass
    
    async def embed_query(self, text: str) -> List[float]:
        # Your implementation
        pass
    
    @property
    def dimension(self) -> int:
        return self._dimension
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
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.node import Chunk

# For Nodes (default)
node_store = QdrantVectorStore(
    client=client,
    collection_name="nodes",
    embeddings=embeddings
)

# For Chunks (preserves chunk-specific properties)
chunk_store = QdrantVectorStore(
    client=client,
    collection_name="chunks",
    embeddings=embeddings,
    document_class=Chunk
)
```

### Vector Store Abstraction

The framework supports multiple vector store backends.

```python
from fetchcraft.vector_store import QdrantVectorStore, ChromaVectorStore
from fetchcraft.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
import chromadb

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="documents",
    embeddings=embeddings,
    distance="Cosine"
)

# ChromaDB
chroma_client = chromadb.Client()
chroma_store = ChromaVectorStore(
    client=chroma_client,
    collection_name="documents",
    embeddings=embeddings
)
```

### Vector Index

The `VectorIndex` provides a high-level interface for working with vector stores.

```python
from fetchcraft.index.vector_index import VectorIndex

index = VectorIndex(vector_store=vector_store, index_id="my-index")

# Add documents (embeddings auto-generated)
ids = await index.add_nodes(DocumentNode, chunks)

# Search by text
results = await index.search_by_text("query", k=5)

# Get specific document
doc = await index.get_node(node_id="123")

# Delete documents
success = await index.delete_nodes(["id1", "id2"])
```

## Advanced Usage

### Hierarchical Chunking with Parent Resolution

Use hierarchical chunking to maintain context and resolve to parent documents:

```python
from fetchcraft.node_parser import HierarchicalNodeParser
from fetchcraft.node import SymNode

# Create hierarchical parser
parser = HierarchicalNodeParser(
    chunk_size=2048,  # Parent chunk size
    overlap=100,
    child_sizes=[512, 128],  # Create 2 levels of children
    child_overlap=20
)

# Parse documents
nodes = parser.get_nodes(documents)

# Index all nodes (parents and children)
await index.add_nodes(DocumentNode, nodes)

# Retrieve with parent resolution
retriever = index.as_retriever(top_k=5, resolve_parents=True)
results = await retriever.aretrieve("query")

# Results will include parent chunks for better context
for result in results:
    print(f"Text: {result.node.text[:100]}...")
    if hasattr(result.node, 'parent_id'):
        print(f"Has parent: {result.node.parent_id}")
```

### Working with SymNodes

SymNodes are symbolic references to parent chunks for efficient hierarchical retrieval:

```python
from fetchcraft.node import SymNode, NodeType

# SymNodes are created automatically by HierarchicalNodeParser
# They point to larger parent chunks while being small and searchable

# Check if a node is a SymNode
if node.node_type == NodeType.SYMNODE:
    print(f"This is a symbolic node pointing to parent: {node.parent_id}")
```

### Multiple Indices in the Same Vector Store

You can create multiple isolated indices within the same vector store, which is useful for multi-tenancy, environment separation, or organizing different types of content.

```python
from qdrant_client import QdrantClient
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.embeddings import OpenAIEmbeddings

# Create a shared vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="shared_collection",
    embeddings=embeddings
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
await tech_docs_index.add_nodes(DocumentNode, tech_chunks)
await marketing_index.add_nodes(DocumentNode, marketing_chunks)

# Searches are automatically isolated to each index
tech_results = await tech_docs_index.search_by_text("query", k=5)
marketing_results = await marketing_index.search_by_text("query", k=5)

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
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.parsing import FilesystemDocumentParser
from fetchcraft.node_parser import SimpleNodeParser


async def build_rag_index():
    # Step 1: Setup embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="knowledge_base",
        embeddings=embeddings
    )
    index = VectorIndex(vector_store=vector_store)

    # Step 2: Load documents
    source = FilesystemDocumentParser.from_file(Path("documents/knowledge_base.txt"))
    documents = [doc async for doc in source.get_documents()]

    # Step 3: Parse into chunks
    parser = SimpleNodeParser(chunk_size=500, overlap=50)
    chunks = parser.get_nodes(documents)

    # Step 4: Index chunks (embeddings auto-generated!)
    document_ids = await index.add_nodes(DocumentNode, chunks, show_progress=True)

    print(f"✓ Indexed {len(document_ids)} chunks")
    return index


async def search_knowledge_base(index, query: str):
    # Search with text query (no manual embedding needed!)
    results = await index.search_by_text(query, k=5)

    # Display results
    for i, (node, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Text: {node.text[:100]}...")
        print(f"   Source: {node.metadata.get('parsing', 'N/A')}")


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
from fetchcraft.vector_store.base import VectorStore
from fetchcraft.node import Node
from typing import List, Optional, Tuple


class MyVectorStore(VectorStore[Node]):
    def __init__(self, client, collection_name: str, embeddings, document_class=Node):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.document_class = document_class

    async def add_documents(
        self,
        documents: List[Node],
        index_id: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        # Implement adding documents to your backend
        # Remember to generate embeddings if not present
        pass

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        index_id: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[Node, float]]:
        # Implement similarity search
        pass

    async def delete(self, ids: List[str], index_id: Optional[str] = None) -> bool:
        # Implement deletion
        pass

    async def get_document(
        self,
        document_id: str,
        index_id: Optional[str] = None
    ) -> Optional[Node]:
        # Implement retrieval
        pass
```

### Creating a Custom Document Source

```python
from fetchcraft.parsing.base import DocumentParser
from fetchcraft.node import DocumentNode
from typing import AsyncIterator
from pathlib import Path


class CustomDocumentSource(DocumentParser):
    def __init__(self, source_path: Path):
        self.source_path = source_path

    async def get_documents(self) -> AsyncIterator[DocumentNode]:
        # Implement your document loading logic
        # Yield DocumentNode objects
        pass

    @classmethod
    def from_config(cls, config: dict) -> 'CustomDocumentSource':
        return cls(source_path=Path(config['path']))
```

## API Reference

### Core Classes

**Node**
- `id`: Unique identifier
- `text`: Content text
- `metadata`: Additional metadata
- `parent_id`, `next_id`, `prev_id`: Relationship IDs
- `children_ids`: List of child node IDs

**Chunk** (inherits from Node)
- `chunk_index`: Position in sequence
- `start_char_idx`: Start position in parent
- `end_char_idx`: End position in parent
- `doc_id`: Reference to parent document

**SymNode** (inherits from Chunk)
- `node_type`: Always `NodeType.SYMNODE` for symbolic nodes
- Used in hierarchical chunking for parent references
- Requires `parent_id` to be set

**DocumentNode** (inherits from Node)
- Represents a full document with children

### Vector Stores

**QdrantVectorStore**
- `add_documents(documents, index_id)`: Add documents with embeddings
- `similarity_search(query_embedding, k, index_id)`: Vector similarity search
- `enable_hybrid`: Enable hybrid search (dense + sparse)
- `fusion_method`: Fusion method ("rrf" or "dbsf")

**ChromaVectorStore**
- Same interface as QdrantVectorStore
- Supports in-memory and persistent modes

### VectorIndex
- `add_nodes(nodes, show_progress)`: Add nodes to index
- `search_by_text(query, k)`: Search with text query
- `search(query_embedding, k)`: Search with embedding
- `get_node(node_id)`: Retrieve specific node
- `delete_nodes(node_ids)`: Delete nodes
- `as_retriever(top_k, resolve_parents)`: Create retriever

### Document Sources

**FilesystemDocumentSource**
- `from_file(file_path)`: Load single file
- `from_directory(directory, pattern, recursive)`: Load directory
- `get_documents()`: Async iterator of DocumentNode objects

### Node Parsers

**SimpleNodeParser**
- `get_nodes(documents)`: Parse documents into chunks
- `chunk_size`: Maximum chunk size
- `overlap`: Overlap between chunks

**HierarchicalNodeParser**
- `get_nodes(documents)`: Create hierarchical structure
- `chunk_size`: Parent chunk size
- `child_sizes`: List of child chunk sizes
- Creates parent Chunks and child SymNodes

### Agents

**ReActAgent**
- `create(model, tools)`: Create agent with tools
- `query(question, messages)`: Query agent
- Returns AgentResponse with citations

**RetrieverTool**
- `from_retriever(retriever)`: Create from retriever
- `get_tool_function()`: Get Pydantic AI tool function

### Evaluation

**DatasetGenerator**
- `generate_dataset(num_documents, questions_per_node)`: Generate eval dataset
- `generate_from_specific_nodes(node_ids)`: Generate from specific nodes

**RetrieverEvaluator**
- `evaluate(dataset, show_progress)`: Evaluate retriever
- `get_failed_queries()`: Get failed queries
- `save_results(filepath)`: Save detailed results

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Advanced Features

### Hybrid Search

Combine dense (semantic) and sparse (keyword) vectors for better retrieval:

```python
from fetchcraft.vector_store import QdrantVectorStore

# Enable hybrid search with RRF fusion
vector_store = QdrantVectorStore(
    client=client,
    collection_name="hybrid_docs",
    embeddings=embeddings,
    enable_hybrid=True,
    fusion_method="rrf"  # or "dbsf"
)

# Search automatically uses hybrid mode
results = await index.search_by_text("machine learning", k=5)
```

### AI Agents with Pydantic AI

Build intelligent agents with retrieval capabilities:

```python
from pydantic_ai import Tool
from fetchcraft.agents import PydanticAgent, RetrieverTool

# Create retriever tool
retriever = index.as_retriever(top_k=3)
retriever_tool = RetrieverTool.from_retriever(retriever)
tools = [Tool(retriever_tool.get_tool_function(), takes_ctx=True)]

# Create ReAct agent
agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools
)

# Query the agent
response = await agent.query("What are the main concepts in the documents?")
print(response.response.content)
print(f"Citations: {len(response.citations)}")
```

### Retriever Evaluation

Evaluate your retriever's performance with comprehensive metrics:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from fetchcraft.evaluation import DatasetGenerator, RetrieverEvaluator

# Generate evaluation dataset
model = OpenAIChatModel(
    model_name="gpt-4-turbo",
    provider=OpenAIProvider(api_key="...")
)

generator = DatasetGenerator(model=model, document_store=doc_store, vector_store=vector_store)
dataset = await generator.generate_dataset(num_documents=50, questions_per_node=3)
dataset.save("eval_dataset.json")

# Evaluate retriever
evaluator = RetrieverEvaluator(retriever=retriever)
metrics = await evaluator.evaluate(dataset, show_progress=True)

print(f"Hit Rate@5: {metrics.hit_rate:.2%}")
print(f"MRR: {metrics.mrr:.4f}")
print(f"NDCG@5: {metrics.ndcg:.4f}")
```

### MongoDB Document Store

Store full documents alongside vector embeddings:

```python
from fetchcraft.document_store import MongoDBDocumentStore

# Create document store
doc_store = MongoDBDocumentStore(
    connection_string="mongodb://localhost:27017",
    database_name="fetchcraft",
    collection_name="documents"
)

# Store documents
await doc_store.add_documents(documents)

# Retrieve by doc_id (gets document + all its chunks)
nodes = await doc_store.get_documents_by_doc_id(doc.id)
```

## Examples

See the `src/examples/` directory for complete examples:

- `simple_usage.py` - Basic usage
- `document_processing_example.py` - Document parsing and chunking
- `hybrid_search_example.py` - Hybrid search configuration
- `agent_example.py` - ReAct agent with retrieval
- `evaluation/evaluate_retriever.py` - Full evaluation workflow
- `chroma_example.py` - Using ChromaDB vector store
- `embeddings_example.py` - Working with embeddings

## Documentation

- [Evaluation Module](src/fetchcraft/evaluation/README.md) - Detailed evaluation guide
- [Examples](src/examples/) - Code examples for all features

## Roadmap

- [x] Hybrid search (dense + sparse)
- [x] ChromaDB support
- [x] AI Agents with Pydantic AI
- [x] Evaluation framework
- [x] MongoDB document store
- [x] Hierarchical chunking
- [ ] Support for PDF and HTML document parsing
- [ ] Additional vector store backends (Pinecone, Weaviate)
- [ ] Query rewriting and expansion
- [ ] Document versioning and updates
- [ ] Batch processing optimizations
