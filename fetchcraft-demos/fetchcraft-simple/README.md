# Simple RAG Framework Demo

This demo showcases the complete capabilities of the RAG framework, including:

- 📂 **Document Loading**: Automatically load and parse text files from a directory
- 🔍 **Vector Indexing**: Index documents using Qdrant vector store
- 🤖 **Intelligent Agent**: ReAct agent with retrieval-augmented generation
- 💬 **Interactive REPL**: Ask questions and get answers based on your documents

## Prerequisites

1. **Qdrant Vector Database**
   - Must be running on `localhost:6333`
   - Start with Docker:
     ```bash
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
     ```

2. **OpenAI API or Compatible Endpoint**
   - Set `OPENAI_API_KEY` environment variable
   - Or configure a custom endpoint via `EMBEDDING_BASE_URL` and `LLM_BASE_URL`

3. **Python Dependencies**
   - All dependencies from the framework's `requirements.txt`
   - Specifically: `qdrant-client`, `pydantic-ai`, `openai`

## Configuration

The demo uses the following configuration (can be customized via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | `fetchcraft` | Qdrant collection name |
| `DOCUMENTS_PATH` | `/mnt/storage/data/knowledge/textfiles_tiny` | Path to text files |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `openai:gpt-4` | LLM model for the agent |
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `EMBEDDING_BASE_URL` | (optional) | Custom embedding API endpoint |

## Usage

### Basic Usage

From the project root directory:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Run the demo
python -m demo.simple_demo.run_demo
```

### Custom Configuration

```bash
# Use a custom embedding model and endpoint
export EMBEDDING_MODEL="text-embedding-ada-002"
export EMBEDDING_BASE_URL="http://localhost:8000/v1"

# Use a different LLM
export LLM_MODEL="openai:gpt-3.5-turbo"

# Run the demo
python -m demo.simple_demo.run_demo
```

### Using with Local Models

If you have a local OpenAI-compatible API (e.g., using Ollama, vLLM, or LocalAI):

```bash
export OPENAI_API_KEY="sk-dummy-key"
export EMBEDDING_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="openai:your-local-model"

python -m demo.simple_demo.run_demo
```

## How It Works

### 1. First Run (Collection Doesn't Exist)

When you run the demo for the first time:

1. ✅ Connects to Qdrant
2. ✅ Checks if collection `fetchcraft` exists
3. ✅ Since it doesn't exist, loads all `.txt` files from the documents directory
4. ✅ Splits documents into chunks (500 chars with 50 char overlap)
5. ✅ Generates embeddings for all chunks
6. ✅ Indexes chunks in Qdrant
7. ✅ Creates a ReAct agent with retrieval capabilities
8. ✅ Starts interactive REPL

### 2. Subsequent Runs (Collection Exists)

On subsequent runs:

1. ✅ Connects to Qdrant
2. ✅ Detects existing `fetchcraft` collection
3. ⏭️ **Skips document ingestion** (data already indexed)
4. ✅ Creates agent and starts REPL

### 3. Interactive Q&A

Once in the REPL, you can:

- Ask questions about your documents
- The agent will search the vector store for relevant chunks
- It will reason about the retrieved information
- Provide answers with citations

Example session:
```
❓ Your Question: What is the main topic of the documents?

🔍 Searching and reasoning...

──────────────────────────────────────────────────────────────────────
💬 Answer:
The documents primarily discuss [agent's answer based on your content]

──────────────────────────────────────────────────────────────────────
📚 Citations:
   [1] document1.txt (score: 0.875)
   [2] document2.txt (score: 0.832)
──────────────────────────────────────────────────────────────────────
```

## Document Format

The demo expects `.txt` files in the specified directory. The parser will:

- Recursively scan for `*.txt` files
- Split each file into chunks
- Preserve metadata (filename, source path)
- Link chunks to their parent documents

## Troubleshooting

### "Failed to connect to Qdrant"

**Solution**: Make sure Qdrant is running:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "Documents path does not exist"

**Solution**: Update `DOCUMENTS_PATH` in the script or create the directory:
```bash
mkdir -p /mnt/storage/data/knowledge/textfiles_tiny
# Add some .txt files to the directory
```

### "Failed to initialize embeddings"

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### "No text files found"

**Solution**: Make sure your documents directory contains `.txt` files:
```bash
ls /mnt/storage/data/knowledge/textfiles_tiny/*.txt
```

## Customization

You can customize the demo by modifying `run_demo.py`:

### Change Chunk Size

```python
num_chunks = await load_and_index_documents(
    vector_index=vector_index,
    documents_path=DOCUMENTS_PATH,
    chunk_size=1000,  # Larger chunks
    overlap=100       # More overlap
)
```

### Adjust Retrieval Parameters

```python
retriever = vector_index.as_retriever(
    top_k=5,              # Retrieve more documents
    resolve_parents=True  # Return full parent documents
)
```

### Use Different File Types

```python
results = parser.parse_directory(
    directory_path=documents_path,
    pattern="*.md",      # Markdown files
    recursive=True
)
```

## Architecture

```
┌─────────────────┐
│  Text Files     │
│  (.txt)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TextFileParser  │
│ (chunking)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OpenAI          │
│ Embeddings      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qdrant Vector   │
│ Store           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Index    │
│ + Retriever     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ReAct Agent     │
│ (pydantic-ai)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Interactive     │
│ REPL            │
└─────────────────┘
```

## Next Steps

After running this demo, you can:

1. **Explore the Code**: Review `run_demo.py` to understand the framework
2. **Add More Documents**: Drop more `.txt` files into the documents directory and re-run (delete the collection first to re-index)
3. **Customize the Agent**: Modify the system prompt or add custom tools
4. **Build Your Own**: Use this demo as a template for your own RAG application

## Clean Up

To start fresh:

```python
# Delete the collection in Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection("fetchcraft")
```

Or simply restart the Qdrant container:
```bash
docker restart <qdrant-container-id>
```

## License

This demo is part of the RAG Framework project.
