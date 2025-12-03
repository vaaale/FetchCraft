# Fetchcraft MCP Server

An MCP (Model Context Protocol) server that provides tools for document search and retrieval using RAG (Retrieval-Augmented Generation).

**Now includes a beautiful web interface!** The server comes with a modern single-page application for semantic file search with pagination.

## Features

This MCP server exposes three main tools via MCP protocol and a REST API:

### 1. `query` - RAG-based Question Answering
Query the RAG agent with natural language questions. The tool retrieves relevant documents and uses an LLM to generate comprehensive answers.

**Parameters:**
- `question` (str): The question to ask the RAG agent
- `top_k` (int, optional): Number of documents to retrieve (1-10, default: 3)
- `include_citations` (bool, optional): Whether to include source citations (default: true)

**Returns:**
- `answer`: The generated answer
- `citations`: List of source citations (if requested)
- `processing_time_ms`: Processing time in milliseconds
- `model`: LLM model used

### 2. `find_files` - Semantic File Search
Find files using semantic search based on vector embeddings.

**Parameters:**
- `query` (str): The search query to find relevant files
- `num_results` (int, optional): Number of results to return (1-100, default: 10)
- `offset` (int, optional): Offset for pagination (default: 0)

**Returns:**
- `files`: List of matching files with metadata
  - `filename`: Name of the file
  - `source`: Full path to the file
  - `score`: Relevance score
  - `text_preview`: Preview of the content
- `total`: Total number of results
- `offset`: Offset used for this request

### 3. `get_file` - Retrieve File Content
Get the full content of a file by name or path.

**Parameters:**
- `filename` (str): The name or path of the file to retrieve

**Returns:**
- `filename`: Name of the file
- `content`: Full file content
- `metadata`: File metadata (size, timestamps, path)

## Installation

```bash
cd packages/fetchcraft-demo-mcp-server
pip install -e .
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

### Required Configuration

- **Qdrant**: Vector database connection
  - `QDRANT_HOST`: Qdrant server host (default: localhost)
  - `QDRANT_PORT`: Qdrant server port (default: 6333)
  - `COLLECTION_NAME`: Collection name (default: fetchcraft_mcp)

- **Documents**: Path to documents to index
  - `DOCUMENTS_PATH`: Path to documents directory

- **Embeddings**: Embedding model configuration
  - `EMBEDDING_MODEL`: Model name (e.g., bge-m3)
  - `OPENAI_API_KEY`: API key for embeddings
  - `OPENAI_BASE_URL`: Base URL for embeddings API

- **LLM**: Language model for RAG
  - `LLM_MODEL`: Model name (e.g., gpt-4-turbo)

### Optional Configuration

- **Chunking**: Document chunking settings
  - `CHUNK_SIZE`: Size of parent chunks (default: 8192)
  - `CHILD_CHUNKS`: Sizes of child chunks (default: 4096,1024)
  - `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

- **Search**: Search configuration
  - `ENABLE_HYBRID`: Enable hybrid search (default: true)
  - `FUSION_METHOD`: Fusion method for hybrid search (default: rrf)

## Usage

### Running the Server

```bash
fetchcraft-demo-mcp-server
```

Or directly with Python:

```bash
python -m fetchcraft.mcp.server
```

The server will start on `http://localhost:8765` by default.

### Web Interface

A modern web interface is available for semantic file search:

1. **Build the frontend** (first time only):
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Start the server**:
   ```bash
   fetchcraft-demo-mcp-server
   ```

3. **Open your browser** to `http://localhost:8765`

The web interface provides:
- Semantic file search with natural language queries
- Pagination for large result sets
- Score-based relevance ranking
- Clean, modern UI built with React and TailwindCSS

See `frontend/README.md` for detailed frontend documentation.

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "fetchcraft": {
      "command": "fetchcraft-demo-mcp-server",
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "DOCUMENTS_PATH": "/path/to/your/documents",
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "LLM_MODEL": "gpt-4-turbo"
      }
    }
  }
}
```

### Using with Other MCP Clients

The server uses the standard MCP protocol over stdio, so it can be integrated with any MCP-compatible client.

## Requirements

- Python 3.12+
- Qdrant (running instance)
- OpenAI-compatible API (for embeddings and LLM)

## Example Workflow

1. **Start Qdrant** (if not already running):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Configure the server** with your documents path and API keys

3. **Run the server**:
   ```bash
   fetchcraft-demo-mcp-server
   ```

4. **First run**: The server will automatically:
   - Load documents from `DOCUMENTS_PATH`
   - Parse them into chunks
   - Generate embeddings
   - Index them in Qdrant

5. **Subsequent runs**: The server will use the existing index

## Architecture

The server is built on:
- **MCP (Model Context Protocol)**: Standard protocol for AI tool integration
- **FastAPI**: REST API for web interface
- **Fetchcraft Core**: Document processing, indexing, and retrieval
- **Qdrant**: Vector database for semantic search
- **Pydantic AI**: Agent framework for RAG
- **React + TypeScript**: Modern web interface (optional)

## License

MIT
