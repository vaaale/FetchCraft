"""
Fetchcraft MCP Server

This MCP server provides tools for querying documents using RAG and finding files
using semantic search.

Features:
- Query tool for RAG-based question answering
- Find files tool using semantic search
- Get file tool to retrieve full file contents

Usage:
    python -m fetchcraft.mcp.server
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from pydantic_ai import Tool
from qdrant_client import QdrantClient

from fetchcraft.agents import RetrieverTool, PydanticAgent
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import SymNode
from fetchcraft.node_parser import HierarchicalNodeParser
from fetchcraft.parsing.filesystem import FilesystemDocumentParser
from fetchcraft.vector_store import QdrantVectorStore

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_mcp")
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# Embeddings configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
INDEX_ID = os.getenv("INDEX_ID", "docs-index")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_CHUNKS = [int(chunk_size) for chunk_size in os.getenv("CHILD_CHUNKS", "4096,1024").split(",")]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Hybrid search configuration
ENABLE_HYBRID = bool(os.getenv("ENABLE_HYBRID", True))
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    agent: Optional[PydanticAgent] = None
    vector_index: Optional[VectorIndex] = None
    doc_store: Optional[MongoDBDocumentStore] = None
    retriever = None
    documents_path: Path = DOCUMENTS_PATH
    initialized: bool = False


app_state = AppState()


# ============================================================================
# Helper Functions
# ============================================================================

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    return collection_name in collection_names


async def load_and_index_documents(
    vector_index: VectorIndex,
    doc_store: MongoDBDocumentStore,
    documents_path: Path,
    chunk_size: int = 8192,
    child_sizes: List[int] = None,
    overlap: int = 200,
) -> int:
    """Load documents from a directory and index them."""
    print(f"\n📂 Loading documents from: {documents_path}")

    if not documents_path.exists():
        print(f"⚠️  Documents path does not exist: {documents_path}")
        return 0

    if child_sizes is None:
        child_sizes = [4096, 1024]

    # Load documents from filesystem
    source = FilesystemDocumentParser.from_directory(
        directory=documents_path,
        pattern="*",
        recursive=True
    )

    documents = []
    async for doc in source.get_documents():
        await doc_store.add_document(doc)
        documents.append(doc)

    if not documents:
        print("⚠️  No files found in the specified directory!")
        return 0

    print(f"  ✓ Loaded {len(documents)} documents")

    # Parse documents into nodes
    parser = HierarchicalNodeParser(
        chunk_size=chunk_size,
        overlap=overlap,
        child_sizes=child_sizes,
        child_overlap=50
    )

    all_nodes = parser.get_nodes(documents)
    all_chunks = [n for n in all_nodes if isinstance(n, SymNode)]
    print(f"  ✓ Created {len(all_chunks)} chunks for indexing")

    # Index all chunks
    await vector_index.add_nodes(all_chunks, show_progress=True)

    print(f"✅ Successfully indexed {len(all_chunks)} chunks!")
    return len(all_chunks)


async def setup_rag_system():
    """Set up the RAG system."""
    print("=" * 70)
    print("🚀 Fetchcraft MCP Server - Initializing")
    print("=" * 70)

    # Initialize embeddings
    print("\n1️⃣  Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )
    print(f"   ✓ Embeddings initialized: {EMBEDDING_MODEL}")

    # Connect to Qdrant
    print(f"\n2️⃣  Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()
    print(f"   ✓ Connected to Qdrant")

    # Check if collection exists
    print(f"\n3️⃣  Checking collection '{COLLECTION_NAME}'...")
    needs_indexing = not collection_exists(client, COLLECTION_NAME)

    if needs_indexing:
        print(f"   ⚠️  Collection does not exist - will create and index")
    else:
        print(f"   ✓ Collection already exists")

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=ENABLE_HYBRID,
        fusion_method=FUSION_METHOD
    )
    print(f"   ✓ Vector store created")

    doc_store = MongoDBDocumentStore(
        database_name="fetchcraft",
        collection_name=COLLECTION_NAME,
    )

    # Create vector index
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )

    # Index documents if needed
    if needs_indexing:
        print(f"\n4️⃣  Indexing documents...")
        await load_and_index_documents(
            vector_index=vector_index,
            doc_store=doc_store,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_CHUNKS,
            overlap=CHUNK_OVERLAP
        )
    else:
        print(f"\n4️⃣  Skipping document indexing (collection already exists)")

    # Create retriever
    print(f"\n5️⃣  Creating retriever...")
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    print(f"   ✓ Retriever created")

    # Create agent
    print(f"\n6️⃣  Creating RAG agent...")
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

    agent = PydanticAgent.create(
        model=LLM_MODEL,
        tools=tools,
        retries=3
    )
    print(f"   ✓ Agent created with model: {LLM_MODEL}")

    print("\n" + "=" * 70)
    print("✅ RAG System Ready!")
    print("=" * 70)

    return agent, vector_index, doc_store, retriever


# ============================================================================
# MCP Server Setup
# ============================================================================

@asynccontextmanager
async def app_lifespan(mcp_app: FastMCP):
    """Initialize the RAG system on startup."""
    try:
        print("\n⚙️  Configuring RAG system...")
        agent, vector_index, doc_store, retriever = await setup_rag_system()
        app_state.agent = agent
        app_state.vector_index = vector_index
        app_state.doc_store = doc_store
        app_state.retriever = retriever
        app_state.initialized = True
        print(f"\n✅ MCP Server ready!")
        print("=" * 70 + "\n")
        yield
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("\n👋 Shutting down MCP server...")


# Create MCP server
mcp = FastMCP(
    name="Fetchcraft MCP Server",
    instructions="This server provides tools for document search and retrieval using RAG.",
    lifespan=app_lifespan,
    host=HOST,
    port=PORT,
)


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def query(
    question: str,
    top_k: int = 3,
    include_citations: bool = True
) -> dict:
    """
    Query the RAG agent with a question.

    This tool retrieves relevant documents from the vector database and uses
    an LLM to generate a comprehensive answer based on the retrieved context.

    Args:
        question: The question to ask the RAG agent
        top_k: Number of documents to retrieve (1-10)
        include_citations: Whether to include parsing citations in the response

    Returns:
        Dictionary containing the answer, citations, processing time, and model used
    """
    if not app_state.initialized or not app_state.agent:
        raise RuntimeError("RAG system not initialized. Please try again in a moment.")

    start_time = time.time()

    try:
        # Query the agent
        response = await app_state.agent.query(question)

        # Extract answer
        answer = response.response.content

        # Extract citations if requested
        citations = None
        if include_citations and response.citations:
            citations = []
            for citation in response.citations[:top_k]:
                citations.append({
                    "parsing": citation.node.metadata.get("parsing", "Unknown"),
                    "filename": citation.node.metadata.get("filename", "N/A"),
                    "score": float(citation.node.score) if hasattr(citation.node, "score") else 0.0,
                    "text_preview": (
                        citation.node.text[:200] + "..."
                        if len(citation.node.text) > 200
                        else citation.node.text
                    )
                })

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "answer": answer,
            "citations": citations,
            "processing_time_ms": round(processing_time, 2),
            "model": LLM_MODEL
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error processing query: {str(e)}")


@mcp.tool()
async def find_files(
    query: str,
    num_results: int = 10,
    offset: int = 0
) -> dict:
    """
    Find files using semantic search.

    This tool searches for files based on semantic similarity to the query.
    It uses vector embeddings to find relevant files.

    Args:
        query: The search query to find relevant files
        num_results: Number of results to return (1-100)
        offset: Offset for pagination (starting from 0)

    Returns:
        Dictionary containing the list of matching files, total count, and offset
    """
    if not app_state.initialized or not app_state.retriever:
        raise RuntimeError("RAG system not initialized. Please try again in a moment.")

    try:
        # Retrieve nodes using the retriever
        # We request num_results + offset to handle pagination
        total_needed = num_results + offset
        nodes = app_state.retriever.retrieve(query, top_k=total_needed)

        # Apply offset and limit
        paginated_nodes = nodes[offset:offset + num_results]

        # Convert nodes to file results
        files = []
        seen_sources = set()
        
        for node in paginated_nodes:
            # Get filename from metadata
            source = node.node.metadata.get("parsing", "Unknown")
            filename = node.node.metadata.get("filename", "N/A")
            
            # Skip duplicates (same parsing file)
            if source in seen_sources:
                continue
            seen_sources.add(source)

            files.append({
                "filename": filename,
                "parsing": source,
                "score": float(node.score) if node.score else 0.0,
                "text_preview": (
                    node.node.text[:200] + "..."
                    if len(node.node.text) > 200
                    else node.node.text
                )
            })

        return {
            "files": files,
            "total": len(nodes),
            "offset": offset
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error finding files: {str(e)}")


@mcp.tool()
async def get_file(filename: str) -> dict:
    """
    Get the full content of a file.

    This tool retrieves the complete content of a file by its name or path.

    Args:
        filename: The name or path of the file to retrieve

    Returns:
        Dictionary containing the filename, full content, and metadata
    """

    print(f"\n📂 Getting file: {filename}")

    if not app_state.initialized:
        raise RuntimeError("RAG system not initialized. Please try again in a moment.")

    try:
        nodes = await app_state.doc_store.list_documents(filters={"metadata.filename": f"{filename}"}, limit=10)
        if len(nodes) == 1:
            node = nodes[-1]
            file_path = Path(node.metadata["parsing"])
        else:
            # Resolve the file path
            file_path = Path(filename)

            # If it's not an absolute path, look in the documents directory
            if not file_path.is_absolute():
                file_path = app_state.documents_path / filename
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encoding
            content = file_path.read_text(encoding='latin-1')
        
        # Get file metadata
        stat = file_path.stat()
        metadata = {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "absolute_path": str(file_path.absolute())
        }

        return {
            "filename": file_path.name,
            "content": content,
            "metadata": metadata
        }

    except FileNotFoundError as e:
        raise RuntimeError(str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error reading file: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    print("\n" + "=" * 70)
    print("🚀 Starting Fetchcraft MCP Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Model: {LLM_MODEL}")
    print(f"  • Collection: {COLLECTION_NAME}")
    print(f"  • Documents: {DOCUMENTS_PATH}")
    print(f"  • Hybrid Search: {ENABLE_HYBRID}")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print("=" * 70 + "\n")

    # Run the MCP server using stdio transport
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
