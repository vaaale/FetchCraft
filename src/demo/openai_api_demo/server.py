"""
FastAPI OpenAI-Compatible RAG API Server

This demo provides an OpenAI-compatible API for hybrid RAG with streaming support:
- Compatible with OpenAI's chat completions API format
- Supports streaming responses
- Hybrid search (dense + sparse vectors) for better results
- Returns citations in the response
- Can be used with any OpenAI client

Usage:
    python -m demo.openai_api_demo.server
    
    # Or with custom settings:
    ENABLE_HYBRID=true FUSION_METHOD=rrf python -m demo.openai_api_demo.server
    
    # Test with curl:
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "rag-hybrid",
        "messages": [{"role": "user", "content": "What is this about?"}],
        "stream": false
      }'
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from pydantic_ai import Tool

from fetchcraft.agents import RetrieverTool, PydanticAgent, OpenWebUIFormatter
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import SymNode
from fetchcraft.source import FilesystemDocumentSource
from fetchcraft.node_parser import HierarchicalNodeParser, SimpleNodeParser
from fetchcraft.vector_store import QdrantVectorStore


# ============================================================================
# Configuration
# ============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_chatbot")
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# Embeddings configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
INDEX_ID = "docs-index"

# LLM configuration for the agent
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_SIZES = [4096, 1024]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
USE_HIERARCHICAL_CHUNKING = os.getenv("USE_HIERARCHICAL_CHUNKING", "true").lower() == "true"

# Hybrid search configuration
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))


# ============================================================================
# OpenAI-Compatible Models
# ============================================================================

class Message(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    citations: Optional[List[Dict[str, Any]]] = None


class DeltaMessage(BaseModel):
    """Delta message for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """Streaming choice."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    agent: Optional[PydanticAgent] = None
    vector_index: Optional[VectorIndex] = None
    initialized: bool = False


app_state = AppState()


# ============================================================================
# RAG System Setup
# ============================================================================

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    return collection_name in collection_names


async def load_and_index_documents(
    vector_index: VectorIndex,
    documents_path: Path,
    chunk_size: int = 8192,
    child_sizes = [4096, 1024],
    overlap: int = 200,
    use_hierarchical: bool = True
) -> int:
    """Load documents from a directory and index them."""
    print(f"\n📂 Loading documents from: {documents_path}")
    
    if not documents_path.exists():
        print(f"⚠️  Documents path does not exist: {documents_path}")
        return 0
    
    # Load documents from filesystem
    print(f"   Loading documents...")
    source = FilesystemDocumentSource.from_directory(
        directory=documents_path,
        pattern="*",
        recursive=True
    )
    
    documents = []
    async for doc in source.get_documents():
        documents.append(doc)
    
    if not documents:
        print("⚠️  No text files found in the specified directory!")
        return 0
    
    print(f"  ✓ Loaded {len(documents)} documents")
    
    # Parse documents into nodes
    if use_hierarchical:
        print(f"   Using HierarchicalNodeParser")
        parser = HierarchicalNodeParser(
            chunk_size=chunk_size,
            overlap=overlap,
            child_sizes=child_sizes,
            child_overlap=50
        )
    else:
        print(f"   Using SimpleNodeParser ({chunk_size} chars)")
        parser = SimpleNodeParser(
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    all_nodes = parser.get_nodes(documents)
    
    # For hierarchical, index only the SymNodes (children)
    if use_hierarchical:
        all_chunks = [n for n in all_nodes if isinstance(n, SymNode)]
        print(f"  ✓ Created {len(all_nodes)} total nodes ({len(all_chunks)} SymNodes for indexing)")
    else:
        all_chunks = all_nodes
        print(f"  ✓ Created {len(all_chunks)} chunks")
    
    print(f"\n🔄 Indexing {len(all_chunks)} chunks with HYBRID SEARCH...")
    
    # Index all chunks
    await vector_index.add_nodes(all_chunks, show_progress=True)
    
    print(f"✅ Successfully indexed {len(all_chunks)} chunks with hybrid search!")
    return len(all_chunks)


async def setup_rag_system():
    """Set up the RAG system with hybrid search."""
    print("="*70)
    print("🚀 FastAPI RAG Server - HYBRID SEARCH MODE")
    print("="*70)
    print("\n💡 Hybrid Search = Dense (semantic) + Sparse (keyword) vectors")
    print("="*70)
    
    # Initialize embeddings
    print("\n1️⃣  Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )
    dimension = embeddings.dimension
    print(f"   ✓ Dense embeddings initialized: {EMBEDDING_MODEL} (dimension: {dimension})")
    
    # Connect to Qdrant
    print(f"\n2️⃣  Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()
    print(f"   ✓ Connected to Qdrant")
    
    # Check if collection exists
    print(f"\n3️⃣  Checking collection '{COLLECTION_NAME}'...")
    needs_indexing = not collection_exists(client, COLLECTION_NAME)
    
    if needs_indexing:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' does not exist - will create and index")
    else:
        print(f"   ✓ Collection '{COLLECTION_NAME}' already exists")
    
    # Create vector store with hybrid search
    print(f"\n🔥 Creating vector store with HYBRID SEARCH...")
    print(f"   • Enable Hybrid: {ENABLE_HYBRID}")
    print(f"   • Fusion Method: {FUSION_METHOD.upper()}")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=ENABLE_HYBRID,
        fusion_method=FUSION_METHOD
    )
    print(f"   ✓ Vector store created with hybrid search enabled!")
    
    # Create vector index
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )
    
    # Index documents if needed
    if needs_indexing:
        print(f"\n4️⃣  Indexing documents with hybrid search...")
        await load_and_index_documents(
            vector_index=vector_index,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_SIZES,
            overlap=CHUNK_OVERLAP,
            use_hierarchical=USE_HIERARCHICAL_CHUNKING
        )
    else:
        print(f"\n4️⃣  Skipping document indexing (collection already exists)")
    
    # Create retriever
    print(f"\n5️⃣  Creating hybrid retriever...")
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    print(f"   ✓ Retriever created (top_k=3, hybrid search enabled)")
    
    # Create agent
    print(f"\n6️⃣  Creating ReAct agent...")
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]
    
    agent = PydanticAgent.create(
        model=LLM_MODEL,
        tools=tools,
        retries=3,
        output_formatter=OpenWebUIFormatter()
    )
    print(f"   ✓ Agent created with hybrid search retrieval")
    
    print("\n" + "="*70)
    print("✅ HYBRID SEARCH RAG System Ready!")
    print("="*70)
    
    return agent, vector_index


# ============================================================================
# FastAPI Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG system on startup."""
    try:
        print("\n🚀 Starting FastAPI RAG Server...")
        agent, vector_index = await setup_rag_system()
        app_state.agent = agent
        app_state.vector_index = vector_index
        app_state.initialized = True
        print(f"\n✅ Server ready at http://{HOST}:{PORT}")
        print(f"   OpenAI-compatible endpoint: http://{HOST}:{PORT}/v1/chat/completions")
        print("="*70 + "\n")
        yield
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("\n👋 Shutting down server...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RAG API with Hybrid Search",
    description="OpenAI-compatible API for RAG with hybrid search (dense + sparse vectors)",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API with Hybrid Search",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "models": "/v1/models"
        },
        "features": {
            "hybrid_search": ENABLE_HYBRID,
            "fusion_method": FUSION_METHOD,
            "streaming": True
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if app_state.initialized else "initializing",
        "hybrid_search_enabled": ENABLE_HYBRID,
        "fusion_method": FUSION_METHOD
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-hybrid",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fetchcraft",
                "permission": [],
                "root": "rag-hybrid",
                "parent": None
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with RAG.
    
    Supports both streaming and non-streaming responses.
    """
    if not app_state.initialized or not app_state.agent:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    
    question = user_messages[-1].content
    
    # Prepare conversation history
    memory = []
    for msg in request.messages[:-1]:  # Exclude the last message
        memory.append(msg.model_dump())
    
    if request.stream:
        return StreamingResponse(
            stream_response(request, question, memory),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response(request, question, memory)


async def non_stream_response(
    request: ChatCompletionRequest,
    question: str,
    memory: List[Dict]
) -> ChatCompletionResponse:
    """Generate a non-streaming response."""
    # Query the agent
    response = await app_state.agent.query(question, messages=memory)
    
    # Format response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    answer = response.response.content
    
    # Extract citations
    citations = []
    if response.citations:
        for citation in response.citations:
            citations.append({
                "source": citation.node.metadata.get("source", "Unknown"),
                "filename": citation.node.metadata.get("filename", "N/A"),
                "score": float(citation.node.score) if hasattr(citation.node, "score") else 0.0,
                "text_preview": citation.node.text[:200] + "..." if len(citation.node.text) > 200 else citation.node.text
            })
    
    # Estimate token counts (rough approximation)
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(answer.split())
    
    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        ),
        citations=citations if citations else None
    )


async def stream_response(
    request: ChatCompletionRequest,
    question: str,
    memory: List[Dict]
) -> AsyncIterator[str]:
    """Generate a streaming response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    
    # Send role first
    chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )
        ]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Query the agent (we'll simulate streaming by chunking the response)
    response = await app_state.agent.query(question, messages=memory)
    answer = response.response.content
    # answer = f"""
    # ```html
    # <form>
    #   <label for="fname">First name:</label><br>
    #   <input type="text" id="fname" name="fname"><br>
    #   <label for="lname">Last name:</label><br>
    #   <input type="text" id="lname" name="lname">
    # </form>
    # ```
    # """
    
    # Stream the response in chunks
    chunk_size = 400  # words per chunk
    words = answer.split(" ")
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        if i + chunk_size < len(words):
            chunk_text += " "
        
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=chunk_text),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.05)  # Small delay for realistic streaming
    
    # Send finish
    chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting FastAPI RAG Server with Hybrid Search")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print(f"  • Hybrid Search: {ENABLE_HYBRID}")
    print(f"  • Fusion Method: {FUSION_METHOD}")
    print(f"  • Documents: {DOCUMENTS_PATH}")
    print("="*70 + "\n")
    
    uvicorn.run(
        "demo.openai_api_demo.server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
