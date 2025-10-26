"""
Simple Demo of the RAG Framework

This demo showcases:
1. Loading documents from a directory
2. Indexing them in Qdrant vector store
3. Creating a ReAct agent with retrieval capabilities
4. Interactive REPL for asking questions

Usage:
    python -m demo.simple_demo.run_demo
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Any

from qdrant_client import QdrantClient
from pydantic_ai import Tool

from fetchcraft import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    TextFileDocumentParser,
    Chunk,
    ReActAgent,
    RetrieverTool
)


# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fetchcraft"
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# Embeddings configuration (adjust based on your setup)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # None = use OpenAI default
INDEX_ID = "39372e06-5cb9-45ef-ab40-cc66649f7362"

# LLM configuration for the agent
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to check
        
    Returns:
        True if collection exists, False otherwise
    """
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    return collection_name in collection_names


async def load_and_index_documents(
    vector_index: VectorIndex,
    documents_path: Path,
    chunk_size: int = 500,
    overlap: int = 50
) -> int:
    """
    Load documents from a directory and index them.
    
    Args:
        vector_index: VectorIndex instance to add documents to
        documents_path: Path to the directory containing text files
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        
    Returns:
        Number of chunks indexed
    """
    print(f"\n📂 Loading documents from: {documents_path}")
    
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")
    
    # Parse all text files in the directory
    parser = TextFileDocumentParser(
        chunk_size=chunk_size,
        overlap=overlap,
        separator=" "
    )
    
    results = parser.parse_directory(
        directory_path=documents_path,
        pattern="*",
        recursive=True
    )
    
    if not results:
        print("⚠️  No text files found in the specified directory!")
        return 0
    
    # Flatten all chunks from all files
    all_chunks: List[Chunk] = []
    for file_path, chunks in results.items():
        print(f"  ✓ Loaded {len(chunks)} chunks from {Path(file_path).name}")
        all_chunks.extend(chunks)
    
    print(f"\n🔄 Indexing {len(all_chunks)} chunks (this may take a moment)...")
    
    # Index all chunks (embeddings will be generated automatically)
    await vector_index.add_documents(all_chunks, show_progress=True)
    
    print(f"✅ Successfully indexed {len(all_chunks)} chunks!")
    return len(all_chunks)


async def setup_rag_system():
    """
    Set up the RAG system: vector store, index, and agent.
    
    Returns:
        Tuple of (agent, vector_index)
    """
    print("="*70)
    print("🚀 RAG Framework Demo - Initializing System")
    print("="*70)
    
    # Initialize embeddings
    print("\n1️⃣  Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )
    dimension = embeddings.dimension
    print(f"   ✓ Embeddings initialized: {EMBEDDING_MODEL} (dimension: {dimension})")
    
    # Connect to Qdrant
    print(f"\n2️⃣  Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()  # Test connection
    print(f"   ✓ Connected to Qdrant")
    
    # Check if collection exists
    print(f"\n3️⃣  Checking collection '{COLLECTION_NAME}'...")
    needs_indexing = not collection_exists(client, COLLECTION_NAME)
    
    if needs_indexing:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' does not exist - will create and index documents")
    else:
        print(f"   ✓ Collection '{COLLECTION_NAME}' already exists - skipping indexing")
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_size=dimension,
        distance="Cosine"
    )
    
    # Create vector index with a consistent index_id
    # This ensures documents can be found across multiple runs
    vector_index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings,
        index_id=INDEX_ID
    )
    
    # Index documents if needed
    if needs_indexing:
        print(f"\n4️⃣  Indexing documents...")
        num_chunks = await load_and_index_documents(
            vector_index=vector_index,
            documents_path=DOCUMENTS_PATH,
            chunk_size=4096,
            overlap=200
        )
        if num_chunks == 0:
            print("\n⚠️  Warning: No documents were indexed!")
    else:
        print(f"\n4️⃣  Skipping document indexing (collection already exists)")
    
    # Create retriever
    print(f"\n5️⃣  Creating retriever...")
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    print(f"   ✓ Retriever created (top_k=3)")
    
    # Create retriever tool and agent
    print(f"\n6️⃣  Creating ReAct agent...")
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]
    
    agent = ReActAgent.create(
        model=LLM_MODEL,
        tools=tools,
        retries=3
    )
    print(f"   ✓ Agent created: {agent}")
    
    print("\n" + "="*70)
    print("✅ RAG System Ready!")
    print("="*70)
    
    return agent, vector_index


async def repl_loop(agent: ReActAgent):
    """
    Run an interactive REPL loop for the agent.
    
    Args:
        agent: The ReAct agent to use for answering questions
    """
    print("\n" + "🤖 Interactive Q&A Session".center(70))
    print("="*70)
    print("\nAsk questions about the indexed documents.")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.\n")
    print("="*70 + "\n")

    memory = []
    while True:
        # Get user input
        question = input("\n❓ Your Question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty input
        if not question:
            continue
        
        # Query the agent
        print("\n🔍 Searching and reasoning...\n")
        response = await agent.query(question, messages=memory)
        memory.append(response.query)
        memory.append(response.response)

        print("─" * 70)
        print(f"💬 Answer:\n{response.response.content}\n")
        
        # Show citations if available
        if response.citations:
            print("─" * 70)
            print("📚 Citations:")
            for i, citation in enumerate(response.citations, 1):
                source = citation.node.metadata.get('source', 'Unknown')
                filename = citation.node.metadata.get('filename', Path(source).name if source != 'Unknown' else 'N/A')
                print(f"   [{i}] {filename} (score: {citation.node.score:.3f})")
        
        print("─" * 70)


def print_error_hints(error: Exception):
    """Print helpful hints based on the error type."""
    error_msg = str(error).lower()
    
    if "api key" in error_msg or "authentication" in error_msg:
        print("\n💡 API Key Issue:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Or configure EMBEDDING_BASE_URL for a custom endpoint")
    elif "connection" in error_msg or "refused" in error_msg or "qdrant" in error_msg:
        print("\n💡 Connection Issue:")
        print("   - Make sure Qdrant is running on localhost:6333")
        print("   - Start with: docker run -p 6333:6333 qdrant/qdrant")
    elif "not found" in error_msg or "no such file" in error_msg:
        print("\n💡 File Path Issue:")
        print(f"   - Check that {DOCUMENTS_PATH} exists")
        print("   - Make sure it contains .txt files")
    elif "pydantic" in error_msg or "import" in error_msg:
        print("\n💡 Dependency Issue:")
        print("   - Install required packages: pip install pydantic-ai qdrant-client openai")
    else:
        print("\n💡 For more help, check the README.md file")


async def main():
    """Main entry point for the demo."""
    try:
        # Set up the RAG system
        agent, vector_index = await setup_rag_system()
        
        # Run the interactive REPL
        await repl_loop(agent)
        
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print_error_hints(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
