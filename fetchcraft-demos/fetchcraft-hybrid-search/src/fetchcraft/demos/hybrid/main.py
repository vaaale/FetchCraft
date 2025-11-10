"""
Hybrid Search Demo of the RAG Framework

This demo showcases HYBRID SEARCH combining dense + sparse vectors for superior results:
1. Loading documents from a directory
2. Indexing them in Qdrant with BOTH dense (semantic) and sparse (keyword) vectors
3. Creating a ReAct agent with hybrid retrieval capabilities
4. Interactive REPL demonstrating improved keyword matching

Key Features:
- 🔍 Hybrid Search: Dense (semantic) + Sparse (BM25-style keyword) vectors
- 🎯 Better Results: Especially for technical content and specific terminology
- ⚡ RRF Fusion: Reciprocal Rank Fusion combines both search types
- 📊 Side-by-side: Compare hybrid vs dense-only search results

Usage:
    python -m demo.hybrid_demo.run_demo
"""

import asyncio
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from pydantic_ai import Tool

from fetchcraft.agents import RetrieverTool, PydanticAgent
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import SymNode
from fetchcraft.parsing.filesystem import FilesystemDocumentParser
from fetchcraft.node_parser import HierarchicalNodeParser, SimpleNodeParser
from fetchcraft.vector_store import QdrantVectorStore

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fetchcraft_hybrid"  # Different collection for hybrid search
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# Embeddings configuration (adjust based on your setup)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # None = use OpenAI default
INDEX_ID = "hybrid-demo-index-001"

# LLM configuration for the agent
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8192"))
CHILD_SIZES = [4096, 1024]
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
USE_HIERARCHICAL_CHUNKING = os.getenv("USE_HIERARCHICAL_CHUNKING", "true").lower() == "true"

# 🔥 HYBRID SEARCH CONFIGURATION
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")  # "rrf" or "dbsf"


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
    chunk_size: int = 8192,
    child_sizes = [4096, 1024],
    overlap: int = 200,
    use_hierarchical: bool = True
) -> int:
    """
    Load documents from a directory and index them.
    
    Args:
        vector_index: VectorIndex instance to add documents to
        documents_path: Path to the directory containing text files
        chunk_size: Size of text chunks
        child_sizes: Sizes for hierarchical child chunks
        overlap: Overlap between chunks
        use_hierarchical: Whether to use hierarchical chunking (default: True)
        
    Returns:
        Number of chunks indexed
    """
    print(f"\n📂 Loading documents from: {documents_path}")
    
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")
    
    # Step 1: Load documents from filesystem
    print(f"   Loading documents...")
    source = FilesystemDocumentParser.from_directory(
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
    
    # Step 2: Parse documents into nodes
    if use_hierarchical:
        print(f"   Using HierarchicalNodeParser")
        print(f"     - Parent chunks: {chunk_size} chars")
        print(f"     - Child chunks: {child_sizes}")
        print(f"     - Recursive splitting: paragraph → line → sentence → word")
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
    # For simple, index all chunks
    if use_hierarchical:
        all_chunks = [n for n in all_nodes if isinstance(n, SymNode)]
        print(f"  ✓ Created {len(all_nodes)} total nodes ({len(all_chunks)} SymNodes for indexing)")
    else:
        all_chunks = all_nodes
        print(f"  ✓ Created {len(all_chunks)} chunks")
    
    print(f"\n🔄 Indexing {len(all_chunks)} chunks with HYBRID SEARCH...")
    print(f"   Each chunk will have:")
    print(f"     🎯 Dense vector (semantic understanding)")
    print(f"     🔍 Sparse vector (keyword matching)")
    
    # Index all chunks (embeddings will be generated automatically)
    await vector_index.add_nodes(all_chunks, show_progress=True)
    
    print(f"✅ Successfully indexed {len(all_chunks)} chunks with hybrid search!")
    return len(all_chunks)


async def setup_rag_system():
    """
    Set up the RAG system with HYBRID SEARCH: vector store, index, and agent.
    
    Returns:
        Tuple of (agent, vector_index)
    """
    print("="*70)
    print("🚀 RAG Framework Demo - HYBRID SEARCH MODE")
    print("="*70)
    print("\n💡 Hybrid Search = Dense (semantic) + Sparse (keyword) vectors")
    print("   Benefits: Better results for technical terms and specific keywords")
    print("="*70)
    
    # Initialize embeddings
    print("\n1️⃣  Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )


    # Connect to Qdrant
    print(f"\n2️⃣  Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()  # Test connection
    print(f"   ✓ Connected to Qdrant")
    
    # Check if collection exists
    print(f"\n3️⃣  Checking collection '{COLLECTION_NAME}'...")
    needs_indexing = not collection_exists(client, COLLECTION_NAME)
    
    if needs_indexing:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' does not exist - will create and index")
    else:
        print(f"   ✓ Collection '{COLLECTION_NAME}' already exists - skipping indexing")
    
    # Create vector store with HYBRID SEARCH enabled
    print(f"\n🔥 Creating vector store with HYBRID SEARCH...")
    print(f"   • Enable Hybrid: {ENABLE_HYBRID}")
    print(f"   • Fusion Method: {FUSION_METHOD.upper()}")
    
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            distance="Cosine",
            enable_hybrid=ENABLE_HYBRID,      # 🔥 Enable hybrid search
            fusion_method=FUSION_METHOD        # Choose RRF or DBSF
        )
        print(f"   ✓ Vector store created with hybrid search enabled!")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Hybrid search requires fastembed:")
        print("   pip install fastembed")
        sys.exit(1)
    
    # Create vector index with a consistent index_id
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )
    
    # Index documents if needed
    if needs_indexing:
        print(f"\n4️⃣  Indexing documents with hybrid search...")
        num_chunks = await load_and_index_documents(
            vector_index=vector_index,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_SIZES,
            overlap=CHUNK_OVERLAP,
            use_hierarchical=USE_HIERARCHICAL_CHUNKING
        )
        if num_chunks == 0:
            print("\n⚠️  Warning: No documents were indexed!")
    else:
        print(f"\n4️⃣  Skipping document indexing (collection already exists)")
    
    # Create retriever
    print(f"\n5️⃣  Creating hybrid retriever...")
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    print(f"   ✓ Retriever created (top_k=3, hybrid search enabled)")
    
    # Create retriever tool and agent
    print(f"\n6️⃣  Creating ReAct agent...")
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]
    
    agent = PydanticAgent.create(
        model=LLM_MODEL,
        tools=tools,
        retries=3
    )
    print(f"   ✓ Agent created with hybrid search retrieval")
    
    print("\n" + "="*70)
    print("✅ HYBRID SEARCH RAG System Ready!")
    print("="*70)
    print("\n💡 Try queries with specific terms, model numbers, or technical jargon")
    print("   to see how hybrid search improves keyword matching!")
    print("="*70)
    
    return agent, vector_index


async def repl_loop(agent: PydanticAgent):
    """
    Run an interactive REPL loop for the agent.
    
    Args:
        agent: The ReAct agent to use for answering questions
    """
    print("\n" + "🤖 Interactive Q&A Session (HYBRID SEARCH)".center(70))
    print("="*70)
    print("\nAsk questions about the indexed documents.")
    print("Hybrid search combines semantic understanding + keyword matching.")
    print("\nType 'quit', 'exit', or press Ctrl+C to exit.\n")
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
        print("\n🔍 Searching with hybrid search (dense + sparse vectors)...\n")
        response = await agent.query(question, messages=memory)
        memory.append(response.query)
        memory.append(response.response)

        print("─" * 70)
        print(f"💬 Answer:\n{response.response.content}\n")
        
        # Show citations if available
        if response.citations:
            print("─" * 70)
            print("📚 Citations (ranked by hybrid search):")
            for i, citation in enumerate(response.citations, 1):
                source = citation.node.metadata.get('parsing', 'Unknown')
                filename = citation.node.metadata.get('filename', Path(source).name if source != 'Unknown' else 'N/A')
                print(f"   [{i}] {filename} (score: {citation.node.score:.3f})")
        
        print("─" * 70)


def print_error_hints(error: Exception):
    """Print helpful hints based on the error type."""
    error_msg = str(error).lower()
    
    if "fastembed" in error_msg:
        print("\n💡 FastEmbed Missing:")
        print("   - Hybrid search requires fastembed")
        print("   - Install with: pip install fastembed")
    elif "api key" in error_msg or "authentication" in error_msg:
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
        print("   - Install required packages: pip install pydantic-ai qdrant-client openai fastembed")
    else:
        print("\n💡 For more help, check the README.md file")


async def main():
    """Main entry point for the hybrid search demo."""
    try:
        # Set up the RAG system with hybrid search
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
