"""
Gradio Chatbot Web App with Hybrid Search RAG

A modern web-based chatbot interface powered by hybrid search (dense + sparse vectors).

Features:
- 🎨 Modern Gradio UI with chat history
- 🔍 Hybrid Search: Dense (semantic) + Sparse (BM25-style keyword) vectors
- 📚 Citation tracking and display
- 💬 Multi-turn conversation support
- ⚡ Real-time responses
- 📊 Source document display

Usage:
    python -m demo.gradio_chatbot.app
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

from fetchcraft.agents import PydanticAgent, RetrieverTool
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.node import SymNode
from fetchcraft.vector_store import QdrantVectorStore

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
from qdrant_client import QdrantClient
from pydantic_ai import Tool

from fetchcraft.parsing.filesystem import FilesystemDocumentParser
from fetchcraft.node_parser import HierarchicalNodeParser, SimpleNodeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fetchcraft_chatbot")
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "Documents"))

# Embeddings configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "sk-321")
EMBEDDING_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
INDEX_ID = "docs-index"

# LLM configuration
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

# Global state
agent: Optional[PydanticAgent] = None
vector_index: Optional[VectorIndex] = None
conversation_memory: Dict[str, List[Any]] = {}


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
    logger.info(f"Loading documents from: {documents_path}")
    
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")
    
    # Step 1: Load documents from filesystem
    logger.info("Loading documents...")
    source = FilesystemDocumentParser.from_directory(
        directory=documents_path,
        pattern="*.txt",
        recursive=True
    )
    
    documents = []
    async for doc in source.get_documents():
        documents.append(doc)
    
    if not documents:
        logger.warning("No text files found in the specified directory!")
        return 0
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Parse documents into nodes
    if use_hierarchical:
        logger.info(f"Using HierarchicalNodeParser (parent={chunk_size}, children={child_sizes})")
        parser = HierarchicalNodeParser(
            chunk_size=chunk_size,
            overlap=overlap,
            child_sizes=child_sizes,
            child_overlap=50
        )
    else:
        logger.info(f"Using SimpleNodeParser (chunk_size={chunk_size})")
        parser = SimpleNodeParser(
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    all_nodes = parser.get_nodes(documents)
    
    # For hierarchical, index only the SymNodes (children)
    # For simple, index all chunks
    if use_hierarchical:
        all_chunks = [n for n in all_nodes if isinstance(n, SymNode)]
        logger.info(f"Created {len(all_nodes)} total nodes ({len(all_chunks)} SymNodes for indexing)")
    else:
        all_chunks = all_nodes
        logger.info(f"Created {len(all_chunks)} chunks")
    
    logger.info(f"Indexing {len(all_chunks)} chunks with hybrid search...")
    await vector_index.add_nodes(DocumentNode, all_chunks, show_progress=True)
    
    logger.info(f"Successfully indexed {len(all_chunks)} chunks!")
    return len(all_chunks)


async def setup_rag_system():
    """Set up the RAG system with hybrid search."""
    logger.info("Initializing RAG system with hybrid search...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )
    
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()  # Test connection
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Check if collection exists
    needs_indexing = not collection_exists(client, COLLECTION_NAME)
    
    # Create vector store with hybrid search
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            distance="Cosine",
            enable_hybrid=ENABLE_HYBRID,
            fusion_method=FUSION_METHOD
        )
        logger.info("Vector store created with hybrid search enabled")
    except ImportError as e:
        logger.error(f"Failed to create vector store: {e}")
        raise
    
    # Create vector index
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=INDEX_ID
    )
    
    # Index documents if needed
    if needs_indexing:
        logger.info("Indexing documents...")
        num_chunks = await load_and_index_documents(
            vector_index=vector_index,
            documents_path=DOCUMENTS_PATH,
            chunk_size=CHUNK_SIZE,
            child_sizes=CHILD_SIZES,
            overlap=CHUNK_OVERLAP,
            use_hierarchical=USE_HIERARCHICAL_CHUNKING
        )
        if num_chunks == 0:
            logger.warning("No documents were indexed!")
    else:
        logger.info("Collection already exists, skipping indexing")
    
    # Create retriever
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)
    
    # Create agent
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]
    
    agent = PydanticAgent.create(
        model=LLM_MODEL,
        tools=tools,
        retries=3
    )
    
    logger.info("RAG system ready!")
    return agent, vector_index


async def chat_async(
    message: str,
    history: List[Dict[str, str]],
    session_id: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a chat message asynchronously.
    
    Args:
        message: User's message
        history: Chat history (messages format with role/content)
        session_id: Session identifier
        
    Returns:
        Tuple of (response text, citations)
    """
    if agent is None:
        return "⚠️ System not initialized. Please wait...", []
    
    # Get or create conversation memory for this session
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    memory = conversation_memory[session_id]
    
    # Query the agent
    logger.info(f"Processing query: {message}")
    response = await agent.query(message, messages=memory)
    
    # Update memory
    memory.append(response.query)
    memory.append(response.response)
    conversation_memory[session_id] = memory
    
    # Extract response text
    response_text = response.response.content
    
    # Format citations
    citations = []
    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            source = citation.node.metadata.get('parsing', 'Unknown')
            filename = citation.node.metadata.get('filename', Path(source).name if source != 'Unknown' else 'N/A')
            score = citation.node.score
            text_preview = citation.node.text[:200] + "..." if len(citation.node.text) > 200 else citation.node.text
            
            citations.append({
                'id': i,
                'filename': filename,
                'score': score,
                'preview': text_preview,
                'parsing': source
            })
    
    return response_text, citations


def format_citations_inline(citations: List[Dict[str, Any]]) -> str:
    """Format citations as expandable accordions."""
    if not citations:
        return ""
    
    result = f"\n\n---\n**📚 Sources ({len(citations)} documents)**\n\n"
    
    # Create expandable accordion for each citation
    for citation in citations:
        score_emoji = "🟢" if citation['score'] > 0.8 else "🟡" if citation['score'] > 0.6 else "⚪"
        score = citation['score']
        
        result += f"""<details style="margin-bottom: 8px; padding: 8px; border-left: 3px solid {'#27ae60' if score > 0.8 else '#f39c12' if score > 0.6 else '#95a5a6'}; background-color: #f8f9fa; border-radius: 4px;">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; user-select: none;">
{score_emoji} [{citation['id']}] {citation['filename']} <span style="color: #666; font-size: 0.9em;">(Score: {score:.3f})</span>
</summary>
<div style="margin-top: 10px; padding: 8px; background-color: white; border-radius: 4px; font-size: 0.95em; line-height: 1.5;">
<p style="margin: 0; color: #444;">{citation['preview']}</p>
<p style="margin-top: 8px; margin-bottom: 0; color: #999; font-size: 0.85em; font-style: italic;">📄 {citation['parsing']}</p>
</div>
</details>
"""
    return result


def chat_wrapper(message: str, history: List[Dict[str, str]], session_id: str):
    """
    Synchronous wrapper for async chat function.
    
    Args:
        message: User's message
        history: Chat history (messages format with role/content)
        session_id: Session identifier
        
    Returns:
        Updated history and citations HTML
    """
    # Run async function
    response_text, citations = asyncio.run(chat_async(message, history, session_id))
    
    # Add inline citations to response
    citations_inline = format_citations_inline(citations)
    response_with_citations = response_text + citations_inline
    
    # Update history with response in messages format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_with_citations})
    
    # Also format citations for side panel (keep for compatibility)
    citations_html = format_citations_html(citations)
    
    return history, citations_html


def format_citations_html(citations: List[Dict[str, Any]]) -> str:
    """Format citations as HTML."""
    if not citations:
        return "<div style='padding: 20px; text-align: center; color: #666;'>No citations available</div>"
    
    html = "<div style='padding: 15px;'>"
    html += "<h3 style='margin-top: 0; color: #2c3e50;'>📚 Sources</h3>"
    
    for citation in citations:
        score_color = "#27ae60" if citation['score'] > 0.8 else "#f39c12" if citation['score'] > 0.6 else "#95a5a6"
        
        html += f"""
        <div style='
            margin-bottom: 15px;
            padding: 12px;
            border-left: 4px solid {score_color};
            background-color: #f8f9fa;
            border-radius: 4px;
        '>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                <strong style='color: #2c3e50;'>[{citation['id']}] {citation['filename']}</strong>
                <span style='
                    background-color: {score_color};
                    color: white;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                '>
                    {citation['score']:.3f}
                </span>
            </div>
            <div style='color: #666; font-size: 13px; line-height: 1.4;'>
                {citation['preview']}
            </div>
            <div style='color: #999; font-size: 11px; margin-top: 5px;'>
                {citation['parsing']}
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def clear_conversation(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return [], "<div style='padding: 20px; text-align: center; color: #666;'>Conversation cleared</div>"


def create_gradio_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    #chatbot {
        height: 600px !important;
    }
    #citations {
        height: 600px !important;
        overflow-y: auto;
    }
    .message {
        padding: 10px !important;
    }
    /* Style for inline citations */
    details {
        transition: all 0.3s ease;
    }
    details summary {
        transition: all 0.2s ease;
    }
    details summary:hover {
        background-color: #e8eaed;
        border-radius: 4px;
    }
    details[open] summary {
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=custom_css,
        title="RAG Chatbot with Hybrid Search"
    ) as interface:
        
        # Session state
        session_id = gr.State(lambda: str(os.urandom(16).hex()))
        
        # Header
        gr.Markdown(
            """
            # 🤖 RAG Chatbot with Hybrid Search
            
            Ask questions about your indexed documents. This chatbot uses **hybrid search** 
            (dense + sparse vectors) for better keyword matching and semantic understanding.
            
            💡 **Tip:** Expand "📚 View Sources" in bot responses to see citation details!
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    elem_id="chatbot",
                    height=600,
                    show_copy_button=True,
                    sanitize_html=False,  # Allow HTML rendering in messages
                    type='messages'  # Use new messages format (openai-style)
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about your documents...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send 📤", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear 🗑️", scale=1)
                    
            with gr.Column(scale=1):
                # Citations panel
                citations_display = gr.HTML(
                    label="Citations",
                    elem_id="citations",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>Citations will appear here</div>"
                )
        
        # Info section
        with gr.Accordion("ℹ️ Configuration", open=False):
            gr.Markdown(
                f"""
                **System Configuration:**
                - **LLM Model:** {LLM_MODEL}
                - **Embedding Model:** {EMBEDDING_MODEL}
                - **Vector Store:** Qdrant ({QDRANT_HOST}:{QDRANT_PORT})
                - **Collection:** {COLLECTION_NAME}
                - **Hybrid Search:** {'✅ Enabled' if ENABLE_HYBRID else '❌ Disabled'}
                - **Fusion Method:** {FUSION_METHOD.upper()}
                - **Documents Path:** {DOCUMENTS_PATH}
                - **Chunk Size:** {CHUNK_SIZE}
                - **Chunking:** {'Hierarchical' if USE_HIERARCHICAL_CHUNKING else 'Character'}
                
                **Environment Variables:**
                - `QDRANT_HOST` - Qdrant host (default: localhost)
                - `QDRANT_PORT` - Qdrant port (default: 6333)
                - `COLLECTION_NAME` - Collection name
                - `DOCUMENTS_PATH` - Path to documents
                - `LLM_MODEL` - LLM model to use
                - `EMBEDDING_MODEL` - Embedding model
                - `ENABLE_HYBRID` - Enable hybrid search (true/false)
                - `FUSION_METHOD` - Fusion method (rrf/dbsf)
                """
            )
        
        # Event handlers
        def respond(message, history, session_id):
            if not message.strip():
                return history, citations_display.value
            return chat_wrapper(message, history, session_id)
        
        # Submit button
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, citations_display]
        ).then(
            lambda: "",
            None,
            [msg]
        )
        
        # Enter key
        msg.submit(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[chatbot, citations_display]
        ).then(
            lambda: "",
            None,
            [msg]
        )
        
        # Clear button
        clear_btn.click(
            clear_conversation,
            inputs=[session_id],
            outputs=[chatbot, citations_display]
        )
    
    return interface


async def initialize_system():
    """Initialize the RAG system before starting the app."""

    try:
        logger.info("Starting system initialization...")
        await setup_rag_system()
        logger.info("System initialization complete!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 70)
    print("🚀 RAG Chatbot with Hybrid Search")
    print("=" * 70)
    print("\nInitializing system...")
    
    # Initialize system
    success = asyncio.run(initialize_system())
    
    if not success:
        print("\n❌ Failed to initialize system. Please check the logs.")
        print("\n💡 Common issues:")
        print("   - Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("   - Check that DOCUMENTS_PATH exists and contains files")
        print("   - Verify API keys are set correctly")
        print("   - Install fastembed for hybrid search: pip install fastembed")
        sys.exit(1)
    
    print("\n✅ System initialized successfully!")
    print("\n" + "=" * 70)
    print("Starting web interface...")
    print("=" * 70)
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
