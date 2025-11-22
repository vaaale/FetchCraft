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
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import psutil
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic_ai import Tool
from qdrant_client import QdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from fetchcraft.agents import RetrieverTool, PydanticAgent
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.mcp.html_formatter import FindFilesHTMLFormatter
from fetchcraft.mcp.settings import MCPServerSettings
from fetchcraft.vector_store import QdrantVectorStore

app = FastAPI()


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    agent: Optional[PydanticAgent] = None
    vector_index: Optional[VectorIndex] = None
    doc_store: Optional[MongoDBDocumentStore] = None
    retriever = None
    initialized: bool = False


# app_state = AppState()


async def setup_rag_system(settings: MCPServerSettings):
    """Set up the RAG system."""
    print("=" * 70)
    print("🚀 Fetchcraft MCP Server - Initializing")
    print("=" * 70)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url
    )

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
        distance="Cosine",
        enable_hybrid=settings.enable_hybrid,
        fusion_method=settings.fusion_method
    )
    print(f"   ✓ Vector store created")

    doc_store = MongoDBDocumentStore(
        database_name=settings.database_name,
        collection_name=settings.collection_name,
    )

    # Create vector index
    vector_index = VectorIndex(
        vector_store=vector_store,
        index_id=settings.index_id
    )

    # Create retriever
    retriever = vector_index.as_retriever(top_k=3, resolve_parents=True)

    # Create agent
    print(f"\n6️⃣  Creating RAG agent...")
    retriever_tool = RetrieverTool.from_retriever(retriever)
    tool_func = retriever_tool.get_tool_function()
    tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

    agent = PydanticAgent.create(
        model=settings.llm_model,
        tools=tools,
        retries=3
    )
    return agent, vector_index, doc_store, retriever


# ============================================================================
# MCP Server Setup
# ============================================================================


def build_mcp_server(settings: MCPServerSettings) -> FastMCP:
    app_state = AppState()

    @asynccontextmanager
    async def app_lifespan(mcp_app: FastMCP):
        """Initialize the RAG system on startup."""
        try:
            print("\n⚙️  Configuring RAG system...")
            agent, vector_index, doc_store, retriever = await setup_rag_system(settings)
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
        host=settings.host,
        port=settings.port,
    )

    # ============================================================================
    # MCP Tools
    # ============================================================================

    @mcp.tool()
    async def query(
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> str:
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
            return answer
            # return {
            #     "answer": answer,
            #     "citations": citations,
            #     "processing_time_ms": round(processing_time, 2),
            #     "model": settings.llm_model
            # }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print({key: val for key, val in os.environ.items()})
            raise RuntimeError(f"Error processing query: {str(e)}")

    @mcp.tool()
    async def find_files(
        query: str,
        num_results: int = 10,
        offset: int = 0,
        # format_html: bool = False
    ) -> dict | str:
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
            If format_html=True, returns a dictionary with 'html' key containing the formatted HTML
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

            for node in paginated_nodes:
                # Get filename from metadata
                source = node.node.metadata.get("source", "Unknown")
                filename = node.node.metadata.get("filename", Path(source).name)

                files.append({
                    "filename": filename,
                    "source": source,
                    "score": float(node.score) if node.score else 0.0,
                    "text_preview": (
                        node.node.text[:200] + "..."
                        if len(node.node.text) > 200
                        else node.node.text
                    )
                })

            result = {
                "files": files,
                "total": len(nodes),
                "offset": offset
            }

            # If HTML format requested, convert to HTML
            format_html = True
            if format_html:
                formatter = FindFilesHTMLFormatter()
                html_content = formatter.format(result)
                # return html_output

                response_header = f"Your search results for query: {query}"
                search_id = "_".join(query.split())
                artifact_header = f':::artifact{{identifier="{search_id}" type="text/html" title="File Search Results"}}'
                result = f"{response_header}\n{artifact_header}\n{html_content}\n"
                return result
            else:
                return result

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
                    file_path = Path(settings.documents_path) / filename

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

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "healthy",
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            **{ key: val for key, val in os.environ.items()}
        })

    return mcp


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server."""

    settings = MCPServerSettings()

    print("\n" + "=" * 70)
    print("🚀 Starting Fetchcraft MCP Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Model: {settings.llm_model}")
    print(f"  • Collection: {settings.collection_name}")
    print(f"  • Host: {settings.host}")
    print(f"  • Port: {settings.port}")
    print("=" * 70 + "\n")

    mcp = build_mcp_server(settings)
    asyncio.run(mcp.run_async(transport="streamable-http"))

if __name__ == "__main__":
    main()
