"""
Default implementation of QueryService using Qdrant and MongoDB.
"""
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Tool
from qdrant_client import QdrantClient

from fetchcraft.agents import RetrieverTool, PydanticAgent
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.mcp.config import FetchcraftMCPConfig
from fetchcraft.mcp.interface import QueryService, QueryResponse
from fetchcraft.vector_store import QdrantVectorStore


class DefaultQueryService(QueryService, BaseModel):
    """
    Default implementation of QueryService using RAG with Qdrant.
    
    This implementation uses:
    - Qdrant for vector similarity search
    - MongoDB for document storage
    - PydanticAgent for RAG query processing
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_index: VectorIndex
    config: FetchcraftMCPConfig

    def __init__(self, vector_index: VectorIndex, config: FetchcraftMCPConfig):
        super().__init__(vector_index=vector_index, config=config)

    @classmethod
    def create(cls, config: FetchcraftMCPConfig) -> "DefaultQueryService":
        """
        Create a DefaultQueryService from config.
        
        Args:
            config: MCP server configuration with database settings
            
        Returns:
            Configured DefaultQueryService instance
        """
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url
        )

        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embeddings=embeddings,
            distance="Cosine",
            enable_hybrid=config.enable_hybrid,
            fusion_method=config.fusion_method
        )

        doc_store = MongoDBDocumentStore(
            database_name=config.database_name,
            collection_name=config.collection_name,
        )

        # Create vector index
        vector_index = VectorIndex(
            vector_store=vector_store,
            doc_store=doc_store,
            index_id=config.index_id
        )

        return cls(vector_index=vector_index, config=config)

    async def query(
        self,
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> QueryResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            top_k: Number of documents to retrieve
            include_citations: Whether to include citations in response
            
        Returns:
            QueryResponse with answer, citations, and model info
        """
        try:
            # Query the agent
            print(f"\n6️⃣  Creating RAG agent...")

            # Create retriever Tool
            retriever = self.vector_index.as_retriever(top_k=top_k, resolve_parents=True)
            retriever_tool = RetrieverTool.from_retriever(retriever, max_chunk_size=2000)
            tool_func = retriever_tool.get_tool_function()
            tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

            agent = PydanticAgent.create(
                model=self.config.llm_model,
                tools=tools,
                retries=3
            )

            response = await agent.query(question)

            # Extract answer
            answer = response.response.content

            # Extract citations if requested
            citations = None
            if include_citations and response.citations:
                citations = {}
                for citation_id, citation in response.citations.items():
                    # Get a sample. Using n first paragraphs
                    text = citation.node.text.replace("\r\n", "\n")
                    paragraphs = text.split("\n\n")
                    preview = "\n\n".join(paragraphs[:5]) + f" ....\n({max(len(paragraphs) - 5, 0)} more paragraphs)"

                    filename = citation.node.metadata.get("filename", Path(citation.node.metadata.get("source", "N/A")).name)
                    citations[citation_id] = {
                        "citation_id": citation_id,
                        "source": citation.node.metadata.get("source", "Unknown"),
                        "filename": filename,
                        "score": float(citation.node.score) if hasattr(citation.node, "score") else 0.0,
                        "text_preview": preview
                    }

            result = QueryResponse(
                answer=answer,
                citations=citations or {},
                model=self.config.llm_model
            )
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            print({key: val for key, val in os.environ.items()})
            raise RuntimeError(f"Error processing query: {str(e)}")
