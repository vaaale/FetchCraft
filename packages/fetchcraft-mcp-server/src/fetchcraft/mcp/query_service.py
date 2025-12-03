import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Tool
from qdrant_client import QdrantClient

from fetchcraft.agents import RetrieverTool, PydanticAgent
from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.mcp.model import QueryResponse
from fetchcraft.mcp.settings import MCPServerSettings
from fetchcraft.vector_store import QdrantVectorStore


class QueryService(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_index: VectorIndex
    settings: MCPServerSettings

    def __init__(self, vector_index: VectorIndex, settings: MCPServerSettings):
        super().__init__(vector_index=vector_index, settings=settings)

    @classmethod
    def create(cls, settings: MCPServerSettings) -> "QueryService":
        """Set up the RAG system."""
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

        doc_store = MongoDBDocumentStore(
            database_name=settings.database_name,
            collection_name=settings.collection_name,
        )

        # Create vector index
        vector_index = VectorIndex(
            vector_store=vector_store,
            doc_store=doc_store,
            index_id=settings.index_id
        )

        return cls(vector_index=vector_index, settings=settings)

    async def query(
        self,
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> QueryResponse:
        try:
            # Query the agent
            print(f"\n6️⃣  Creating RAG agent...")

            # Create retriever Tool
            retriever = self.vector_index.as_retriever(top_k=top_k, resolve_parents=True)
            retriever_tool = RetrieverTool.from_retriever(retriever, max_chunk_size=2000)
            tool_func = retriever_tool.get_tool_function()
            tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

            agent = PydanticAgent.create(
                model=self.settings.llm_model,
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
                model=self.settings.llm_model
            )
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            print({key: val for key, val in os.environ.items()})
            raise RuntimeError(f"Error processing query: {str(e)}")
