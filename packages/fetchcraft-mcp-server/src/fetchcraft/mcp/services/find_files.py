"""
Default implementation of FindFilesService using Qdrant and MongoDB.
"""
import logging
from typing import List

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.mcp.config import FetchcraftMCPConfig
from fetchcraft.mcp.interface import FindFilesService
from fetchcraft.node import NodeWithScore
from fetchcraft.vector_store import QdrantVectorStore


class DefaultFindFilesService(FindFilesService, BaseModel):
    """
    Default implementation of FindFilesService using Qdrant vector store.
    
    This implementation uses:
    - Qdrant for vector similarity search
    - MongoDB for document storage
    - OpenAI-compatible embeddings
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: logging.Logger = logging.getLogger("FindFilesService")

    vector_index: VectorIndex

    def __init__(self, vector_index: VectorIndex, log_level: str = logging.INFO):
        super().__init__(vector_index=vector_index)
        self.logger.setLevel(log_level)
        self.logger.info(f"   ✓ Retriever initialized.")

    @classmethod
    def create(cls, config: FetchcraftMCPConfig) -> "DefaultFindFilesService":
        """
        Create a DefaultFindFilesService from config.
        
        Args:
            config: MCP server configuration with database settings
            
        Returns:
            Configured DefaultFindFilesService instance
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
            connection_string=config.mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name,
        )

        # Create vector index
        vector_index = VectorIndex(
            vector_store=vector_store,
            doc_store=doc_store,
            index_id=config.index_id
        )

        return cls(vector_index=vector_index)

    async def find_files(self, query: str, num_results: int = 10, offset: int = 0) -> List[NodeWithScore]:
        """
        Find files using semantic search with pagination.
        
        Args:
            query: The search query
            num_results: Number of results to return
            offset: Offset for pagination
            
        Returns:
            List of NodeWithScore objects
        """
        self.logger.debug(f"find_files(query={query}, num_results={num_results}, offset={offset})")

        total_needed = num_results + offset

        retriever = self.vector_index.as_retriever(top_k=3, resolve_parents=True)
        nodes = await retriever.aretrieve(query, top_k=total_needed)

        # Apply offset and limit
        paginated_nodes = nodes[offset:offset + num_results]
        self.logger.debug(f"find_files() -> {len(paginated_nodes)} nodes")
        return paginated_nodes
