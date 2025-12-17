"""
Default implementation of DocumentPreviewService.

This module provides the default document preview service that retrieves
document content from Qdrant vector store.
"""
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.mcp.config import FetchcraftMCPConfig
from fetchcraft.mcp.interface import DocumentPreviewService, DocumentContent
from fetchcraft.vector_store import QdrantVectorStore


logger = logging.getLogger(__name__)


class DefaultDocumentPreviewService(BaseModel, DocumentPreviewService):
    """
    Default implementation of DocumentPreviewService using Qdrant.
    
    Retrieves document content from the vector store by node ID.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vector_store: QdrantVectorStore
    documents_path: str

    def __init__(self, vector_store: QdrantVectorStore, documents_path: str):
        super().__init__(vector_store=vector_store, documents_path=documents_path)

    @classmethod
    def create(cls, config: FetchcraftMCPConfig) -> "DefaultDocumentPreviewService":
        """
        Create a DefaultDocumentPreviewService from config.
        
        Args:
            config: MCP server configuration with database settings
            
        Returns:
            Configured DefaultDocumentPreviewService instance
        """
        embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url
        )

        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embeddings=embeddings,
            distance="Cosine",
            enable_hybrid=config.enable_hybrid,
            fusion_method=config.fusion_method
        )
        
        return cls(vector_store=vector_store, documents_path=config.documents_path)

    async def get_document(self, node_id: str) -> Optional[DocumentContent]:
        """
        Get document content by node ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            DocumentContent with full document content, or None if not found
        """
        try:
            node = await self.vector_store.get_node(node_id)
            
            if not node:
                logger.warning(f"Document not found: {node_id}")
                return None
            
            source = node.metadata.get("source", "Unknown")
            filename = node.metadata.get("filename", Path(source).name)

            full_path = Path(self.documents_path) / source

            if not full_path.exists():
                logger.warning(f"Document not found: {full_path}")
                text = node.text
            else:
                text = full_path.read_text()

            return DocumentContent(
                node_id=node.id,
                filename=filename,
                source=source,
                content=text,
                metadata=node.metadata,
            )
        except Exception as e:
            logger.error(f"Error retrieving document {node_id}: {e}")
            return None
