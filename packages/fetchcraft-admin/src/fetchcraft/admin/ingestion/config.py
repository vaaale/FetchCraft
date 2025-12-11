"""
Ingestion-specific configuration.

This module provides configuration settings specific to the ingestion handler.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field

from fetchcraft.admin.config import FetchcraftAdminConfig


class IngestionConfig(FetchcraftAdminConfig):
    """
    Configuration for the ingestion handler.
    
    Extends the base FetchcraftAdminConfig with ingestion-specific settings.
    """
    
    # Document paths
    documents_path: Path = Field(
        default=Path("Documents"), 
        description="Root path for documents"
    )
    
    # Vector store configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant server port")
    collection_name: str = Field(default="fetchcraft_chatbot", description="Qdrant collection name")
    
    # External services
    docling_server: str = Field(
        default="http://localhost:8001",
        description="Docling parser server URL"
    )
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )

    # LLM Config
    llm_model: str = Field(default="gpt-4-turbo", description="LLM model name")
    openai_api_key: str = Field(default="sk-321", description="LLM API key")
    openai_base_url: Optional[str] = Field(default=None, description="LLM API base URL")
    
    # Embeddings configuration
    embedding_model: str = Field(default="bge-m3", description="Embedding model name")
    embedding_api_key: str = Field(default="sk-321", description="Embedding API key")
    embedding_base_url: Optional[str] = Field(default=None, description="Embedding API base URL")
    index_id: str = Field(default="docs-index", description="Vector index identifier")
    
    # Chunking configuration
    chunk_size: int = Field(default=8192, ge=1, description="Base chunk size")
    child_chunks_str: str = Field(
        default="4096,1024",
        description="Child chunk sizes for hierarchical chunking (comma-separated)",
        alias="child_chunks"
    )
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    
    # Hybrid search configuration
    enable_hybrid: bool = Field(default=True, description="Enable hybrid search")
    fusion_method: str = Field(default="rrf", description="Fusion method for hybrid search")
    
    @property
    def child_chunks(self) -> list[int]:
        """Parse child_chunks from comma-separated string."""
        return [int(x.strip()) for x in self.child_chunks_str.split(',') if x.strip()]
