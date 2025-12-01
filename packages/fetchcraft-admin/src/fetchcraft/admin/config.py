"""
Configuration management using pydantic-settings.

This module provides type-safe configuration loading from environment variables
with validation and default values.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )
    
    # Database configuration
    postgres_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/ingestion",
        description="PostgreSQL connection URL"
    )
    pool_min_size: int = Field(
        default=5, 
        ge=1, 
        description="Minimum connection pool size (will be auto-calculated based on workers if needed)"
    )
    pool_max_size: int = Field(
        default=20, 
        ge=1, 
        description="Maximum connection pool size (recommended: num_workers * 3 + 10)"
    )
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    
    # Document paths
    documents_path: Path = Field(default=Path("Documents"), description="Root path for documents")
    
    # Vector store configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant server port")
    collection_name: str = Field(default="fetchcraft_chatbot", description="Qdrant collection name")
    
    # External services
    docling_server: str = Field(
        default="http://localhost:8001",
        description="Docling parser server URL"
    )
    callback_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for callbacks from external services"
    )
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    
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
    
    # Worker configuration
    num_workers: int = Field(default=4, ge=1, description="Number of concurrent workers")
    
    @property
    def child_chunks(self) -> list[int]:
        """Parse child_chunks from comma-separated string."""
        return [int(x.strip()) for x in self.child_chunks_str.split(',') if x.strip()]
    
    @property
    def database_display(self) -> str:
        """Get sanitized database URL for display (without password)."""
        if '@' in self.postgres_url:
            return self.postgres_url.split('@')[1]
        return self.postgres_url


# Global settings instance
settings = Settings()
