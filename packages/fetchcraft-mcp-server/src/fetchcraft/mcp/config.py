"""
Configuration for Fetchcraft MCP Server.

This module provides the configuration class that can be extended
for custom server settings.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FetchcraftMCPConfig(BaseSettings):
    """
    Configuration for Fetchcraft MCP Server.
    
    This class provides core server settings. Users can extend this
    class to add their own configuration options.
    """
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, ge=1, le=65535, description="Server port")

    # Logging configuration
    logdir: Path = Field(default=Path("logs"), description="Directory for log files")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    # MCP configuration
    mcp_server_name: str = Field(default="fetchcraft", description="MCP server name")
    mcp_path: str = Field(default="/mcp", description="MCP endpoint path")
    
    # Frontend configuration
    frontend_base_url: str = Field(
        default="http://localhost:8001",
        description="Base URL for frontend assets (used in iframe redirects)"
    )
    frontend_port: int = Field(default=8003, description="Frontend port")
    frontend_dist: Optional[Path] = Field(
        default=None,
        description="Path to frontend dist directory (optional, auto-detected if not set)"
    )

    # Vector store configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    
    # Document store configuration
    mongo_uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection string")
    database_name: str = Field(default="fetchcraft", description="MongoDB database name")
    collection_name: str = Field(default="fetchcraft_chatbot", description="Collection name")
    documents_path: str = Field(default="Documents", description="Documents path")
    index_id: str = Field(default="docs-index", description="Index ID")

    # Embedding configuration
    embedding_model: str = Field(default="bge-m3", description="Embedding model name")
    embedding_api_key: str = Field(default="sk-321", description="Embedding API key")
    embedding_base_url: str = Field(default="http://wingman.akhbar.home:8000/v1", description="Embedding API base URL")

    # LLM configuration
    openai_api_key: str = Field(default="sk-321", description="OpenAI API key")
    openai_base_url: str = Field(default="http://wingman.akhbar.home:8000/v1", description="OpenAI API base URL")
    llm_model: str = Field(default="gpt-4-turbo", description="LLM model name")

    # Retrieval configuration
    enable_hybrid: bool = Field(default=True, description="Enable hybrid search")
    fusion_method: str = Field(default="rrf", description="Fusion method for hybrid search")
