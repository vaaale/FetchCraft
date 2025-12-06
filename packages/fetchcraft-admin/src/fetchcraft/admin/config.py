"""
Base configuration for Fetchcraft Admin Server.

This module provides the base configuration class that can be extended
by handlers for their specific settings.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FetchcraftAdminConfig(BaseSettings):
    """
    Base configuration for Fetchcraft Admin Server.
    
    This class provides core server settings. Handlers can extend this
    class to add their own configuration options.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Logging configuration
    logdir: Path = Field(default=Path("logs"), description="Directory for log files")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    # Database configuration
    postgres_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/ingestion",
        description="PostgreSQL connection URL"
    )
    pool_min_size: int = Field(
        default=5, 
        ge=1, 
        description="Minimum connection pool size"
    )
    pool_max_size: int = Field(
        default=20, 
        ge=1, 
        description="Maximum connection pool size"
    )
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    
    # Worker configuration
    num_workers: int = Field(default=4, ge=1, description="Number of concurrent workers")
    
    # Callback configuration
    callback_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for callbacks from external services"
    )
    
    @property
    def database_display(self) -> str:
        """Get sanitized database URL for display (without password)."""
        if '@' in self.postgres_url:
            return self.postgres_url.split('@')[1]
        return self.postgres_url
