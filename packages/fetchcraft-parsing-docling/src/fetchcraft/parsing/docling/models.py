"""
Pydantic models for the Docling parsing API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParseResponse(BaseModel):
    """Response model for document parsing."""
    filename: str = Field(description="Name of the parsed file")
    success: bool = Field(description="Whether parsing was successful")
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DocumentNode objects as dictionaries"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if parsing failed"
    )
    num_nodes: int = Field(description="Number of nodes created")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class BatchParseResponse(BaseModel):
    """Response model for batch document parsing."""
    results: List[ParseResponse] = Field(description="Results for each file")
    total_files: int = Field(description="Total number of files processed")
    successful: int = Field(description="Number of successfully parsed files")
    failed: int = Field(description="Number of failed files")
    total_nodes: int = Field(description="Total number of nodes created")
    total_processing_time_ms: float = Field(description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    config: Dict[str, Any] = Field(description="Server configuration")
    environment: Dict[str, Any] = Field(description="Environment variables")
