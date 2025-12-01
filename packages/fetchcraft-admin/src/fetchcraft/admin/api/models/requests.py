"""Request models for the Admin API."""
from typing import Any, Dict, Optional
from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    """Request to create a new ingestion job."""
    name: str
    source_path: str  # Relative to document root


class DoclingNodeCallback(BaseModel):
    """Callback payload from docling server for a single node."""
    job_id: str
    filename: str
    node_index: int
    total_nodes: int
    node: Dict[str, Any]


class DoclingCompletionCallback(BaseModel):
    """Callback payload from docling server for completion."""
    job_id: str
    filename: str
    status: str  # "completed" or "failed"
    total_nodes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None
