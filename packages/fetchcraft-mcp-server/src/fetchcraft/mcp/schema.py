from pydantic import BaseModel


class FileResultSchema(BaseModel):
    """A file search result."""
    node_id: str
    filename: str
    source: str
    score: float
    text_preview: str


class DocumentContentSchema(BaseModel):
    """Full document content for preview."""
    node_id: str
    filename: str
    source: str
    content: str
    metadata: dict


class FindFilesResponseSchema(BaseModel):
    """Response for file search."""
    files: list[FileResultSchema]
    total: int
    offset: int
    has_more: bool

class QueryStructuredResponseSchema(BaseModel):
    answer: str
    citations: dict
    processing_time_ms: float
    model: str
