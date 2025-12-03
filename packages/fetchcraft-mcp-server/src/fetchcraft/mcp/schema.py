from pydantic import BaseModel


class FileResultSchema(BaseModel):
    """A file search result."""
    filename: str
    source: str
    score: float
    text_preview: str


class FindFilesResponseSchema(BaseModel):
    """Response for file search."""
    files: list[FileResultSchema]
    total: int
    offset: int

class QueryStructuredResponseSchema(BaseModel):
    answer: str
    citations: dict
    processing_time_ms: float
    model: str
