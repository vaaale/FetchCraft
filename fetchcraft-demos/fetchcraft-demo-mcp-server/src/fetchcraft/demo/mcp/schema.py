from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for querying the RAG agent."""
    question: str = Field(
        description="The question to ask the RAG agent",
        examples=["What is the main topic discussed in the documents?"]
    )
    top_k: Optional[int] = Field(
        default=3,
        description="Number of documents to retrieve",
        ge=1,
        le=10
    )
    include_citations: Optional[bool] = Field(
        default=True,
        description="Whether to include parsing citations in the response"
    )


class Citation(BaseModel):
    """Citation information from retrieved documents."""
    source: str = Field(description="Source document identifier")
    filename: str = Field(description="Name of the parsing file")
    score: float = Field(description="Relevance score of this citation")
    text_preview: str = Field(description="Preview of the cited text")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(description="The agent's answer to the question")
    citations: Optional[List[Citation]] = Field(
        default=None,
        description="List of citations used to generate the answer"
    )
    processing_time_ms: float = Field(description="Time taken to process the query in milliseconds")
    model: str = Field(description="LLM model used for generation")


class FindFilesRequest(BaseModel):
    """Request model for finding files using semantic search."""
    query: str = Field(
        description="The search query to find relevant files"
    )
    num_results: Optional[int] = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=100
    )
    offset: Optional[int] = Field(
        default=0,
        description="Offset for pagination",
        ge=0
    )


class FileResult(BaseModel):
    """A file search result."""
    filename: str = Field(description="Name of the file")
    source: str = Field(description="Full path or parsing of the file")
    score: float = Field(description="Relevance score")
    text_preview: str = Field(description="Preview of the file content")


class FindFilesResponse(BaseModel):
    """Response model for file search."""
    files: List[FileResult] = Field(description="List of matching files")
    total: int = Field(description="Total number of results")
    offset: int = Field(description="Offset used for this request")


class GetFileRequest(BaseModel):
    """Request model for getting a file."""
    filename: str = Field(
        description="The name or path of the file to retrieve"
    )


class GetFileResponse(BaseModel):
    """Response model for getting a file."""
    filename: str = Field(description="Name of the file")
    content: str = Field(description="Full content of the file")
    metadata: dict = Field(default_factory=dict, description="File metadata")
