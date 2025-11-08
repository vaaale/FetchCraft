from typing import List, Optional, Dict, Any

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


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    model: str = Field(description="LLM model in use")
    collection: str = Field(description="Vector database collection name")
    hybrid_search_enabled: bool = Field(description="Whether hybrid search is enabled")


class ToolDefinition(BaseModel):
    """Tool definition for integration with AI agents."""
    type: str = Field(default="function", description="Tool type")
    function: Dict[str, Any] = Field(description="Function definition")
