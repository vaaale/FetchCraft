"""
API endpoints for the Fetchcraft MCP Server frontend.

This module provides REST API endpoints for the file finder web application.
"""
import time
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException

from fetchcraft.mcp.interface import FindFilesService, QueryService
from fetchcraft.mcp.schema import FileResultSchema, FindFilesResponseSchema, QueryStructuredResponseSchema


def create_api_router(find_files_service: FindFilesService, query_service: QueryService) -> APIRouter:
    """
    Create the API router with all endpoints.
    
    Args:
        find_files_service: The file finder service instance
        
    Returns:
        Configured API router
    """
    router = APIRouter(prefix="/api")

    @router.get("/query", name="query", operation_id="query", response_model=QueryStructuredResponseSchema)
    async def query(
        question: str,
        top_k: int = 3,
        include_citations: bool = True
    ) -> QueryStructuredResponseSchema:
        """
        Query the RAG agent with a question.

        This tool retrieves relevant documents from the vector database and uses
        an LLM to generate a comprehensive answer based on the retrieved context.

        Args:
            question: The question to ask the RAG agent
            top_k: Number of documents to retrieve (1-10)
            include_citations: Whether to include parsing citations in the response

        Returns:
            Dictionary containing the answer, citations, processing time, and model used
        """

        start_time = time.time()
        result = await query_service.query(question, top_k, include_citations)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        query_response = QueryStructuredResponseSchema(
            answer=result.answer,
            citations=result.citations or {},
            processing_time_ms=round(processing_time, 2),
            model=result.model
        )
        return query_response

    @router.get("/find-files", name="find-files", operation_id="find_files", response_model=FindFilesResponseSchema)
    async def find_files(
        query: str = Query(..., description="Search query"),
        num_results: int = Query(10, ge=1, le=100, description="Number of results to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination")
    ) -> FindFilesResponseSchema:
        """
        Find files using semantic search.
        
        Args:
            query: The search query
            num_results: Number of results to return (1-100)
            offset: Offset for pagination (starting from 0)
            
        Returns:
            FindFilesResponseSchema with matching files
        """
        try:
            paginated_nodes = await find_files_service.find_files(
                query=query,
                num_results=num_results,
                offset=offset
            )

            # Convert nodes to file results
            files = []
            for node in paginated_nodes:
                # Get filename from metadata
                source = node.node.metadata.get("source", "Unknown")
                filename = node.node.metadata.get("filename", Path(source).name)

                # Get a sample. Using n first paragraphs
                text = node.node.text.replace("\r\n", "\n")
                paragraphs = text.split("\n\n")
                preview = "\n\n".join(paragraphs[:5])

                files.append({
                    "filename": filename,
                    "source": source,
                    "score": float(node.score) if node.score else 0.0,
                    "text_preview": (
                        preview + f" ....\n({max(len(paragraphs) - 5, 0)} more paragraphs)"
                    )
                })

            return FindFilesResponseSchema(
                files=[FileResultSchema(**file) for file in files],
                total=len(paginated_nodes),
                offset=offset
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")

    return router
