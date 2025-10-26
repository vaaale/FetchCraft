"""
RetrieverTool for integrating retrievers with agents.
"""

from typing import Optional, Callable, Any, List
import logging

from .. import Node

try:
    from pydantic_ai import RunContext
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    RunContext = None

from ..retriever.base import Retriever
from .base import CitationContainer, Citation

logger = logging.getLogger(__name__)

class RetrieverTool:
    """
    Tool wrapper for retrievers to be used with agents.
    
    This class wraps a retriever and provides a tool function
    that can be registered with pydantic-ai agents.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        name: str = "search_documents",
        description: Optional[str] = None,
        formatter: Optional[Callable] = None
    ):
        """
        Initialize the RetrieverTool.
        
        Args:
            retriever: The retriever to wrap
            name: Name of the tool (default: "search_documents")
            description: Tool description for the LLM
            formatter: Custom formatter for results (optional)
        """
        self.retriever = retriever
        self.name = name
        self.formatter = formatter or self._default_formatter
        
        # Default description
        if description is None:
            description = """Search for relevant documents.

Args:
    query: The search query

Returns:
    Retrieved documents as formatted text"""
        
        self.description = description
    
    @classmethod
    def from_retriever(
        cls,
        retriever: Retriever,
        name: str = "search_documents",
        description: Optional[str] = None,
        formatter: Optional[Callable] = None
    ) -> 'RetrieverTool':
        """
        Create a RetrieverTool from a retriever.
        
        Args:
            retriever: The retriever to wrap
            name: Name of the tool (default: "search_documents")
            description: Tool description for the LLM
            formatter: Custom formatter for results (optional)
            
        Returns:
            RetrieverTool instance
        """
        return cls(
            retriever=retriever,
            name=name,
            description=description,
            formatter=formatter
        )
    
    def _default_formatter(self, citations: List[Citation]) -> str:
        """
        Default formatter for retrieval results.
        
        Args:
            results: List of NodeWithScore objects
            citations: List of Citation objects
            
        Returns:
            Formatted string representation
        """
        if not citations:
            return "No relevant documents found."

        formatted_results = []
        for citation in citations:
            formatted_results.append(
                f"Document {citation.citation_id} (relevance: {citation.node.score:.3f}):\n{citation.node.text}\n"
            )
        
        return "\n".join(formatted_results)
    
    async def __call__(self, ctx, query: str) -> str:
        """
        Execute the retrieval tool.
        
        Args:
            ctx: Context from pydantic-ai (RunContext) - not type-checked here for flexibility
            query: The search query
            
        Returns:
            Formatted retrieval results
        """
        logger.info(f"Retrieving documents for query: {query}")
        results = await self.retriever.retrieve(query)

        # Add each retrieved node as a citation
        citations = [ctx.deps.add(ctx.tool_call_id, self.name, query, result) for result in results]

        logger.info(f"Retrieved {len(results)} documents")
        tool_response = self.formatter(citations)
        logger.info(f"{tool_response}")
        return tool_response
    
    def get_tool_function(self):
        """
        Get the async function to be registered as a tool.
        
        Returns:
            Async function for tool registration
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is required for RetrieverTool. "
                "Install it with: pip install pydantic-ai"
            )
        
        async def search_documents(ctx: RunContext[Any], query: str) -> str:
            """Search for relevant documents."""
            return await self(ctx, query)
        
        # Set the docstring from description
        search_documents.__doc__ = self.description
        search_documents.__name__ = self.name
        
        return search_documents
    
    def __repr__(self) -> str:
        return f"RetrieverTool(name={self.name})"
