"""
Base agent interface for RAG framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

from pydantic import BaseModel, Field

from rag_framework import Node


class AgentResponse(BaseModel):
    """Agent response with citations"""
    response: str = Field(description="Response text", default=None)
    citations: List[str] = Field(description="List of citations", default=[])

@dataclass
class Citation:
    citation_id: int
    call_id: str
    tool_name: str
    query: str
    node: Node

@dataclass
class CitationContainer:
    _citations: List[Citation] = field(default_factory=list)
    _id_map: Dict[str, int] = field(default_factory=dict)

    def add(self, call_id: str, tool_name: str, query: str, node: Node) -> Citation:
        citation_id = len(self._citations) + 1
        citation = Citation(citation_id=citation_id, call_id=call_id, tool_name=tool_name, query=query, node=node)
        self._citations.append(citation)
        self._id_map[node.id] = citation_id
        return citation

    @property
    def citations(self):
        return self._citations




class BaseAgent(BaseModel, ABC):
    """
    Abstract base class for agent implementations.
    
    Agents provide conversational interfaces that can use tools
    like retrievers to answer questions and perform tasks.
    """
    
    @abstractmethod
    async def query(self, question: str, **kwargs) -> str:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        pass
    
    @abstractmethod
    async def aquery(self, question: str, **kwargs) -> str:
        """
        Async version of query.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        pass
    
    def query_sync(self, question: str, **kwargs) -> str:
        """
        Synchronous version of query.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            raise RuntimeError(
                "Cannot use sync method when async loop is already running. "
                "Use query() or aquery() instead."
            )
        
        return loop.run_until_complete(self.query(question, **kwargs))
