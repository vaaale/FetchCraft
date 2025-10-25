"""
Base agent interface for RAG framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List

from pydantic import BaseModel

class AgentResponse(BaseModel):
    """Response from an agent."""
    response: str
    citations: List[Any] = []


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
