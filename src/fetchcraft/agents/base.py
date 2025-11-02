"""
Base agent interface for RAG framework.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

from .model import ChatMessage, AgentResponse
from .output_formatters.base import OutputFormatter
from .output_formatters.default_formatter import DefaultOutputFormatter


class BaseAgent(BaseModel, ABC):
    output_formatter: OutputFormatter | None = Field(description="Output formatter", default=DefaultOutputFormatter())

    """
    Abstract base class for agent implementations.
    
    Agents provide conversational interfaces that can use tools
    like retrievers to answer questions and perform tasks.
    """

    @abstractmethod
    async def query(self, question: str, messages: Optional[List[ChatMessage]] = None, **kwargs) -> AgentResponse:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
            :param question:
            :param messages:
        """
        pass

    async def aquery(self, question: str, messages: Optional[List[ChatMessage]] = None, **kwargs) -> AgentResponse:
        """
        Async version of query.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
            :param question:
            :param messages:
        """
        return await self.query(question, messages, **kwargs)

    def query_sync(self, question: str, messages: Optional[List[ChatMessage]] = None, **kwargs) -> AgentResponse:
        """
        Synchronous version of query.
        
        Args:
            question: The question to ask
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
            :param question:
            :param messages:
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

        return loop.run_until_complete(self.query(question, messages, **kwargs))
