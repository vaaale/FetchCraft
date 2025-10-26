"""
Base agent interface for RAG framework.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, TypedDict, Optional

from pydantic import BaseModel, Field

from rag_framework import Node


class ChatMessage(BaseModel):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

    @classmethod
    def user_message(cls, content: str) -> 'ChatMessage':
        return cls(role='user', timestamp=datetime.now().isoformat(), content=content)

    @classmethod
    def assistant_message(cls, content: str) -> 'ChatMessage':
        return cls(role='model', timestamp=datetime.now().isoformat(), content=content)

@dataclass
class Citation:
    citation_id: int
    call_id: str
    tool_name: str
    query: str
    node: Node

    @property
    def title(self):
        return self.node.metadata.get('title', self.node.metadata.get("name", f"Document {self.citation_id}"))

    @property
    def url(self):
        metadata = self.node.metadata
        url = metadata.get("url", metadata.get("link", metadata.get("path", metadata.get("file", metadata.get("filename", None)))))
        return url

    def __repr__(self):
        return f"{self.title}\n{self.node.text[:100]}"

@dataclass
class CitationContainer:
    _citations: List[Citation] = field(default_factory=list)
    _id_map: Dict[int, Citation] = field(default_factory=dict)
    _cited: List[Citation] = field(default_factory=list)

    def add(self, call_id: str, tool_name: str, query: str, node: Node) -> Citation:
        citation_id = len(self._citations) + 1
        citation = Citation(citation_id=citation_id, call_id=call_id, tool_name=tool_name, query=query, node=node)
        self._citations.append(citation)
        self._id_map[citation_id] = citation
        return citation

    def add_cited(self, citation: Citation):
        self._cited.append(citation)

    @property
    def citations(self):
        return self._cited

    @property
    def all_citations(self):
        return self._citations

    def citation(self, citation_id: int) -> Citation:
        return self._id_map.get(citation_id, None)


class AgentResponse(BaseModel):
    """Agent response with citations"""
    query: ChatMessage = Field(description="Query text", default=None)
    response: ChatMessage = Field(description="Response text", default=None)
    citations: List[Citation] = Field(description="List of used citations", default=[])
    all_citations: List[Citation] = Field(description="List of all citations", default=[])

    def __str__(self):
        return self.response

    def __repr__(self):
        return self.response


class BaseAgent(BaseModel, ABC):

    def _post_process_citations(self, response: str, citations: CitationContainer) -> str:
        matches = re.finditer(r"\[(?P<title>.*)\]\((?P<citation_id>\d+)\).*", response)
        for match in matches:
            citation_id = match.group("citation_id")
            title = match.group("title")
            citation = citations.citation(int(citation_id))
            citations.add_cited(citation)

            # Replace
            orig_citation = response[match.start():match.end()]
            new_citation = f"[{citation.title or title}]({citation.url or citation_id})"
            response = response.replace(orig_citation, new_citation)

        return response

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
