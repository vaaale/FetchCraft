from datetime import datetime
from typing import *
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from fetchcraft.node import Node

class ChatMessage(BaseModel):
    """Format of messages sent to the browser."""

    role: Literal['user', 'assistant']
    timestamp: str
    content: str

    @classmethod
    def user_message(cls, content: str) -> 'ChatMessage':
        return cls(role='user', timestamp=datetime.now().isoformat(), content=content)

    @classmethod
    def assistant_message(cls, content: str) -> 'ChatMessage':
        return cls(role='assistant', timestamp=datetime.now().isoformat(), content=content)


@dataclass
class Citation:
    citation_id: int
    call_id: str
    tool_name: str
    query: str
    node: Node
    cited_name: str = ""

    @property
    def title(self):
        return self.node.metadata.get('title', self.node.metadata.get("name", f"Document {self.citation_id} - {self.cited_name}"))

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
    _cited: Dict[Any,Citation] = field(default_factory=dict)

    def add(self, call_id: str, tool_name: str, query: str, node: Node) -> Citation:
        citation_id = len(self._citations) + 1
        citation = Citation(citation_id=citation_id, call_id=call_id, tool_name=tool_name, query=query, node=node)
        self._citations.append(citation)
        self._id_map[citation_id] = citation
        return citation

    def add_cited(self, citation_id: Any, citation: Citation):
        self._cited[citation_id] = citation

    @property
    def citations(self):
        return self._cited

    @property
    def all_citations(self) -> dict:
        return {c.citation_id: c for c in self._citations}

    def citation(self, citation_id: int) -> Citation:
        return self._id_map.get(citation_id, None)


class AgentResponse(BaseModel):
    """Agent response with citations"""
    query: ChatMessage = Field(description="Query text", default=None)
    response: ChatMessage = Field(description="Response text", default=None)
    citations: Dict[Any, Citation] = Field(description="List of used citations", default=dict)
    all_citations: Dict[Any, Citation] = Field(description="List of all citations", default=[])

    def __str__(self):
        return self.response.content

    def __repr__(self):
        return self.response.content
