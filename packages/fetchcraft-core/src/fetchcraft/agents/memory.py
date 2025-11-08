from pathlib import Path
from typing import *

from pydantic import BaseModel, Field

from fetchcraft.agents.base import ChatMessage
from fetchcraft.agents.model import Citation

CONTEXT_TEMPLATE = (
    "Below are some previously used context that might be useful.\n"
    "{context}\n"
    "\n"
    "If the context is not relevant to the question, ignore it."
)

class Memory(BaseModel):
    history: List[Any] = Field(default_factory=list)
    hist_size: int = Field(default=2)

    def add_memory(self, query: ChatMessage, response: ChatMessage, citations: List[Citation]):
        self.history.append({
            "query": query,
            "response": response,
            "citations": citations
        })

    def get_prompt(self, query: str) -> Tuple[str, List[ChatMessage]]:
        if len(self.history) < 1:
            return query, []

        history = []
        for item in self.history:
            history.append(item["query"])
            history.append(item["response"])

        context = self.build_user_prompt(query)
        return context, history

    def build_user_prompt(self, query: str) -> str:
        context = (
            f"Query: {query}\n\n"
            "Below is some information from previous interactions that may, or may not be useful:\n"
            "## Previously used citations:\n"
        )
        for item in self.history:
            citations = item["citations"]
            if citations:

                for cit in citations:
                    m = cit.node.metadata
                    filepath = m.get("path", m.get("file", m.get("filename", m.get("parsing", None))))
                    if filepath:
                        filepath = f"File: {Path(filepath).name}\n--------------------------\n"
                    else:
                        filepath = ""

                    context += (
                        f"Citation: {cit.citation_id}\n"
                        "--------------------------\n"
                        f"{filepath}"
                        "Content:\n"
                        "--------------------------\n"
                        f"{cit.node.text}\n"
                        "--------------------------\n\n"
                    )
        context += "The information above should never be cited. If you need to cite it, use the tools to fetch it again."
        return context
