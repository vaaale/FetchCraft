from abc import abstractmethod

from pydantic import BaseModel

from ..model import CitationContainer


class OutputFormatter(BaseModel):

    @abstractmethod
    def format(self, response: str, citations: CitationContainer) -> str:
        pass
