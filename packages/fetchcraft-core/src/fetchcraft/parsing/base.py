from abc import ABC, abstractmethod
from typing import *

from pydantic import BaseModel

from fetchcraft.node import DocumentNode


class DocumentParser(BaseModel, ABC):

    @abstractmethod
    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[DocumentNode, None]:
        """
        Read documents from the parsing.
        """
        pass

