from abc import ABC, abstractmethod
from typing import *

from pydantic import BaseModel

from fetchcraft.node import DocumentNode


class DocumentSource(BaseModel, ABC):

    @abstractmethod
    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:
        """
        Read documents from the source.
        """
        pass

