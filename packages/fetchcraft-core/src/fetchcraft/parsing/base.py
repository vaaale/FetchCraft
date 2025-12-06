from abc import ABC, abstractmethod
from typing import *

from pydantic import BaseModel

from fetchcraft.connector import File
from fetchcraft.node import DocumentNode


class DocumentParser(BaseModel, ABC):

    @abstractmethod
    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None, task_id: Optional[str] = None) -> AsyncGenerator[DocumentNode, None]:
        """
        Parse a file and yield document nodes.
        """
        ...


    @abstractmethod
    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[DocumentNode, None]:
        """
        Read documents from the parsing.
        """
        ...

