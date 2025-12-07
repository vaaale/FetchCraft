from abc import ABC, abstractmethod
from typing import *

from pydantic import BaseModel

from fetchcraft.connector import File
from fetchcraft.node import DocumentNode


class DocumentParser(BaseModel, ABC):

    @property
    def is_remote(self):
        return False

    @abstractmethod
    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None, **parser_kwargs) -> AsyncGenerator[DocumentNode, None]:
        """
        Parse a file and yield document nodes.
        """
        ...
