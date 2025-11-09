from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import *
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(BaseModel):
    name: str
    read: bool = False
    write: bool = False
    execute: bool = False


class File(BaseModel, ABC):
    id: str = Field(description="File ID", default=str(uuid4().int))
    path: Path

    @property
    def path(self) -> Path:
        return self.path

    @abstractmethod
    async def read(self) -> bytes:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def permissions(self) -> List[Role]:
        ...

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        ...


class Connector(ABC):

    @abstractmethod
    async def read(self) -> AsyncIterable['File']:
        ...
