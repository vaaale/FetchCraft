from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import *
from uuid import uuid4

import fsspec
from pydantic import BaseModel, Field
import mimetypes


class Role(BaseModel):
    name: str
    read: bool = False
    write: bool = False
    execute: bool = False


class File(BaseModel, ABC):
    id: str = Field(description="File ID", default=str(uuid4().int))
    path: Path
    mimetype: str
    encoding: str

    def __init__(self, path: Path, fs: fsspec.AbstractFileSystem, mimetype: Optional[str] = None, encoding: Optional[str] = None, **kwargs):
        if mimetype is None or encoding is None:
            _mimetype, _encoding = mimetypes.guess_type(path) or "application/octet-stream"
            mimetype = mimetype or _mimetype or "text/plain"
            encoding = encoding or _encoding or "utf-8"

        super().__init__(path=path, fs=fs, mimetype=mimetype, encoding=encoding, **kwargs)

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
