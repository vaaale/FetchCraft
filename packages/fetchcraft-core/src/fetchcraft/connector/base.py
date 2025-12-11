import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import *
from uuid import uuid4

import fsspec
from pydantic import BaseModel, Field
import magic

logger = logging.getLogger(__name__)


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
            with fs.open(str(path), "rb") as f:
                buf = f.read()
                _mimetype = magic.from_buffer(buf, mime=True)

            _mimetype_, _encoding_ = mimetypes.guess_type(path) or ("text/plain", "utf-8")
            if "text/plain" == _mimetype_ or "text/plain" == _mimetype:
                mimetype = "text/plain"
            else:
                mimetype = mimetype or _mimetype or _mimetype_
            encoding = encoding or _encoding_ or "utf-8"

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
    def get_name(self) -> str:
        ...

    @abstractmethod
    async def list_directories(self) -> List[str]:
        ...

    @abstractmethod
    async def glob(self) -> AsyncIterable['File']:
        ...
