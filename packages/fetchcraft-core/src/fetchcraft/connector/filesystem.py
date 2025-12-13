import stat
from pathlib import Path
from pwd import getpwuid
from typing import *

import fsspec
from pydantic import ConfigDict, BaseModel

from fetchcraft.connector import Connector, File
from fetchcraft.connector.base import Role
import logging

logger = logging.getLogger(__name__)

class LocalFile(File):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fs: fsspec.AbstractFileSystem

    def __init__(self, path: Path, fs: Optional[fsspec.AbstractFileSystem] = None):
        if fs is None:
            fs = fsspec.filesystem("file")
        super().__init__(fs=fs, path=path)

    @classmethod
    def from_path(cls, path: Path, fs: Optional[fsspec.AbstractFileSystem] = None):
        return cls(path=path, fs=fs)


    async def read(self) -> bytes:
        return self.fs.open(str(self.path), "rb").read()


    def name(self) -> str:
        return self.path.name

    def permissions(self) -> List[Role]:
        st = self.fs.stat(str(self.path))
        ur = bool(stat.S_IRUSR & st['mode'])
        uw = bool(stat.S_IWUSR & st['mode'])
        ux = bool(stat.S_IXUSR & st['mode'])
        user_role = Role(name="user", read=ur, write=uw, execute=ux)

        gr = bool(stat.S_IRGRP & st['mode'])
        gw = bool(stat.S_IWGRP & st['mode'])
        gx = bool(stat.S_IXGRP & st['mode'])
        group_role = Role(name="group", read=gr, write=gw, execute=gx)

        otr = bool(stat.S_IRGRP & st['mode'])
        otw = bool(stat.S_IWGRP & st['mode'])
        otx = bool(stat.S_IXGRP & st['mode'])
        other_role = Role(name="other", read=otr, write=otw, execute=otx)
        return [user_role.model_dump(), group_role.model_dump(), other_role.model_dump()]


    def metadata(self) -> dict:

        def _getpwuid(uid: int) -> str:
            try:
                return getpwuid(uid).pw_name
            except KeyError:
                return str(uid)

        def _getpwgid(gid: int) -> str:
            try:
                return getpwuid(gid).pw_name
            except KeyError:
                return str(gid)

        st = self.fs.stat(str(self.path))
        return {
            "filename": self.path.name,
            "mimetype": self.mimetype,
            "encoding": self.encoding,
            "source": str(self.path),
            "size": st["size"],
            "modified": st["mtime"],
            "created": st["created"],
            "owner": _getpwuid(st["uid"]),
            "group": _getpwgid(st["gid"]),
            "permissions": self.permissions()
        }


class FilesystemConnector(BaseModel, Connector):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root_path: Path
    sub_path: Path
    fs: fsspec.AbstractFileSystem
    filter: Optional[Callable[[LocalFile], Awaitable[bool]]] = None

    def __init__(self, root_path: Path, fs: Optional[fsspec.AbstractFileSystem] = None, sub_path: Path = Path("/"), filter: Optional[Callable[[LocalFile], Awaitable[bool]]] = None):
        if fs is None:
            fs = fsspec.filesystem("dir", path=root_path)

        super().__init__(root_path=root_path, sub_path=sub_path, fs=fs, filter=filter)


    def get_name(self) -> str:
        return f"FilesystemConnector"

    async def list_directories(self) -> List[str]:
        dirs = []
        for path in self.fs.glob("**/*"):
            if self.fs.isdir(path):
                dirs.append(path)
        return dirs

    async def glob(self) -> AsyncIterable[LocalFile]:
        logger.info(f"Ingesting files from {self.root_path}")
        ingest_from_path = self.sub_path / "**/*"
        for path in self.fs.glob(str(ingest_from_path)):
            if self.fs.isdir(path) or (self.filter and not await self.filter(LocalFile(path=path, fs=self.fs))):
                continue
            logger.info(f"Ingesting file: {path}")
            yield LocalFile(path=path, fs=self.fs)
