import stat
from pathlib import Path
from pwd import getpwuid
from typing import *

import fsspec
from pydantic import ConfigDict

from fetchcraft.connector import Connector, File
from fetchcraft.connector.base import Role


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
            "source": str(self.path),
            "size": st["size"],
            "modified": st["mtime"],
            "created": st["created"],
            "owner": _getpwuid(st["uid"]),
            "group": _getpwgid(st["gid"]),
            "permissions": self.permissions()
        }


class FilesystemConnector(Connector):
    path: Path
    fs: fsspec.AbstractFileSystem
    filter: Optional[Callable[[LocalFile], bool]] = None

    def __init__(self, path: Path, fs: Optional[fsspec.AbstractFileSystem] = None, filter: Optional[Callable[[LocalFile], bool]] = None):
        if fs is None:
            fs = fsspec.filesystem("dir", path=path)
        self.path = path
        self.fs =fs
        self.filter = filter

    def get_name(self) -> str:
        return f"FilesystemConnector"

    async def list_directories(self) -> List[str]:
        dirs = []
        for path in self.fs.glob("**/*"):
            if self.fs.isdir(path):
                dirs.append(path)
        return dirs

    async def glob(self) -> AsyncIterable[LocalFile]:
        print(f"Ingesting files from {self.path}")
        for path in self.fs.glob("**/*"):
            if self.fs.isdir(path) or (self.filter and not self.filter(LocalFile(path=path, fs=self.fs))):
                continue
            print(f"Ingesting file: {path}")
            yield LocalFile(path=path, fs=self.fs)
