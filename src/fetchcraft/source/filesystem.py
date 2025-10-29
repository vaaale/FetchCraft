from pathlib import Path
from typing import *

import fsspec

from fetchcraft.node import DocumentNode
from fetchcraft.source.base import DocumentSource


class FilesystemDocumentSource(DocumentSource):
    fs: Optional[fsspec.AbstractFileSystem] = None
    pattern: str
    recursive: bool = True
    directory: Path
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:
        # Get files matching pattern

        if self.recursive:
            if self.fs:
                glob_path = f"{self.directory}/{self.pattern}" if self.pattern.endswith("**") else f"{self.directory}/**/{self.pattern}"
                files = self.fs.glob(glob_path)
            else:
                files = self.directory.rglob(self.pattern)
        else:
            if self.fs:
                glob_path = f"{self.directory}/{self.pattern}"
                files = self.fs.glob(glob_path)
            else:
                files = self.directory.glob(self.pattern)

        for file_path in files:
            # Convert to Path if it's a string (from fsspec)
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # Check if it's a file (skip directories)
            if self.fs:
                # For fsspec, we need to check differently
                try:
                    if self.fs.isfile(str(file_path)):
                        yield await self._read_file(file_path, metadata)
                except Exception:
                    continue
            else:
                if file_path.is_file():
                    yield await self._read_file(file_path, metadata)



    async def _read_file(self, file_path: Path, metadata: Optional[Dict[str, Any]] = None) -> DocumentNode:
        text = None
        for encoding in ["utf-8", "iso8859-1"]:
            try:
                if self.fs:
                    text = self.fs.read_text(str(file_path), encoding=encoding)
                else:
                    text = file_path.read_text(encoding=encoding)
                break
            except Exception as e:
                continue
        else:
            raise ValueError(f"Failed to read file {file_path} with any encoding")

        # Prepare metadata
        if not metadata:
            metadata = {}

        # Get file size
        if self.fs:
            file_size = self.fs.size(str(file_path))
        else:
            file_size = file_path.stat().st_size
        
        metadata = {
            **metadata,
            "source": str(file_path),
            "filename": file_path.name,
            "file_size": file_size,
            "total_length": len(text)
        }

        # Create parent document node
        document = DocumentNode(
            text=text,
            metadata=metadata
        )
        return document


    @classmethod
    def from_fs(cls, fs: fsspec.AbstractFileSystem, directory: Path, pattern: str = "*", recursive: bool = True):
        return cls(fs=fs, directory=directory, pattern=pattern, recursive=recursive)

    @classmethod
    def from_directory(cls, directory: Path, pattern: str = "*", recursive: bool = True):
        return cls(fs=None, directory=directory, pattern=pattern, recursive=recursive)

    @classmethod
    def from_file(cls, file_path: Path):
        return cls(fs=None, directory=file_path.parent, pattern=file_path.name, recursive=False)
