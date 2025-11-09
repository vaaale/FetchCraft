from typing import Optional, Dict, Any, AsyncGenerator

from fetchcraft.connector import File
from fetchcraft.node import DocumentNode
from fetchcraft.parsing.base import DocumentParser


class TextFileParser(DocumentParser):


    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:

        def _decode(data: bytes) -> str:
            for encoding in ["utf-8", "iso-8859-1"]:
                try:
                    return data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Failed to decode file with any encoding")

        data = await file.read()
        text = _decode(data)
        if not metadata:
            metadata = file.metadata()
        else:
            metadata = {**file.metadata(), **metadata}
        yield DocumentNode(
            text=text,
            metadata=metadata
        )

    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[DocumentNode, None]:
        raise NotImplemented("Not implemented")