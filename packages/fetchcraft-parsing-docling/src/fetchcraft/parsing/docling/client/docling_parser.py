import tempfile
from typing import Optional, Dict, Any, AsyncGenerator

from pydantic import ConfigDict

from fetchcraft.connector import File
from fetchcraft.node import DocumentNode
from fetchcraft.parsing.base import DocumentParser
from fetchcraft.parsing.docling import AsyncDoclingParserClient
from fetchcraft.parsing.docling.docling_parser import DOCLING_SUPPORTED_EXTENSIONS


class RemoteDoclingParser(DocumentParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    docling_url: str
    client: AsyncDoclingParserClient

    def __init__(self, docling_url: str = "http://localhost:8080"):
        client = AsyncDoclingParserClient(docling_url)
        super().__init__(docling_url=docling_url, client=client)


    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:
        file_suffix = file.path.suffix
        if not file_suffix in DOCLING_SUPPORTED_EXTENSIONS:
            print(f"WARNING: Filetype not supported: {file_suffix}")

        with tempfile.NamedTemporaryFile(delete=True, suffix=file_suffix) as tmp:
            tmp.write(await file.read())
            result = await self.client.parse_single(tmp.name)
            for data in result['nodes']:
                yield DocumentNode.model_validate(data)


    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[DocumentNode, None]:
        raise NotImplemented("Not implemented")
