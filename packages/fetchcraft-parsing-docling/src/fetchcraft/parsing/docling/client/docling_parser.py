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
    timeout: int = 60 * 40
    callback_url: Optional[str] = None
    task_id: Optional[str] = None

    def __init__(
        self, 
        docling_url: str = "http://localhost:8080", 
        timeout: int = 60*40,
        callback_url: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        client = AsyncDoclingParserClient(docling_url, timeout)
        super().__init__(
            docling_url=docling_url, 
            client=client, 
            timeout=timeout,
            callback_url=callback_url,
            task_id=task_id
        )


    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:
        """
        Parse a file using the remote docling server.
        
        In async mode (when callback_url is set):
        - Submits the file to the docling server with a callback URL
        - Yields a single "marker" DocumentNode with the job_id in metadata
        - The actual parsed nodes will arrive via callbacks and be processed separately
        
        In sync mode:
        - Waits for parsing to complete and yields all nodes
        """
        file_suffix = file.path.suffix
        if not file_suffix in DOCLING_SUPPORTED_EXTENSIONS:
            print(f"WARNING: Filetype not supported: {file_suffix}")

        with tempfile.NamedTemporaryFile(delete=True, suffix=file_suffix) as tmp:
            tmp.write(await file.read())
            
            # If callback_url is set, use async mode
            if self.callback_url:
                # Submit job asynchronously with callback and task_id
                response = await self.client.submit_job(
                    tmp.name, 
                    callback_url=self.callback_url,
                    task_id=self.task_id
                )
                job_id = response.get('job_id')
                
                print(f"Submitted docling parsing job: {job_id} (task_id: {self.task_id}) for file {file.path.name}")
                
                # Yield a marker node that indicates async processing
                # This will be used to create a parent document that tracks the job
                marker_node = DocumentNode(
                    text="[Async parsing in progress]",
                    metadata={
                        **file.metadata(),
                        'docling_job_id': job_id,
                        'task_id': self.task_id,
                        'is_parent_document': 'true',
                        'async_parsing': 'true',
                        'filename': file.path.name,
                    }
                )
                yield marker_node
                return
                
            # Synchronous mode - parse and yield nodes immediately
            result = await self.client.parse_single(tmp.name)
            for data in result['nodes']:
                doc = DocumentNode.model_validate(data)
                doc.metadata.update(file.metadata())
                yield doc


    async def get_documents(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[DocumentNode, None]:
        raise NotImplemented("Not implemented")
