"""
Parsing service for document processing business logic.

This module handles the core business logic for parsing documents.
"""

import asyncio
import time
from pathlib import Path
from typing import List

from fetchcraft.connector.filesystem import LocalFile
from ..docling_parser import DoclingDocumentParser
from ..models import ParseResponse


class ParsingService:
    """Service for handling document parsing operations."""
    
    def __init__(
        self,
        page_chunks: bool = True,
        do_ocr: bool = True,
        do_table_structure: bool = True
    ):
        """
        Initialize the parsing service.
        
        Args:
            page_chunks: Split documents into pages
            do_ocr: Enable OCR for scanned documents
            do_table_structure: Extract table structure
        """
        self.page_chunks = page_chunks
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
    
    def parse_file_sync(
        self,
        file_path: Path,
        metadata: dict = None,
        callback_handler=None,
        callback_metadata: dict = None,
        completion_callback_handler=None
    ) -> ParseResponse:
        """
        Synchronously parse a single file.
        
        This runs in a thread pool to avoid blocking the event loop.
        
        Args:
            file_path: Path to the file to parse
            metadata: Optional metadata to include in parsed documents
            callback_handler: Optional async function to call for each node
            callback_metadata: Optional metadata to pass to callback handler
            completion_callback_handler: Optional async function to call on completion/failure
            
        Returns:
            ParseResponse with parsing results
        """
        start_time = time.time()
        filename = file_path.name
        
        try:
            # Create parser - this is CPU-intensive
            parser = DoclingDocumentParser(page_chunks=self.page_chunks, do_ocr=self.do_ocr, do_table_structure=self.do_table_structure)
            # parser = DoclingDocumentParser.from_file(
            #     file_path=file_path,
            #     page_chunks=self.page_chunks,
            #     do_ocr=self.do_ocr,
            #     do_table_structure=self.do_table_structure
            # )
            
            # Parse document synchronously
            # Note: get_documents() is async, but we need to run it sync here
            # We'll use asyncio.run() to run the async generator in this thread
            import asyncio as async_lib
            
            async def collect_nodes():
                collected = []
                node_index = 0
                file = LocalFile.from_path(file_path)
                async for node in parser.parse(file, metadata=metadata or {}):
                    node_dict = node.model_dump()
                    collected.append(node_dict)
                    
                    # Call callback handler if provided
                    if callback_handler and callback_metadata:
                        try:
                            await callback_handler(
                                node_dict=node_dict,
                                node_index=node_index,
                                metadata=callback_metadata
                            )
                        except Exception as e:
                            print(f"⚠️  Callback error for node {node_index}: {e}")
                    
                    node_index += 1
                
                return collected
            
            nodes = async_lib.run(collect_nodes())
            
            processing_time = (time.time() - start_time) * 1000
            
            # Send completion callback if provided
            if completion_callback_handler and callback_metadata:
                import asyncio as async_lib
                try:
                    async_lib.run(completion_callback_handler(
                        success=True,
                        total_nodes=len(nodes),
                        processing_time_ms=round(processing_time, 2),
                        error=None,
                        metadata=callback_metadata
                    ))
                except Exception as e:
                    print(f"⚠️  Completion callback error: {e}")
            
            return ParseResponse(
                filename=filename,
                success=True,
                nodes=nodes,
                error=None,
                num_nodes=len(nodes),
                processing_time_ms=round(processing_time, 2)
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Send failure callback if provided
            if completion_callback_handler and callback_metadata:
                import asyncio as async_lib
                try:
                    async_lib.run(completion_callback_handler(
                        success=False,
                        total_nodes=0,
                        processing_time_ms=round(processing_time, 2),
                        error=error_msg,
                        metadata=callback_metadata
                    ))
                except Exception as callback_error:
                    print(f"⚠️  Failure callback error: {callback_error}")
            
            return ParseResponse(
                filename=filename,
                success=False,
                nodes=[],
                error=error_msg,
                num_nodes=0,
                processing_time_ms=round(processing_time, 2)
            )
    
    async def parse_file_async(
        self,
        file_path: Path,
        executor
    ) -> ParseResponse:
        """
        Asynchronously parse a single file using a thread pool.
        
        Args:
            file_path: Path to the file to parse
            executor: ThreadPoolExecutor to use for parsing
            
        Returns:
            ParseResponse with parsing results
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            self.parse_file_sync,
            file_path
        )
        # Yield control back to event loop
        await asyncio.sleep(0)
        return result
    
    async def parse_files_batch(
        self,
        file_paths: List[Path],
        executor,
        file_semaphore: asyncio.Semaphore
    ) -> List[ParseResponse]:
        """
        Parse multiple files concurrently with semaphore control.
        
        Args:
            file_paths: List of file paths to parse
            executor: ThreadPoolExecutor to use for parsing
            file_semaphore: Semaphore to control concurrent file processing
            
        Returns:
            List of ParseResponse objects
        """
        results = []
        
        for file_path in file_paths:
            async with file_semaphore:
                result = await self.parse_file_async(file_path, executor)
                results.append(result)
        
        return results
