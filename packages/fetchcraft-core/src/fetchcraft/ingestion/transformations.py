"""
Concrete transformation implementations.

This module provides ready-to-use transformation implementations for
common document processing tasks.
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Optional, Iterable, Dict, Any, Literal, List

from pydantic import ConfigDict
from fetchcraft.connector.base import File
from fetchcraft.ingestion.interfaces import Transformation, AsyncTransformation
from fetchcraft.ingestion.models import DocumentRecord, DocumentStatus
from fetchcraft.node import DocumentNode
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser

logger = logging.getLogger(__name__)


class ParsingTransformation(Transformation):
    """
    Parse file content into documents using DocumentParsers.
    
    This transformation takes DocumentRecords containing file metadata
    and content, and parses them into structured documents using the
    appropriate parser for each file type.
    
    Attributes:
        parser_map: Map of mimetype -> parser (use "default" for fallback)
    """
    
    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        parser_map: Optional[Dict[str, DocumentParser]] = None,
    ):
        """
        Initialize the parsing transformation.
        
        Args:
            parser: Default parser (added to parser_map as "default")
            parser_map: Map of mimetype to parser
        """
        self.parser_map = parser_map or {}
        
        if parser and "default" not in self.parser_map:
            self.parser_map["default"] = parser
        
        logger.debug(f"ParsingTransformation initialized with {len(self.parser_map)} parsers")
    
    async def process(
        self,
        record: DocumentRecord,
        context: Optional[dict] = None
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Parse file content into documents.
        
        Args:
            record: DocumentRecord containing file metadata and content
            context: Optional pipeline context
            
        Returns:
            One or more DocumentRecords with parsed document data
        """
        # Get file metadata
        mimetype = record.metadata.get("mimetype", "")
        file_path = record.metadata.get("file_path", "")
        file_content_b64 = record.metadata.get("file_content_b64")
        
        if not file_content_b64:
            logger.warning(f"No file content in record for {record.source}, skipping")
            return None
        
        # Decode base64 content
        try:
            file_content = base64.b64decode(file_content_b64)
        except Exception as e:
            logger.error(f"Error decoding file content for {record.source}: {e}", exc_info=True)
            return None
        
        # Select parser based on mimetype
        parser = self.parser_map.get(mimetype, self.parser_map.get("default"))
        
        if not parser:
            logger.warning(
                f"No parser found for file {record.source} "
                f"with mimetype {mimetype}, skipping"
            )
            return None
        
        try:
            # Create a temporary file-like object for the parser
            # Most parsers expect a File object
            from pathlib import Path
            import fsspec
            
            # Parse the content
            # Note: Parsers typically expect a File object, so we need to adapt
            class FileAdapter(File):
                """Adapter to provide file interface from content."""
                model_config = ConfigDict(arbitrary_types_allowed=True)
                
                def __init__(self, path_str: str, content: bytes, mimetype: str, encoding: str):
                    # Create a memory filesystem for the adapter
                    fs = fsspec.filesystem('memory')
                    super().__init__(
                        path=Path(path_str),
                        fs=fs,
                        mimetype=mimetype,
                        encoding=encoding,
                    )
                    self._content = content
                
                async def read(self) -> bytes:
                    return self._content
                
                def name(self) -> str:
                    return self.path.name
                
                def permissions(self) -> list:
                    # Return empty permissions for memory-based file
                    return []
                
                def metadata(self) -> dict:
                    # Return basic metadata
                    return {
                        "path": str(self.path),
                        "mimetype": self.mimetype,
                        "encoding": self.encoding,
                        "size": len(self._content)
                    }
            
            file_adapter = FileAdapter(
                path_str=file_path,
                content=file_content,  # Already decoded bytes
                mimetype=mimetype,
                encoding=record.metadata.get("encoding", "utf-8")
            )
            
            # Parse file into documents
            documents = parser.parse(file_adapter)
            
            # Collect all parsed documents
            parsed_docs = []
            async for doc in documents:
                # Ensure the document has the source in its metadata
                if "source" not in doc.metadata:
                    doc.metadata["source"] = record.source
                parsed_docs.append(doc)
            
            if not parsed_docs:
                logger.warning(f"No documents parsed from {record.source}")
                return None
            
            # Use the first document (or merge if multiple)
            # Most files parse to a single document; if multiple, we take the first
            primary_doc = parsed_docs[0]
            
            # Check if this is an async parsing marker document
            if primary_doc.metadata.get('async_parsing') == 'true':
                # This is a parent document for async parsing
                # Store the marker doc and metadata, then return the record
                # The parent document should NOT continue through the pipeline
                # It will be marked as completed when the completion callback arrives
                record.metadata["document"] = primary_doc.model_dump()
                record.metadata["file_path"] = file_path
                record.metadata["mimetype"] = mimetype
                
                # Add job tracking metadata
                record.metadata["docling_job_id"] = primary_doc.metadata.get('docling_job_id')
                record.metadata["is_parent_document"] = 'true'
                record.metadata["nodes_received_count"] = 0
                
                logger.info(
                    f"Async parsing initiated for {record.source}, "
                    f"job_id={primary_doc.metadata.get('docling_job_id')}"
                )
                return record
            
            # If there are multiple documents, log this
            if len(parsed_docs) > 1:
                logger.info(
                    f"Parser produced {len(parsed_docs)} documents from {record.source}, "
                    f"using the first one"
                )
            
            # Update the record with parsed document
            # Note: We don't include file_content_b64 here as it's no longer needed
            record.metadata["document"] = primary_doc.model_dump()
            record.metadata["file_path"] = file_path
            record.metadata["mimetype"] = mimetype
            
            logger.debug(f"Parsed document from {record.source}")
            return record
                
        except Exception as e:
            logger.error(f"Error parsing file {record.source}: {e}", exc_info=True)
            raise
    
    def get_name(self) -> str:
        """Get transformation name."""
        return "ParsingTransformation"


class AsyncParsingTransformation(AsyncTransformation):
    """
    Async parsing transformation for remote document parsing services.
    
    This transformation submits documents to an external parsing service
    (e.g., docling) and handles callbacks as nodes are parsed.
    
    The callback message format expected:
    - status: "PROCESSING" with message.type: "node" - contains a parsed node
    - status: "COMPLETED" with message.type: "completion" - job finished
    - status: "FAILED" - job failed with error
    
    Attributes:
        parser_map: Map of mimetype -> parser (parsers must support async callbacks)
    """
    
    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        parser_map: Optional[Dict[str, DocumentParser]] = None,
    ):
        """
        Initialize the async parsing transformation.
        
        Args:
            parser: Default parser (added to parser_map as "default")
            parser_map: Map of mimetype to parser
        """
        self.parser_map = parser_map or {}
        
        if parser and "default" not in self.parser_map:
            self.parser_map["default"] = parser
        
        # Track accumulated nodes per task
        self._task_nodes: Dict[str, List[Dict[str, Any]]] = {}
        self._task_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"AsyncParsingTransformation initialized with {len(self.parser_map)} parsers")
    
    async def submit(
        self,
        record: DocumentRecord,
        task_id: str,
        callback_url: str
    ) -> None:
        """
        Submit document to external parsing service.
        
        Args:
            record: The document record to process
            task_id: Unique task identifier for correlation
            callback_url: URL where callbacks should be sent
        """
        # Get file metadata
        mimetype = record.metadata.get("mimetype", "")
        file_path = record.metadata.get("file_path", "")
        file_content_b64 = record.metadata.get("file_content_b64")
        
        if not file_content_b64:
            raise ValueError(f"No file content in record for {record.source}")
        
        # Decode base64 content
        file_content = base64.b64decode(file_content_b64)
        
        # Select parser based on mimetype
        parser = self.parser_map.get(mimetype, self.parser_map.get("default"))
        
        if not parser:
            raise ValueError(
                f"No parser found for file {record.source} with mimetype {mimetype}"
            )
        
        # Create file adapter
        from pathlib import Path
        import fsspec
        
        class FileAdapter(File):
            """Adapter to provide file interface from content."""
            model_config = ConfigDict(arbitrary_types_allowed=True)
            
            def __init__(self, path_str: str, content: bytes, mimetype: str, encoding: str):
                fs = fsspec.filesystem('memory')
                super().__init__(
                    path=Path(path_str),
                    fs=fs,
                    mimetype=mimetype,
                    encoding=encoding,
                )
                self._content = content
            
            async def read(self) -> bytes:
                return self._content
            
            def name(self) -> str:
                return self.path.name
            
            def permissions(self) -> list:
                return []
            
            def metadata(self) -> dict:
                return {
                    "path": str(self.path),
                    "mimetype": self.mimetype,
                    "encoding": self.encoding,
                    "size": len(self._content)
                }
        
        file_adapter = FileAdapter(
            path_str=file_path,
            content=file_content,
            mimetype=mimetype,
            encoding=record.metadata.get("encoding", "utf-8")
        )
        
        # Initialize task tracking
        self._task_nodes[task_id] = []
        self._task_metadata[task_id] = {
            "source": record.source,
            "file_path": file_path,
            "mimetype": mimetype,
            "document_id": record.id,
            "job_id": record.job_id,
        }
        
        # Submit to parser - this triggers the async job submission
        # The parser will yield a marker node and return
        async for _ in parser.parse(file_adapter, task_id=task_id):
            pass  # We don't need the marker node here
        
        logger.info(
            f"Submitted {record.source} to async parsing service "
            f"(task_id: {task_id}, callback: {callback_url})"
        )
    
    async def on_message(
        self,
        message: Dict[str, Any],
        status: Literal['PROCESSING', 'COMPLETED', 'FAILED']
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Handle callback message from parsing service.
        
        Args:
            message: Callback payload containing node data or completion info
            status: Callback status
            
        Returns:
            - If COMPLETED: List of DocumentRecords for each parsed node
            - If PROCESSING: None (accumulate nodes)
            - If FAILED: Raises exception
        """
        # Extract task_id from the message context
        # Note: task_id is passed separately to handle_task_callback, 
        # but we need to track which task this message belongs to
        msg_type = message.get("type", "")
        
        if status == "PROCESSING" and msg_type == "node":
            # Accumulate node data
            node_data = message.get("node", {})
            filename = message.get("filename", "")
            node_index = message.get("node_index", 0)
            
            logger.debug(
                f"Received node {node_index} from file {filename}"
            )
            
            # We can't easily track by task_id here since it's not in message
            # The pipeline handles task correlation
            return None
        
        elif status == "COMPLETED" and msg_type == "completion":
            # Parsing completed
            total_nodes = message.get("total_nodes", 0)
            total_files = message.get("total_files", 0)
            processing_time = message.get("processing_time_ms", 0)
            
            logger.info(
                f"Async parsing completed: {total_nodes} nodes from {total_files} files "
                f"in {processing_time}ms"
            )
            
            # The actual node documents are created via PROCESSING callbacks
            # Return None to indicate completion without new documents
            return None
        
        elif status == "FAILED":
            error = message.get("error", "Unknown parsing error")
            raise RuntimeError(f"Async parsing failed: {error}")
        
        else:
            logger.warning(f"Unknown message type: {msg_type} with status: {status}")
            return None
    
    def get_name(self) -> str:
        """Get transformation name."""
        return "AsyncParsingTransformation"


class ExtractKeywords(Transformation):
    """
    Extract keywords from document text.
    
    This is a simple keyword extraction based on word frequency.
    """
    
    def __init__(self, max_keywords: int = 10, min_word_length: int = 4):
        """
        Initialize keyword extractor.
        
        Args:
            max_keywords: Maximum number of keywords to extract
            min_word_length: Minimum word length to consider
        """
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length
    
    async def process(
        self,
        record: DocumentRecord,
        context: Optional[dict] = None
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """Extract keywords from document."""
        try:
            doc = DocumentNode.model_validate(record.metadata["document"])
            text = doc.text
            
            # Simple word frequency analysis
            words = [w.strip(".,!?;:\"'()[]{}-").lower() for w in text.split()]
            counts: dict[str, int] = {}
            
            for w in words:
                if len(w) >= self.min_word_length:
                    counts[w] = counts.get(w, 0) + 1
            
            # Get top keywords
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            top = top[:self.max_keywords]
            
            keywords = [k for k, _ in top]
            record.metadata["keywords"] = keywords
            
            logger.debug(f"Extracted {len(keywords)} keywords from {record.source}")
            return record
            
        except Exception as e:
            logger.error(f"Error extracting keywords from {record.source}: {e}")
            raise


class DocumentSummarization(Transformation):
    """
    Create a simple extractive summary of document text.
    
    This uses a basic sentence extraction approach. For production,
    consider using a more sophisticated summarization model.
    """
    
    def __init__(self, max_sentences: int = 30):
        """
        Initialize summarization.
        
        Args:
            max_sentences: Maximum number of sentences in summary
        """
        self.max_sentences = max_sentences
    
    async def process(
        self,
        record: DocumentRecord,
        context: Optional[dict] = None
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """Create document summary."""
        try:
            doc = DocumentNode.model_validate(record.metadata["document"])
            text = doc.text
            
            # Split into sentences
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            selected = sentences[:self.max_sentences]
            summary = ". ".join(selected)
            
            # Update document metadata
            doc.metadata['summary'] = summary
            record.metadata["document"] = doc.model_dump()
            record.metadata["summary"] = summary
            
            logger.debug(
                f"Created summary of {len(selected)} sentences for {record.source}"
            )
            return record
            
        except Exception as e:
            logger.error(f"Error summarizing {record.source}: {e}")
            raise


class ChunkingTransformation(Transformation):
    """
    Split documents into chunks using a NodeParser.
    
    This transformation takes a document and splits it into smaller
    chunks suitable for embedding and retrieval.
    """
    
    def __init__(self, chunker: NodeParser):
        """
        Initialize chunking transformation.
        
        Args:
            chunker: The NodeParser to use for chunking
        """
        self.chunker = chunker
    
    async def process(
        self,
        record: DocumentRecord,
        context: Optional[dict] = None
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """Chunk the document."""
        try:
            doc = DocumentNode.model_validate(record.metadata["document"])
            
            # Chunk the document
            nodes = self.chunker.get_nodes([doc])
            
            # Store chunks in record
            record.metadata["document"] = doc.model_dump()
            record.metadata["chunks"] = [n.model_dump() for n in nodes]
            
            logger.debug(f"Chunked {record.source} into {len(nodes)} chunks")
            return record
            
        except Exception as e:
            logger.error(f"Error chunking {record.source}: {e}")
            raise
    
    def get_name(self) -> str:
        """Get transformation name."""
        return "ChunkingTransformation"
