"""
Concrete transformation implementations.

This module provides ready-to-use transformation implementations for
common document processing tasks.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Iterable, Dict

from pydantic import ConfigDict
from fetchcraft.connector.base import File
from fetchcraft.ingestion.interfaces import ITransformation
from fetchcraft.ingestion.models import DocumentRecord
from fetchcraft.node import DocumentNode
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser

logger = logging.getLogger(__name__)


class ParsingTransformation(ITransformation):
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
        record: DocumentRecord
    ) -> Optional[DocumentRecord | Iterable[DocumentRecord]]:
        """
        Parse file content into documents.
        
        Args:
            record: DocumentRecord containing file metadata and content
            
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


class ExtractKeywords(ITransformation):
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
        record: DocumentRecord
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


class DocumentSummarization(ITransformation):
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
        record: DocumentRecord
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


class ChunkingTransformation(ITransformation):
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
        record: DocumentRecord
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
