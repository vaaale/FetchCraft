"""
Concrete transformation implementations.

This module provides ready-to-use transformation implementations for
common document processing tasks.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator

import fsspec
from fetchcraft.connector.base import File
from fetchcraft.evaluation import DatasetGenerator
from fetchcraft.ingestion.interfaces import (
    Transformation,
    Record,
    AsyncRemote,
    TransformationResult,
    PostProcessResult,
)
from fetchcraft.node import DocumentNode
from fetchcraft.node_parser import NodeParser
from fetchcraft.parsing.base import DocumentParser
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


# ============================================================================
# File Adapter for Parsers
# ============================================================================

class FileAdapter(File):
    """Adapter to provide file interface from in-memory content."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, path_str: str, content: bytes, mimetype: str, encoding: str, metadata: dict):
        fs = fsspec.filesystem('memory')
        super().__init__(
            path=Path(path_str),
            fs=fs,
            mimetype=mimetype,
            encoding=encoding,
        )
        self._content = content
        self._metadata = {**metadata, "encoding": encoding, "mimetype": mimetype}

    async def read(self) -> bytes:
        return self._content

    def name(self) -> str:
        return self.path.name

    def permissions(self) -> list:
        return []

    def metadata(self) -> dict:
        return self._metadata


class ParsingTransformation(Transformation):
    """
    Unified parsing transformation that handles both local and remote parsers.
    
    This transformation takes Records containing file metadata and content,
    and parses them into structured documents. The behavior (sync vs async)
    is determined by the parser's is_remote property.
    
    For local parsers: Yields Record objects for each parsed document.
    For remote parsers: Returns AsyncRemote to await callback.
    
    Attributes:
        parser_map: Map of mimetype -> parser (use "default" for fallback)
    """

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        parser_map: Optional[Dict[str, DocumentParser]] = None
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
        record: Record,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransformationResult:
        """
        Parse file content into documents.
        
        Args:
            record: Record containing file metadata and content
            correlation_id: Unique identifier for callback correlation
            context: Optional pipeline context
            
        Returns:
            - AsyncRemote for remote parsers (callback-based)
            - AsyncGenerator[Record] for local parsers (yields parsed documents)
            - None if no content or parser found
        """
        # Get file metadata
        mimetype = record.get("mimetype", "")
        file_path = record.get("path", "")
        source = record.get("source", file_path)

        if not record.content:
            raise ValueError(f"No file content in record for {source}")

        # Decode base64 content
        file_content = base64.b64decode(record.content)

        # Select parser based on mimetype
        logger.info(f"Getting parser for {file_path} -> {mimetype}")
        parser = self.parser_map.get(mimetype, self.parser_map.get("default", None))

        if not parser:
            raise ValueError(f"No parser found for file {source} with mimetype {mimetype}")

        # Create file adapter
        file_adapter = FileAdapter(
            path_str=file_path,
            content=file_content,
            mimetype=mimetype,
            encoding=record.get("encoding", "utf-8"),
            metadata=record.metadata()
        )

        # Build metadata to pass to parser
        parser_metadata = {
            "path": file_path,
            **record.metadata()
        }

        # Check if parser is remote (async callback-based)
        if parser.is_remote:
            nodes = await self._parse_remote(correlation_id, file_adapter, parser, parser_metadata)
        else:
            # Local parser - return async generator
            nodes = self._parse_local(parser, file_adapter, parser_metadata, source)

        return nodes

    async def _parse_remote(self, correlation_id: str, file_adapter: FileAdapter, parser: DocumentParser, parser_metadata: dict[str, str]) -> AsyncRemote:
        # Submit to remote parser using correlation_id for callback matching
        # parser.parse() is an async generator, so we iterate to trigger submission
        async for _ in parser.parse(file_adapter, metadata=parser_metadata, task_id=correlation_id):
            pass  # Consume the generator to trigger the job submission

        # Return AsyncRemote with correlation_id for callback correlation
        return AsyncRemote(task_id=correlation_id, metadata=parser_metadata)

    async def _parse_local(
        self,
        parser: DocumentParser,
        file_adapter: FileAdapter,
        parser_metadata: Dict[str, Any],
        source: str,
    ) -> AsyncGenerator[Record, None]:
        """Parse file locally and yield Records for each document."""
        try:
            async for doc in parser.parse(file_adapter, metadata=parser_metadata):
                # Ensure the document has the source in its metadata
                if "source" not in doc.metadata:
                    doc.metadata["source"] = source

                # Create Record from parsed document
                yield Record({
                    "document": doc.model_dump(),
                    "source": source,
                    **parser_metadata,
                })

        except Exception as e:
            logger.error(f"Error parsing file {source}: {e}", exc_info=True)
            raise

    def post_process(self, message: Dict[str, Any]) -> PostProcessResult:
        """
        Post-process callback message from remote parsing service.
        
        Transforms node callback data into a Record.
        """
        msg_type = message.get("type", "")

        if msg_type == "node":
            node_data = message.get("node", {})
            filename = message.get("filename", "")
            node_index = message.get("node_index", 0)
            node_metadata = node_data.get("metadata", {})

            # Create Record from node data
            # Note: "document" must be set AFTER spreading metadata to avoid being overwritten
            record_data = {
                **node_metadata,
                "source": node_metadata.get("source", filename),
                "filename": filename,
                "node_index": node_index,
                "document": node_data,  # Set last to ensure it's not overwritten
            }
            return Record(record_data)

        # For other message types, use default behavior
        return Record(message)

    def get_name(self) -> str:
        """Get transformation name."""
        return "ParsingTransformation"


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
        record: Record,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransformationResult:
        """Extract keywords from document."""
        doc = DocumentNode.model_validate(record["document"])
        text = doc.text
        source = record.get("source", "unknown")

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
        record["keywords"] = keywords

        logger.debug(f"Extracted {len(keywords)} keywords from {source}")
        return record


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
        record: Record,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransformationResult:
        """Create document summary."""
        doc = DocumentNode.model_validate(record["document"])
        text = doc.text
        source = record.get("source", "unknown")

        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        selected = sentences[:self.max_sentences]
        summary = ". ".join(selected)

        # Update document metadata
        doc.metadata['summary'] = summary
        record["document"] = doc.model_dump()
        record["summary"] = summary

        logger.debug(
            f"Created summary of {len(selected)} sentences for {source}"
        )
        return record


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
        record: Record,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransformationResult:
        """Chunk the document."""
        doc = DocumentNode.model_validate(record["document"])
        source = record.get("source", "unknown")

        # Chunk the document
        nodes = self.chunker.get_nodes([doc])

        # Store chunks in record
        record["document"] = doc.model_dump()
        record["chunks"] = [n.model_dump() for n in nodes]

        logger.debug(f"Chunked {source} into {len(nodes)} chunks")
        return record

    def get_name(self) -> str:
        """Get transformation name."""
        return "ChunkingTransformation"


class GenerateQuestionContextPairsTransformation(Transformation):
    """
    Generate question-context pairs from document nodes.

    This transformation takes a document and generates question-context pairs
    using a DatasetGenerator.
    """

    def __init__(
        self,
        num_questions_per_node: int = 3,
        model: str = "gpt-5",
        show_progress: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.num_questions_per_node = num_questions_per_node
        self.generator = DatasetGenerator(model=model)
        self.show_progress = show_progress

    async def process(self, record: Record, correlation_id: str, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        doc = DocumentNode.model_validate(record["document"])

        qa_pairs = await self.generator.from_nodes([doc], questions_per_node=self.num_questions_per_node, show_progress=self.show_progress)

        doc.metadata["questions"] = [q.question for q in qa_pairs]
        record["document"] = doc.model_dump()

        return record
