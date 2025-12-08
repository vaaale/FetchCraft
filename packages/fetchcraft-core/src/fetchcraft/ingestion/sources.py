"""
Concrete implementations of pipeline sources.

This module provides ready-to-use source implementations that integrate
with connectors and parsers to produce DocumentRecord objects.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import AsyncIterable, Optional

from fetchcraft.connector.base import Connector as BaseConnector
from fetchcraft.ingestion.interfaces import Source, Connector as ConnectorInterface
from fetchcraft.ingestion.models import DocumentRecord

logger = logging.getLogger(__name__)


class ConnectorSource(Source):
    """
    Source that reads files from a connector.
    
    This source uses a Connector to discover files and yields DocumentRecords
    containing file metadata. Parsing is handled by a separate ParsingTransformation.
    
    Attributes:
        connector: The connector to use for reading files
        document_root: Root path for computing relative paths
    """
    
    def __init__(
        self,
        connector: BaseConnector,
        root: Path,
        document_root: Optional[str | Path] = None,
    ):
        """
        Initialize the connector source.
        
        Args:
            connector: The connector to read files from
            document_root: Root path for relative path computation
        """
        self.connector = connector
        self.root = root
        self.document_root = str(document_root) if document_root else ""
        
        logger.debug("ConnectorSource initialized")

    def get_name(self) -> str:
        """
        Get the source name.

        Returns:
            Name of the source (defaults to class name)
        """
        return f"Source({self.connector.get_name()})"

    async def read(self) -> AsyncIterable[DocumentRecord]:
        """
        Read files from connector and yield file records.
        
        Yields:
            DocumentRecord objects containing file metadata
        """
        logger.info("Starting to read files from connector")
        file_count = 0

        async for file in self.connector.glob():
            file_count += 1
            logger.debug(f"Processing file: {file.path}")
            
            # Compute relative path if document_root is set
            relative_source = str((self.document_root / Path(file.path)).relative_to(self.root))

            # Read file content
            try:
                content = await file.read()
                # Base64 encode content for JSON serialization
                content_b64 = base64.b64encode(content).decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading file {file.path}: {e}", exc_info=True)
                continue
            
            # Create document record with file metadata
            doc_record = DocumentRecord(
                source=relative_source,
                content=content_b64,
                metadata={
                    **file.metadata(),
                    "source": relative_source,
                    "path": relative_source,
                    "mimetype": file.mimetype,
                    "encoding": file.encoding,
                }
            )
            
            logger.debug(f"Created file record for {file.path}")
            yield doc_record
        
        logger.info(f"Finished reading from connector: {file_count} files")
    
    def get_connector(self) -> ConnectorInterface:
        """Get the underlying connector."""
        return self.connector  # type: ignore
