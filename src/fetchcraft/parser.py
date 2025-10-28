from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import Field

from .node import Chunk, Node
from .chunking import ChunkingStrategy, HierarchicalChunkingStrategy, CharacterChunkingStrategy


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    
    Document parsers are responsible for converting raw documents
    into structured chunks that can be indexed and searched.
    """
    
    @abstractmethod
    def parse(self, **kwargs) -> List[Chunk]:
        """
        Parse a document and return a list of chunks.
        
        Returns:
            List of Chunk objects
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_file(cls, file_path: Path, **kwargs) -> List[Chunk]:
        """
        Parse a document from a file path.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional parsing parameters
            
        Returns:
            List of Chunk objects
        """
        pass


class TextFileDocumentParser(DocumentParser):
    """
    Parser for text files that splits them into chunks using a chunking strategy.
    """
    
    def __init__(
        self,
        chunker: Optional[ChunkingStrategy] = Field(default=CharacterChunkingStrategy()),
        chunk_size: int = 4096,
        overlap: int = 200,
        separator: str = " ",
        keep_separator: bool = True
    ):
        """
        Initialize the text file parser.
        
        Args:
            chunker: ChunkingStrategy to use (default: HierarchicalChunkingStrategy)
            chunk_size: Maximum size of each chunk in characters (used if chunker not provided)
            overlap: Number of characters to overlap between chunks (used if chunker not provided)
            separator: Character/string to use as split boundaries (used if chunker not provided)
            keep_separator: Whether to keep the separator in chunks (used if chunker not provided)
        """
        # Use provided chunker or create default HierarchicalChunkingStrategy
        if chunker is None:
            self.chunker = HierarchicalChunkingStrategy(
                chunk_size=chunk_size,
                overlap=overlap,
                separator=separator,
                keep_separator=keep_separator
            )
        else:
            self.chunker = chunker
        
        # Keep these for backward compatibility with parse_directory
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        self.keep_separator = keep_separator
    
    def parse(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_node: Optional[Node] = None
    ) -> List[Chunk]:
        """
        Parse text into chunks using the configured chunking strategy.
        
        Args:
            text: The text to parse
            metadata: Metadata to attach to chunks
            parent_node: Optional parent node for the chunks
            
        Returns:
            List of Chunk objects (and SymNodes if hierarchical strategy)
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Use the chunking strategy to split the text
        return self.chunker.chunk_text(
            text=text,
            metadata=metadata,
            parent_node=parent_node
        )
    
    def from_file(
        self,
        file_path: Path,
        encoding: str | None = None,
        include_file_metadata: bool = True
    ) -> List[Chunk]:
        """
        Parse a text file into chunks using the configured chunking strategy.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding (auto-detect if None)
            include_file_metadata: Whether to include file metadata in chunks
            
        Returns:
            List of Chunk objects (and SymNodes if hierarchical strategy)
        """
        # Read the file
        text = None
        if not encoding:
            for encoding in ["utf-8", "iso8859-1"]:
                try:
                    text = file_path.read_text(encoding=encoding)
                    break
                except Exception as e:
                    continue
            else:
                raise ValueError(f"Failed to read file {file_path} with any encoding")
        else:
            text = file_path.read_text(encoding=encoding)

        # Prepare metadata
        metadata = {}
        if include_file_metadata:
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
            }
        
        # Create parent document node
        parent_node = Node(
            text=text,
            metadata={
                **metadata,
                "type": "document",
                "total_length": len(text)
            }
        )
        
        # Parse into chunks using the chunking strategy
        return self.parse(
            text=text,
            metadata=metadata,
            parent_node=parent_node
        )
    
    def parse_directory(
        self,
        directory_path: Path,
        pattern: str = "*.txt",
        encoding: str | None = None,
        recursive: bool = False
    ) -> Dict[str, List[Chunk]]:
        """
        Parse all files in a directory using the configured chunking strategy.
        
        Args:
            directory_path: Path to the directory
            pattern: Glob pattern for file matching (default: "*.txt")
            encoding: File encoding (auto-detect if None)
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary mapping file paths to their chunk lists
        """
        if not directory_path.is_dir():
            raise ValueError(f"{directory_path} is not a directory")
        
        results = {}
        
        # Get files matching pattern
        if recursive:
            files = directory_path.rglob(pattern)
        else:
            files = directory_path.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                chunks = self.from_file(
                    file_path=file_path,
                    encoding=encoding
                )
                results[str(file_path)] = chunks
        
        return results
