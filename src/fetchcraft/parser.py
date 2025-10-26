from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from .node import Chunk, Node


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
    Parser for text files that splits them into overlapping chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 20,
        separator: str = " ",
        keep_separator: bool = True
    ):
        """
        Initialize the text file parser.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            separator: Character/string to use as split boundaries
            keep_separator: Whether to keep the separator in chunks
        """
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
        Parse text into chunks.
        
        Args:
            text: The text to parse
            metadata: Metadata to attach to chunks
            parent_node: Optional parent node for the chunks
            
        Returns:
            List of Chunk objects with relationships
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = self._split_text(text)
        chunk_nodes = []
        
        for idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
            chunk = Chunk.from_text(
                text=chunk_text,
                chunk_index=idx,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                metadata={**metadata, "total_chunks": len(chunks)}
            )
            
            # Set parent if provided
            if parent_node:
                chunk.parent = parent_node
            
            # Link to previous chunk
            if chunk_nodes:
                chunk.link_to_previous(chunk_nodes[-1])
            
            chunk_nodes.append(chunk)
        
        return chunk_nodes
    
    def _split_text(self, text: str) -> List[tuple[str, int, int]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to split
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at separator
            if end < len(text):
                # Look for the last separator within the chunk
                chunk_text = text[start:end]
                last_sep_idx = chunk_text.rfind(self.separator)
                
                if last_sep_idx != -1 and last_sep_idx > self.overlap:
                    # Adjust end to the separator position
                    if self.keep_separator:
                        end = start + last_sep_idx + len(self.separator)
                    else:
                        end = start + last_sep_idx
            else:
                # Last chunk, take everything
                end = len(text)
            
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = end - self.overlap
            
            # Ensure we make progress
            if start <= chunks[-1][1]:
                start = end
        
        return chunks
    
    def from_file(
        self,
        file_path: Path,
        chunk_size: int = 200,
        overlap: int = 20,
        separator: str = " ",
        keep_separator: bool = True,
        encoding: str | None = None,
        include_file_metadata: bool = True
    ) -> List[Chunk]:
        """
        Parse a text file into chunks.
        
        Args:
            file_path: Path to the text file
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            separator: Character/string to use as split boundaries
            keep_separator: Whether to keep the separator in chunks
            encoding: File encoding
            include_file_metadata: Whether to include file metadata in chunks
            
        Returns:
            List of Chunk objects
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
        
        # Parse into chunks
        return self.parse(
            text=text,
            metadata=metadata,
            parent_node=parent_node
        )
    
    def parse_directory(
        self,
        directory_path: Path,
        pattern: str = "*.txt",
        separator: str = " ",
        keep_separator: bool = True,
        encoding: str | None = None,
        recursive: bool = False
    ) -> Dict[str, List[Chunk]]:
        """
        Parse all text files in a directory.
        
        Args:
            directory_path: Path to the directory
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            pattern: Glob pattern for file matching
            separator: Character/string to use as split boundaries
            keep_separator: Whether to keep the separator in chunks
            encoding: File encoding
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
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    separator=separator,
                    keep_separator=keep_separator,
                    encoding=encoding
                )
                results[str(file_path)] = chunks
        
        return results
