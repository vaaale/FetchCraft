"""
PDF Document Source using pymupdf4llm for parsing PDF files.

This module provides a DocumentSource implementation for extracting text
from PDF files with markdown formatting using pymupdf4llm.
"""

from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

from fetchcraft.node import DocumentNode
from fetchcraft.source.base import DocumentSource


class PDFDocumentSource(DocumentSource):
    """
    Document source for parsing PDF files using pymupdf4llm.
    
    This source extracts text from PDFs with markdown formatting,
    preserving document structure, tables, and other formatting.
    
    Example:
        ```python
        # Load single PDF file
        source = PDFDocumentSource.from_file(Path("document.pdf"))
        
        # Load all PDFs from directory
        source = PDFDocumentSource.from_directory(
            Path("./pdfs"), 
            recursive=True
        )
        
        # Get documents
        async for doc in source.get_documents():
            print(f"Loaded: {doc.metadata['filename']}")
        ```
    """
    
    directory: Path
    pattern: str = "*.pdf"
    recursive: bool = True
    page_chunks: bool = False  # If True, return each page as separate document
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    def __init__(self, **data):
        """Initialize PDFDocumentSource."""
        super().__init__(**data)
        if not PYMUPDF4LLM_AVAILABLE:
            raise ImportError(
                "pymupdf4llm is not installed. "
                "Install it with: pip install pymupdf4llm"
            )
    
    async def get_documents(
        self, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[DocumentNode, None]:
        """
        Read PDF documents from the configured directory.
        
        Args:
            metadata: Additional metadata to add to each document
            
        Yields:
            DocumentNode objects for each PDF file or page
        """
        # Get PDF files matching pattern
        if self.recursive:
            files = self.directory.rglob(self.pattern)
        else:
            files = self.directory.glob(self.pattern)
        
        for file_path in files:
            if file_path.is_file():
                # Yield one or more documents from this PDF
                async for doc in self._read_pdf(file_path, metadata, **kwargs):
                    yield doc
    
    async def _read_pdf(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None,
        page_chunks=True,
        **kwargs
    ) -> AsyncGenerator[DocumentNode, None]:
        """
        Read a single PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            metadata: Additional metadata
            
        Yields:
            DocumentNode for the entire PDF or each page
        """
        try:
            # Extract text with markdown formatting
            md_text = pymupdf4llm.to_markdown(
                str(file_path),
                page_chunks=page_chunks,
                **kwargs
            )


            # Prepare base metadata
            base_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": "pdf",
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            if self.page_chunks:
                # Split by page and create separate documents
                # pymupdf4llm includes page markers like "-----\n\n"
                pages = md_text.split("-----\n\n")
                
                for page_num, page_text in enumerate(pages, start=1):
                    if page_text.strip():  # Skip empty pages
                        page_metadata = {
                            **base_metadata,
                            "page_number": page_num,
                            "total_pages": len(pages),
                            "total_length": len(page_text)
                        }
                        
                        yield DocumentNode(
                            text=page_text.strip(),
                            metadata=page_metadata
                        )
            else:
                # Return entire PDF as single document
                doc_metadata = {
                    **base_metadata,
                    "total_length": len(md_text)
                }
                
                yield DocumentNode(
                    text=md_text,
                    metadata=doc_metadata
                )
                
        except Exception as e:
            raise ValueError(
                f"Failed to parse PDF {file_path}: {str(e)}"
            ) from e
    
    @classmethod
    def from_file(
        cls, 
        file_path: Path, 
        page_chunks: bool = True
    ) -> "PDFDocumentSource":
        """
        Create a PDFDocumentSource for a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            page_chunks: If True, split into separate documents per page
            
        Returns:
            PDFDocumentSource instance
        """
        return cls(
            directory=file_path.parent,
            pattern=file_path.name,
            recursive=False,
            page_chunks=page_chunks
        )
    
    @classmethod
    def from_directory(
        cls,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
        page_chunks: bool = True
    ) -> "PDFDocumentSource":
        """
        Create a PDFDocumentSource for a directory of PDF files.
        
        Args:
            directory: Directory containing PDF files
            pattern: Glob pattern for matching files (default: "*.pdf")
            recursive: If True, search subdirectories
            page_chunks: If True, split PDFs into separate documents per page
            
        Returns:
            PDFDocumentSource instance
        """
        return cls(
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            page_chunks=page_chunks
        )
