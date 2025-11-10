"""
Document Source using Docling for advanced document parsing.

This module provides a DocumentSource implementation for extracting text
from various document formats using Docling, which offers superior
document understanding including layout analysis, table extraction, and more.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List

from huggingface_hub import snapshot_download

from fetchcraft.connector import File

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from fetchcraft.node import DocumentNode
from fetchcraft.parsing.base import DocumentParser
from collections import defaultdict


# Supported file formats by Docling
DOCLING_SUPPORTED_EXTENSIONS = {
    '.pdf',      # PDF documents
    '.docx',     # Microsoft Word
    '.pptx',     # Microsoft PowerPoint
    '.html',     # HTML documents
    '.htm',      # HTML documents
    '.xlsx',     # Microsoft Excel
    '.png',      # Images (with OCR)
    '.jpg',      # Images (with OCR)
    '.jpeg',     # Images (with OCR)
    '.tiff',     # Images (with OCR)
    '.bmp',      # Images (with OCR)
    '.asciidoc', # AsciiDoc
    '.adoc',     # AsciiDoc
    '.md',       # Markdown
}


class DoclingDocumentParser(DocumentParser):
    """
    Document parsing for parsing documents using Docling.
    
    Docling provides advanced document understanding with:
    - Superior table extraction and formatting
    - Layout analysis and structure preservation
    - Support for multiple formats (PDF, DOCX, PPTX, HTML, XLSX, Images, AsciiDoc, Markdown)
    - OCR capabilities for scanned documents and images
    - Metadata extraction (titles, authors, etc.)
    
    Supported file formats:
    - PDF (.pdf)
    - Microsoft Word (.docx)
    - Microsoft PowerPoint (.pptx)
    - Microsoft Excel (.xlsx)
    - HTML (.html, .htm)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp) - with OCR
    - AsciiDoc (.asciidoc, .adoc)
    - Markdown (.md)
    
    Example:
        ```python
        # Load single PDF file
        parsing = DoclingDocumentSource.from_file(Path("document.pdf"))
        
        # Load all supported documents from directory
        parsing = DoclingDocumentSource.from_directory(
            Path("./docs"), 
            pattern="*",  # Will auto-filter to supported formats
            recursive=True
        )
        
        # Load only specific format
        parsing = DoclingDocumentSource.from_directory(
            Path("./docs"),
            pattern="*.pdf",
            recursive=True
        )
        
        # Configure with OCR for scanned documents
        parsing = DoclingDocumentSource.from_file(
            Path("scanned.pdf"),
            do_ocr=True
        )
        
        # Get documents
        async for doc in parsing.get_documents():
            print(f"Loaded: {doc.metadata['filename']}")
            print(f"Tables found: {doc.metadata.get('num_tables', 0)}")
        ```
    """

    directory: Path
    pattern: str = "*"
    recursive: bool = True
    page_chunks: bool = True  # If True, return each page as separate document
    do_ocr: bool = True  # Enable OCR for scanned documents
    do_table_structure: bool = True  # Extract table structure
    filter_supported_only: bool = True  # If True, only process supported file formats
    ocr_options: Optional[RapidOcrOptions] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }
    
    def __init__(self, **data):
        """Initialize DoclingDocumentSource."""
        super().__init__(**data)
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "docling is not installed. "
                "Install it with: pip install docling"
            )

        # print("Downloading RapidOCR models")
        # download_path = snapshot_download(repo_id="RapidAI/RapidOCR")
        #
        # # Setup RapidOcrOptions for English detection
        # det_model_path = os.path.join(
        #     download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_server_det.onnx"
        # )
        # rec_model_path = os.path.join(
        #     download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
        # )
        # cls_model_path = os.path.join(
        #     download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        # )
        # self.ocr_options = RapidOcrOptions(
        #     det_model_path=det_model_path,
        #     rec_model_path=rec_model_path,
        #     cls_model_path=cls_model_path,
        # )

    async def parse(self, file: File, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DocumentNode, None]:
        raise NotImplementedError("Not implemented")


    async def get_documents(
        self, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[DocumentNode, None]:
        """
        Read documents from the configured directory.
        
        Args:
            metadata: Additional metadata to add to each document
            **kwargs: Additional arguments passed to Docling converter
            
        Yields:
            DocumentNode objects for each document or page
        """
        # Get files matching pattern
        if self.recursive:
            files = self.directory.rglob(self.pattern)
        else:
            files = self.directory.glob(self.pattern)
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Filter by supported extensions if enabled
            if self.filter_supported_only:
                file_ext = file_path.suffix.lower()
                if file_ext not in DOCLING_SUPPORTED_EXTENSIONS:
                    continue
            
            # Yield one or more documents from this file
            try:
                async for doc in self._read_document(file_path, metadata, **kwargs):
                    yield doc
            except Exception as e:
                # Log error but continue processing other files
                print(f"Warning: Failed to process {file_path}: {str(e)}")
                continue
    
    async def _read_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[DocumentNode, None]:
        """
        Read a single document file and extract content.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata
            **kwargs: Additional arguments for Docling converter
            
        Yields:
            DocumentNode for the entire document or each page
        """
        try:
            # Configure pipeline options
            # Download RapidOCR models from Hugging Face
            # if self.ocr_options:
            #     pipeline_options = PdfPipelineOptions(
            #         ocr_options=self.ocr_options,
            #     )
            # else:
            #     pipeline_options = PdfPipelineOptions()

            pipeline_options = PdfPipelineOptions()

            # pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.do_ocr
            pipeline_options.do_table_structure = self.do_table_structure
            
            # Create converter
            converter = DocumentConverter()
            
            # Convert document
            result = converter.convert(str(file_path))
            
            # Prepare base metadata
            base_metadata = {
                "parsing": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix[1:],  # Remove the dot
            }
            
            if metadata:
                base_metadata.update(metadata)

            page_table = defaultdict(list)
            for table in result.document.tables:
                for prov in table.prov:
                    page_table[prov.page_no].append(table)

            if self.page_chunks and hasattr(result.document, 'pages'):
                # Return each page as separate document
                pages = result.document.pages

                for page_num, page in enumerate(pages, start=1):
                    page_text = result.document.export_to_markdown(page_no=page_num)
                    tables = page_table.get(page_num, [])

                    table_dicts = [table.export_to_dataframe().to_dict(orient='records') for table in tables]
                    table_json = json.dumps(table_dicts, indent=2)
                    table_list_md = "\n\n".join([table.export_to_markdown() for table in tables])

                    if page_text.strip() or tables:  # Skip empty pages
                        page_metadata = {
                            **base_metadata,
                            "page_number": page_num,
                            "total_pages": len(pages),
                            "total_length": len(page_text),
                            "tables_json": table_json,
                            "tables_md": table_list_md
                        }

                        # Add page-specific metadata
                        if hasattr(page, 'size'):
                            page_metadata["width"] = page.size.width
                            page_metadata["height"] = page.size.height

                        if not page_text:
                            page_text = table_list_md

                        yield DocumentNode(
                            text=page_text,
                            metadata=page_metadata
                        )
            else:
                # Return entire document as single DocumentNode
                # Get markdown representation
                md_text = result.document.export_to_markdown()

                doc_metadata = {
                    **base_metadata,
                    "total_length": len(md_text)
                }
                
                yield DocumentNode(
                    text=md_text,
                    metadata=doc_metadata
                )
                
        except Exception as e:
            raise ValueError(f"Failed to parse document {file_path}: {str(e)}") from e
    
    @classmethod
    def from_file(
        cls, 
        file_path: Path, 
        page_chunks: bool = True,
        do_ocr: bool = True,
        do_table_structure: bool = True
    ) -> "DoclingDocumentParser":
        """
        Create a DoclingDocumentSource for a single document file.
        
        Supports all Docling-compatible formats:
        PDF, DOCX, PPTX, XLSX, HTML, Images (PNG, JPG, etc.), AsciiDoc, Markdown
        
        Args:
            file_path: Path to the document file
            page_chunks: If True, split into separate documents per page
            do_ocr: Enable OCR for scanned documents and images
            do_table_structure: Extract table structure
            
        Returns:
            DoclingDocumentSource instance
        """
        return cls(
            directory=file_path.parent,
            pattern=file_path.name,
            recursive=False,
            page_chunks=page_chunks,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            filter_supported_only=False  # Single file, no filtering needed
        )
    
    @classmethod
    def from_directory(
        cls,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
        page_chunks: bool = True,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        filter_supported_only: bool = True
    ) -> "DoclingDocumentParser":
        """
        Create a DoclingDocumentSource for a directory of documents.
        
        Automatically processes all Docling-supported file formats:
        - PDF (.pdf)
        - Microsoft Office (.docx, .pptx, .xlsx)
        - HTML (.html, .htm)
        - Images (.png, .jpg, .jpeg, .tiff, .bmp)
        - AsciiDoc (.asciidoc, .adoc)
        - Markdown (.md)
        
        Args:
            directory: Directory containing document files
            pattern: Glob pattern for matching files (default: "*" for all files)
            recursive: If True, search subdirectories
            page_chunks: If True, split documents into separate documents per page
            do_ocr: Enable OCR for scanned documents and images
            do_table_structure: Extract table structure
            filter_supported_only: If True, only process Docling-supported formats (default: True)
            
        Returns:
            DoclingDocumentSource instance
        """
        return cls(
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            page_chunks=page_chunks,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            filter_supported_only=filter_supported_only
        )
