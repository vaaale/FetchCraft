"""
Example: Using DoclingDocumentSource for advanced document parsing

This example demonstrates:
1. Loading documents with Docling (superior document understanding)
2. Extracting tables and structured content
3. OCR for scanned documents
4. Multi-format support (PDF, DOCX, PPTX, etc.)
5. Indexing and searching document content
"""

import asyncio
import os
from pathlib import Path

from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.parsing.docling import DoclingDocumentParser
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.node_parser import SimpleNodeParser
from qdrant_client import QdrantClient


async def example_single_document():
    """Load and process a single document with Docling."""
    print("=" * 80)
    print("Example 1: Loading a Single Document with Docling")
    print("=" * 80)

    # Use the Crayon annual report from your example
    doc_path = Path("/mnt/storage/data/Finance/Crayon/Crayon_annual-report_2023.pdf")

    if not doc_path.exists():
        print(f"\n⚠️  Document not found: {doc_path}")
        print("Please provide a valid document path.")
        return

    # Create Docling parsing with table extraction
    source = DoclingDocumentParser.from_file(
        doc_path,
        do_table_structure=True,  # Extract table structure
        do_ocr=False  # Set to True for scanned documents
    )

    # Load document
    documents = []
    print("\n📄 Processing document with Docling...")
    async for doc in source.get_documents():
        documents.append(doc)
        print(f"\n✓ Loaded: {doc.metadata['filename']}")
        print(f"  - File size: {doc.metadata['file_size']:,} bytes")
        print(f"  - Text length: {doc.metadata['total_length']:,} characters")
        print(f"  - Tables found: {doc.metadata.get('num_tables', 0)}")

        # Show metadata if available
        if doc.metadata.get('title'):
            print(f"  - Title: {doc.metadata['title']}")
        if doc.metadata.get('author'):
            print(f"  - Author: {doc.metadata['author']}")
        if doc.metadata.get('num_pages'):
            print(f"  - Pages: {doc.metadata['num_pages']}")

        # Show preview
        print(f"\n  Preview (first 300 chars):")
        print(f"  {doc.text[:300]}...")

    return documents


async def example_directory_documents():
    """Load all documents from a directory."""
    print("\n" + "=" * 80)
    print("Example 2: Loading Documents from Directory")
    print("=" * 80)

    doc_dir = Path("/mnt/storage/data/knowledge/diverse")

    if not doc_dir.exists():
        print(f"\n⚠️  Directory not found: {doc_dir}")
        print("Creating example directory...")
        doc_dir.mkdir(exist_ok=True)
        print(f"Please add documents to: {doc_dir.absolute()}")
        return

    # Create Docling parsing for directory
    # Supports: PDF, DOCX, PPTX, XLSX, HTML, and more
    source = DoclingDocumentParser.from_directory(
        directory=doc_dir,
        pattern="*",  # Or use "*" for all formats
        recursive=True,
        page_chunks=True,
        do_table_structure=True
    )

    # Load all documents
    documents = []
    async for doc in source.get_documents():
        documents.append(doc)
        print(f"\n✓ Loaded: {doc.metadata['filename']}")
        print(f"  - Type: {doc.metadata['file_type']}")
        print(f"  - Size: {len(doc.text):,} characters")
        print(f"  - Tables: {doc.metadata.get('num_tables', 0)}")

    print(f"\n📊 Total documents loaded: {len(documents)}")
    return documents


async def example_indexing_and_search():
    """Index document content and perform semantic search."""
    print("\n" + "=" * 80)
    print("Example 5: Indexing and Searching Document Content")
    print("=" * 80)

    doc_path = Path("/mnt/storage/data/Finance/Crayon/Crayon_annual-report_2023.pdf")

    if not doc_path.exists():
        print(f"\n⚠️  Document not found: {doc_path}")
        return

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping indexing example.")
        return

    print("\n1. Loading document with Docling...")
    source = DoclingDocumentParser.from_file(
        doc_path,
        do_table_structure=True
    )

    documents = []
    async for doc in source.get_documents():
        documents.append(doc)

    print(f"   ✓ Loaded {len(documents)} document(s)")

    # Parse into chunks
    print("\n2. Parsing into chunks...")
    parser = SimpleNodeParser(chunk_size=1000, overlap=200)
    nodes = parser.get_nodes(documents)
    print(f"   ✓ Created {len(nodes)} chunks")

    # Create vector index
    print("\n3. Creating vector index...")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key,
        base_url=base_url
    )

    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="docling_docs",
        embeddings=embeddings
    )

    index = VectorIndex(vector_store=vector_store)

    # Index nodes
    print("\n4. Indexing chunks...")
    await index.add_nodes(nodes, show_progress=True)
    print("   ✓ Indexing complete")

    # Search
    print("\n5. Searching indexed content...")
    queries = [
        "What are the key financial highlights?",
        "How many employees does the company have?",
        "What is the revenue growth?"
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = await index.search_by_text(query, k=2)

        for i, (node, score) in enumerate(results, 1):
            print(f"\n   Result {i} (score: {score:.4f}):")
            print(f"   {node.text[:200]}...")



async def main():
    """Run all examples."""
    print("\n🔖 Docling Document Source Examples\n")

    # Example 0: Single document
    await example_directory_documents()

    # Example 1: Single document
    await example_single_document()

    # Example 5: Indexing and search
    await example_indexing_and_search()

    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)

    print("\n📚 Docling Advantages:")
    print("  1. Superior table extraction and formatting")
    print("  2. Advanced layout analysis")
    print("  3. OCR support for scanned documents")
    print("  4. Multi-format support (PDF, DOCX, PPTX, XLSX, HTML)")
    print("  5. Better handling of complex document structures")
    print("  6. Metadata extraction (title, author, creation date)")
    print("  7. Page-level processing with dimensions")
    print("\n💡 Use Cases:")
    print("  - Financial reports with complex tables")
    print("  - Academic papers with figures and equations")
    print("  - Legal documents with structured content")
    print("  - Scanned documents requiring OCR")


if __name__ == "__main__":
    asyncio.run(main())
