"""
Example: Using PDFDocumentSource to parse PDF files

This example demonstrates:
1. Loading a single PDF file
2. Loading all PDFs from a directory
3. Splitting PDFs by page
4. Indexing PDF content for search
"""

import asyncio
import os
from pathlib import Path

from fetchcraft.parsing import PDFDocumentParser
from fetchcraft.embeddings import OpenAIEmbeddings
from fetchcraft.index.vector_index import VectorIndex
from fetchcraft.vector_store import QdrantVectorStore
from fetchcraft.node_parser import SimpleNodeParser
from qdrant_client import QdrantClient


async def example_single_pdf():
    """Load and process a single PDF file."""
    print("=" * 80)
    print("Example 1: Loading a Single PDF")
    print("=" * 80)
    
    pdf_path = Path("/mnt/storage/data/Finance/Crayon/Crayon_annual-report_2023.pdf")  # Replace with your PDF path
    
    if not pdf_path.exists():
        print(f"\n⚠️  PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")
        return
    
    # Create PDF parsing
    source = PDFDocumentParser.from_file(pdf_path)
    
    # Load documents
    documents = []
    async for doc in source.get_documents():
        documents.append(doc)
        print(f"\n✓ Loaded: {doc.metadata['filename']}")
        print(f"  - Size: {doc.metadata['file_size']:,} bytes")
        print(f"  - Text length: {doc.metadata['total_length']:,} characters")
        print(f"  - Preview: {doc.text[:200]}...")
    
    return documents


async def example_directory_pdfs():
    """Load all PDFs from a directory."""
    print("\n" + "=" * 80)
    print("Example 2: Loading PDFs from Directory")
    print("=" * 80)
    
    pdf_dir = Path("/mnt/storage/data/diverse")  # Replace with your directory
    
    if not pdf_dir.exists():
        print(f"\n⚠️  Directory not found: {pdf_dir}")
        print("Creating example directory...")
        pdf_dir.mkdir(exist_ok=True)
        print(f"Please add PDF files to: {pdf_dir.absolute()}")
        return
    
    # Create PDF parsing for directory
    source = PDFDocumentParser.from_directory(
        directory=pdf_dir,
        pattern="*",
        page_chunks=True,
        recursive=True
    )
    
    # Load all PDFs
    documents = []
    async for doc in source.get_documents():
        documents.append(doc)
        print(f"\n✓ Loaded: {doc.metadata['filename']}")
        print(f"  - Source: {doc.metadata['parsing']}")
        print(f"  - Size: {len(doc.text):,} characters")
    
    print(f"\n📊 Total PDFs loaded: {len(documents)}")
    return documents


async def example_page_chunks():
    """Split PDFs into separate documents per page."""
    print("\n" + "=" * 80)
    print("Example 3: Splitting PDF by Pages")
    print("=" * 80)
    
    pdf_path = Path("./sample.pdf")  # Replace with your PDF path
    
    if not pdf_path.exists():
        print(f"\n⚠️  PDF file not found: {pdf_path}")
        return
    
    # Create parsing with page_chunks enabled
    source = PDFDocumentParser.from_file(
        pdf_path,
        page_chunks=True
    )
    
    # Load pages as separate documents
    pages = []
    async for page_doc in source.get_documents():
        pages.append(page_doc)
        page_num = page_doc.metadata['page_number']
        total_pages = page_doc.metadata['total_pages']
        print(f"\n✓ Page {page_num}/{total_pages}")
        print(f"  - Length: {len(page_doc.text):,} characters")
        print(f"  - Preview: {page_doc.text[:150]}...")
    
    print(f"\n📄 Total pages: {len(pages)}")
    return pages


async def example_pdf_indexing():
    """Index PDF content for semantic search."""
    print("\n" + "=" * 80)
    print("Example 4: Indexing PDF Content")
    print("=" * 80)
    
    pdf_path = Path("./sample.pdf")
    
    if not pdf_path.exists():
        print(f"\n⚠️  PDF file not found: {pdf_path}")
        return
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping indexing example.")
        return
    
    print("\n1. Loading PDF...")
    source = PDFDocumentParser.from_file(pdf_path)
    
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
        model="text-embedding-3-small",
        api_key=api_key
    )
    
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="pdf_docs",
        embeddings=embeddings
    )
    
    index = VectorIndex(vector_store=vector_store)
    
    # Index nodes
    print("\n4. Indexing chunks...")
    await index.add_nodes(DocumentNode, nodes, show_progress=True)
    print("   ✓ Indexing complete")
    
    # Search
    print("\n5. Searching indexed content...")
    query = "What is the main topic?"
    results = await index.search_by_text(query, k=3)
    
    print(f"\n   Query: '{query}'")
    print(f"   Found {len(results)} results:\n")
    
    for i, (node, score) in enumerate(results, 1):
        print(f"   {i}. Score: {score:.4f}")
        print(f"      Text: {node.text[:200]}...")
        print()


async def main():
    """Run all examples."""
    print("\n🔖 PDF Document Source Examples\n")

    # Example 2: Directory of PDFs
    await example_directory_pdfs()

    # Example 1: Single PDF
    await example_single_pdf()
    
    # Example 3: Page chunks
    await example_page_chunks()
    
    # Example 4: Indexing and search
    await example_pdf_indexing()
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)
    
    print("\n📚 Usage Tips:")
    print("  1. Use page_chunks=True for large PDFs to process pages separately")
    print("  2. pymupdf4llm preserves markdown formatting (headings, tables, etc.)")
    print("  3. Metadata includes filename, file_size, page numbers, etc.")
    print("  4. Combine with node parsers for optimal chunk sizes")
    print("  5. Index PDF content for semantic search capabilities")


if __name__ == "__main__":
    asyncio.run(main())
