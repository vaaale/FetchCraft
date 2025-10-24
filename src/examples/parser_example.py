"""
Example demonstrating the use of DocumentParser and TextFileDocumentParser.
"""

from pathlib import Path
from rag_framework import TextFileDocumentParser, Chunk


def create_sample_file(file_path: Path) -> None:
    """Create a sample text file for demonstration."""
    sample_text = """
    Artificial Intelligence (AI) is transforming the world as we know it. 
    Machine learning algorithms are becoming increasingly sophisticated, 
    enabling computers to learn from data and make intelligent decisions. 
    Natural Language Processing (NLP) is a subfield of AI that focuses on 
    the interaction between computers and human language. It enables machines 
    to understand, interpret, and generate human language in valuable ways.
    
    Deep learning, a subset of machine learning, uses neural networks with 
    multiple layers to progressively extract higher-level features from raw input. 
    This approach has led to breakthroughs in computer vision, speech recognition, 
    and natural language understanding. The applications of AI are vast and growing, 
    from autonomous vehicles to medical diagnosis, from recommendation systems to 
    financial trading.
    """.strip()
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(sample_text)


def example_basic_parsing():
    """Basic example of parsing a text file."""
    print("="*60)
    print("EXAMPLE 1: Basic Text File Parsing")
    print("="*60 + "\n")
    
    # Create a sample file
    sample_file = Path("temp_examples/sample_document.txt")
    create_sample_file(sample_file)
    
    # Parse the file with default settings
    chunks = TextFileDocumentParser.from_file(
        file_path=sample_file,
        chunk_size=200,
        overlap=20
    )
    
    print(f"Total chunks created: {len(chunks)}\n")
    
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  ID: {chunk.id}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Length: {len(chunk.text)} characters")
        print(f"  Position: {chunk.start_char_idx} - {chunk.end_char_idx}")
        print(f"  Has parent: {chunk.has_parent()}")
        print(f"  Has previous: {chunk.has_previous()}")
        print(f"  Has next: {chunk.has_next()}")
        print()


def example_custom_chunking():
    """Example with custom chunking parameters."""
    print("="*60)
    print("EXAMPLE 2: Custom Chunking Parameters")
    print("="*60 + "\n")
    
    sample_file = Path("temp_examples/sample_document.txt")
    
    # Parse with larger chunks and more overlap
    chunks = TextFileDocumentParser.from_file(
        file_path=sample_file,
        chunk_size=300,
        overlap=50,
        separator=" "
    )
    
    print(f"Total chunks created: {len(chunks)}\n")
    
    # Show the overlap between first two chunks
    if len(chunks) >= 2:
        chunk1_end = chunks[0].text[-50:]
        chunk2_start = chunks[1].text[:50]
        
        print("Demonstrating overlap:")
        print(f"\nChunk 0 (last 50 chars): ...{chunk1_end}")
        print(f"Chunk 1 (first 50 chars): {chunk2_start}...")
        print()


def example_with_context():
    """Example showing context retrieval from chunks."""
    print("="*60)
    print("EXAMPLE 3: Retrieving Context from Chunks")
    print("="*60 + "\n")
    
    sample_file = Path("temp_examples/sample_document.txt")
    chunks = TextFileDocumentParser.from_file(
        file_path=sample_file,
        chunk_size=200,
        overlap=20
    )
    
    # Get a middle chunk with context
    if len(chunks) >= 3:
        middle_chunk = chunks[1]
        
        print("Middle chunk text:")
        print(middle_chunk.text)
        print("\n" + "-"*60 + "\n")
        
        print("With surrounding context (1 chunk before and after):")
        context = middle_chunk.get_surrounding_context(
            num_chunks_before=1,
            num_chunks_after=1
        )
        print(context)
        print()


def example_directory_parsing():
    """Example of parsing multiple files from a directory."""
    print("="*60)
    print("EXAMPLE 4: Parsing Multiple Files from Directory")
    print("="*60 + "\n")
    
    # Create multiple sample files
    temp_dir = Path("temp_examples/documents")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        file_path = temp_dir / f"document_{i}.txt"
        file_path.write_text(f"This is document number {i}. " * 50)
    
    # Parse all files in the directory
    results = TextFileDocumentParser.parse_directory(
        directory_path=temp_dir,
        chunk_size=150,
        overlap=15,
        pattern="*.txt"
    )
    
    print(f"Parsed {len(results)} files:\n")
    
    for file_path, chunks in results.items():
        print(f"File: {Path(file_path).name}")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  First chunk preview: {chunks[0].text[:50]}...")
        print()


def example_metadata_inspection():
    """Example showing metadata attached to chunks."""
    print("="*60)
    print("EXAMPLE 5: Inspecting Chunk Metadata")
    print("="*60 + "\n")
    
    sample_file = Path("temp_examples/sample_document.txt")
    chunks = TextFileDocumentParser.from_file(
        file_path=sample_file,
        chunk_size=200,
        overlap=20
    )
    
    if chunks:
        chunk = chunks[0]
        print("Chunk metadata:")
        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")
        print()
        
        if chunk.parent:
            print("Parent node metadata:")
            for key, value in chunk.parent.metadata.items():
                print(f"  {key}: {value}")


def cleanup():
    """Clean up temporary files."""
    import shutil
    temp_path = Path("temp_examples")
    if temp_path.exists():
        shutil.rmtree(temp_path)


if __name__ == "__main__":
    try:
        example_basic_parsing()
        print("\n")
        
        example_custom_chunking()
        print("\n")
        
        example_with_context()
        print("\n")
        
        example_directory_parsing()
        print("\n")
        
        example_metadata_inspection()
        
    finally:
        # Clean up temporary files
        cleanup()
        print("\n" + "="*60)
        print("Temporary files cleaned up")
        print("="*60)
