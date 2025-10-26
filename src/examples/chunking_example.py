"""
Example demonstrating different chunking strategies.

This example shows how to:
1. Use CharacterChunkingStrategy for simple fixed-size chunks
2. Use HierarchicalChunkingStrategy for parent-child relationships
3. Compare the output of different strategies
"""

from fetchcraft import (
    TextFileDocumentParser,
    CharacterChunkingStrategy,
    HierarchicalChunkingStrategy,
    Chunk,
    SymNode
)


def demonstrate_character_chunking():
    """Demonstrate simple character-based chunking."""
    print("="*70)
    print("Character Chunking Strategy")
    print("="*70)
    
    # Create a character chunking strategy
    chunker = CharacterChunkingStrategy(
        chunk_size=100,
        overlap=20,
        separator=" "
    )
    
    # Sample text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any system that perceives its environment and takes actions that maximize 
    its chance of achieving its goals.
    """
    
    # Create parser with character chunking
    parser = TextFileDocumentParser(chunker=chunker)
    chunks = parser.parse(text.strip())
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chunk.text)} chars):")
        print(f"  {chunk.text[:80]}...")
        print(f"  Metadata: {chunk.metadata.get('chunk_strategy')}")
        print()


def demonstrate_hierarchical_chunking():
    """Demonstrate hierarchical chunking with parent-child relationships."""
    print("\n" + "="*70)
    print("Hierarchical Chunking Strategy (Multi-Level)")
    print("="*70)
    
    # Create a hierarchical chunking strategy with multiple child sizes
    chunker = HierarchicalChunkingStrategy(
        chunk_size=300,      # Parent chunk size
        overlap=20,
        child_chunks=[100, 50],  # Multiple child chunk sizes
        child_overlap=10
    )
    
    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It focuses on the development of computer programs that can 
    access data and use it to learn for themselves. The process of learning 
    begins with observations or data, such as examples, direct experience, 
    or instruction, in order to look for patterns in data and make better 
    decisions in the future.
    """
    
    # Create parser with hierarchical chunking
    parser = TextFileDocumentParser(chunker=chunker)
    nodes = parser.parse(text.strip())
    
    # Separate parent chunks from child SymNodes
    parent_chunks = [n for n in nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    child_nodes = [n for n in nodes if isinstance(n, SymNode)]
    
    print(f"\nCreated {len(parent_chunks)} parent chunks and {len(child_nodes)} child nodes:\n")
    
    for i, parent in enumerate(parent_chunks):
        print(f"Parent Chunk {i + 1} ({len(parent.text)} chars):")
        print(f"  ID: {parent.id[:8]}...")
        print(f"  Text: {parent.text[:80]}...")
        print(f"  Type: {parent.metadata.get('chunk_type')}")
        
        # Find child nodes for this parent, grouped by size
        children = [c for c in child_nodes if c.parent_id == parent.id]
        
        # Group by size
        size_groups = {}
        for child in children:
            size = child.metadata.get('child_size', 0)
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(child)
        
        print(f"\n  Child Nodes (total: {len(children)}, {len(size_groups)} size levels):")
        for size, group in sorted(size_groups.items(), reverse=True):
            print(f"    Size {size}: {len(group)} nodes")
            for j, child in enumerate(group[:2]):  # Show first 2
                print(f"      [{j + 1}] {child.text[:40]}...")
        print()


def compare_strategies():
    """Compare the two strategies side by side."""
    print("\n" + "="*70)
    print("Strategy Comparison")
    print("="*70)
    
    text = "AI is transforming how we live and work. " * 10
    
    # Character chunking
    char_chunker = CharacterChunkingStrategy(chunk_size=50, overlap=10)
    char_parser = TextFileDocumentParser(chunker=char_chunker)
    char_chunks = char_parser.parse(text)
    
    # Hierarchical chunking with multiple child sizes
    hier_chunker = HierarchicalChunkingStrategy(
        chunk_size=100,
        overlap=10,
        child_chunks=[50, 25],  # Two levels of children
        child_overlap=5
    )
    hier_parser = TextFileDocumentParser(chunker=hier_chunker)
    hier_nodes = hier_parser.parse(text)
    
    print(f"\nText length: {len(text)} characters")
    print(f"\nCharacter Strategy:")
    print(f"  - Total chunks: {len(char_chunks)}")
    print(f"  - All chunks are searchable")
    print(f"  - Returns chunk context only")
    
    parent_chunks = [n for n in hier_nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    child_nodes = [n for n in hier_nodes if isinstance(n, SymNode)]
    
    print(f"\nHierarchical Strategy:")
    print(f"  - Parent chunks: {len(parent_chunks)} (large context)")
    print(f"  - Child nodes: {len(child_nodes)} (searchable)")
    print(f"  - Total nodes to index: {len(hier_nodes)}")
    print(f"  - Returns parent chunk (more context) when child matches")
    
    print("\n" + "="*70)
    print("Key Differences:")
    print("="*70)
    print("\n1. **Character Strategy**:")
    print("   ✓ Simpler, single-level chunks")
    print("   ✓ Each chunk is independent")
    print("   ✗ May lose context with small chunks")
    
    print("\n2. **Hierarchical Strategy** (Default):")
    print("   ✓ Small child nodes for precise search")
    print("   ✓ Large parent chunks for context")
    print("   ✓ Best of both worlds: precision + context")
    print("   ✗ More complex, more nodes to index")


def main():
    """Run all chunking examples."""
    print("\n")
    print("🔪 Chunking Strategies Demo")
    print("="*70)
    
    demonstrate_character_chunking()
    demonstrate_hierarchical_chunking()
    compare_strategies()
    
    print("\n" + "="*70)
    print("✅ Demo completed!")
    print("="*70)
    
    print("\n💡 Usage in your code:")
    print("""
    # Option 1: Character chunking (simple, single-level)
    chunker = CharacterChunkingStrategy(chunk_size=4096, overlap=200)
    parser = TextFileDocumentParser(chunker=chunker)
    
    # Option 2: Hierarchical chunking with multiple child sizes (recommended)
    chunker = HierarchicalChunkingStrategy(
        chunk_size=4096,           # Parent chunk size
        overlap=200,
        child_chunks=[1024, 512, 256],  # Multiple child sizes
        child_overlap=50
    )
    parser = TextFileDocumentParser(chunker=chunker)
    
    # Option 3: Use defaults (HierarchicalChunkingStrategy with single child size)
    parser = TextFileDocumentParser()  # Uses hierarchical by default
    
    # Parse documents
    chunks = parser.parse_directory(path, pattern="*", recursive=True)
    
    # Features of hierarchical chunking:
    # - Recursive splitting: paragraph → line → sentence → word
    # - Multiple child sizes for multi-granularity search
    # - All children resolve to parent chunk (more context on retrieval)
    """)


if __name__ == "__main__":
    main()
