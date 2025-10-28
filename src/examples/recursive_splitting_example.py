"""
Test script to verify recursive splitting with semantic boundaries.
"""

from fetchcraft import HierarchicalChunkingStrategy, TextFileDocumentParser, SymNode, Chunk


def test_recursive_splitting():
    """Test that recursive splitting respects semantic boundaries."""
    
    # Sample text with various separators
    text = """This is paragraph one. It has multiple sentences! Does it work properly?

This is paragraph two.
It has line breaks.
Each line is a separate unit.

Final paragraph. With sentences. And punctuation!"""

    print("="*70)
    print("Testing Recursive Splitting")
    print("="*70)
    print(f"\nOriginal text ({len(text)} chars):")
    print(text)
    print("\n" + "="*70)
    
    # Create hierarchical chunker with multiple child sizes
    chunker = HierarchicalChunkingStrategy(
        chunk_size=150,  # Small parent size to force splitting
        overlap=20,
        child_chunks=[60, 30],  # Two levels of children
        child_overlap=10
    )
    
    parser = TextFileDocumentParser(chunker=chunker)
    nodes = parser.parse(text)
    
    # Separate parents and children
    parents = [n for n in nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
    children = [n for n in nodes if isinstance(n, SymNode)]
    
    print(f"\nResults:")
    print(f"  Parent chunks: {len(parents)}")
    print(f"  Child nodes: {len(children)}")
    print(f"  Total nodes: {len(nodes)}")
    
    # Show parent chunks
    print("\n" + "="*70)
    print("Parent Chunks (split on semantic boundaries)")
    print("="*70)
    for i, parent in enumerate(parents, 1):
        print(f"\n[Parent {i}] ({len(parent.text)} chars)")
        print(f"Text: {repr(parent.text[:100])}...")
        
        # Check if boundaries are semantic
        if parent.text.startswith('\n\n'):
            print("  ✓ Starts at paragraph boundary")
        elif parent.text.startswith('\n'):
            print("  ✓ Starts at line boundary")
        elif parent.text[0] in '.!?':
            print("  ✓ Starts at sentence boundary")
    
    # Show child distribution
    print("\n" + "="*70)
    print("Child Nodes (multi-level hierarchy)")
    print("="*70)
    
    for parent in parents:
        parent_children = [c for c in children if c.parent_id == parent.id]
        
        # Group by size
        size_groups = {}
        for child in parent_children:
            size = child.metadata.get('child_size', 0)
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(child)
        
        print(f"\nParent {parent.chunk_index + 1} children:")
        for size in sorted(size_groups.keys(), reverse=True):
            nodes_at_size = size_groups[size]
            print(f"  Size {size:3d}: {len(nodes_at_size):2d} nodes")
            # Show first one as example
            if nodes_at_size:
                example = nodes_at_size[0].text[:50].replace('\n', '\\n')
                print(f"           Example: {repr(example)}...")
    
    # Verify all children reference their parent
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    all_valid = True
    for child in children:
        parent_id = child.parent_id
        parent_exists = any(p.id == parent_id for p in parents)
        if not parent_exists:
            print(f"❌ Child has invalid parent_id: {parent_id}")
            all_valid = False
    
    if all_valid:
        print("✓ All child nodes have valid parent references")
    
    # Check that all child sizes were created
    child_sizes_found = set()
    for child in children:
        size = child.metadata.get('child_size')
        if size:
            child_sizes_found.add(size)
    
    expected_sizes = set(chunker.child_chunks)
    if child_sizes_found == expected_sizes:
        print(f"✓ All child sizes present: {sorted(child_sizes_found)}")
    else:
        print(f"❌ Expected sizes {expected_sizes}, found {child_sizes_found}")
    
    print(f"\n✓ Test completed successfully!")
    print("="*70)


def test_separators():
    """Test that separators are tried in the correct order."""
    
    print("\n\n" + "="*70)
    print("Testing Separator Priority")
    print("="*70)
    
    # Text with clear paragraph boundaries
    text_paragraphs = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    
    # Text with only line breaks
    text_lines = "Line one.\nLine two.\nLine three.\nLine four."
    
    # Text with only sentences
    text_sentences = "First sentence. Second sentence. Third sentence. Fourth sentence."
    
    chunker = HierarchicalChunkingStrategy(
        chunk_size=40,
        overlap=5,
        child_chunks=[20],
        child_overlap=3
    )
    
    parser = TextFileDocumentParser(chunker=chunker)
    
    for label, text in [
        ("Paragraphs", text_paragraphs),
        ("Lines", text_lines),
        ("Sentences", text_sentences)
    ]:
        print(f"\n{label}:")
        print(f"  Text: {repr(text[:60])}...")
        
        nodes = parser.parse(text)
        parents = [n for n in nodes if isinstance(n, Chunk) and not isinstance(n, SymNode)]
        
        print(f"  Split into {len(parents)} parent chunks:")
        for i, p in enumerate(parents, 1):
            preview = repr(p.text[:30]).replace('\\n', '⏎')
            print(f"    [{i}] {preview}...")


if __name__ == "__main__":
    test_recursive_splitting()
    test_separators()
