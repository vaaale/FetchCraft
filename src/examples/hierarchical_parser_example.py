"""
Example: Using HierarchicalNodeParser for document parsing

This example demonstrates:
1. Creating a hierarchical parser
2. Parsing documents into parent chunks and child SymNodes
3. Analyzing the hierarchical structure
4. Comparing with SimpleNodeParser
"""

from fetchcraft.node_parser import HierarchicalNodeParser, SimpleNodeParser
from fetchcraft.node import DocumentNode, Chunk, SymNode


def main():
    print("="*80)
    print("HierarchicalNodeParser Example")
    print("="*80)
    
    # Sample document
    document_text = """
    Artificial Intelligence (AI) has revolutionized numerous fields in recent years.
    Machine learning, a subset of AI, enables computers to learn from data without 
    explicit programming. Deep learning, which uses neural networks with multiple 
    layers, has achieved remarkable success in image recognition and natural language 
    processing.
    
    The applications of AI are vast and growing. In healthcare, AI systems can 
    diagnose diseases from medical images with accuracy comparable to expert 
    radiologists. In finance, algorithms detect fraudulent transactions in real-time.
    Autonomous vehicles use AI to navigate complex environments safely.
    
    However, AI also raises important ethical considerations. Issues of bias in 
    training data, privacy concerns, and the impact on employment require careful 
    attention. As AI systems become more powerful, ensuring they align with human 
    values becomes increasingly critical.
    """ * 3  # Repeat to make it longer
    
    doc = DocumentNode.from_text(text=document_text)
    
    # Example 1: Basic Hierarchical Parsing
    print("\n" + "="*80)
    print("Example 1: Basic Hierarchical Parsing")
    print("="*80)
    
    parser = HierarchicalNodeParser(
        chunk_size=500,
        overlap=50,
        child_sizes=[150, 75],
        child_overlap=15
    )
    
    print(f"\nParser Configuration:")
    print(f"  - Parent chunk size: {parser.chunk_size}")
    print(f"  - Parent overlap: {parser.overlap}")
    print(f"  - Child sizes: {parser.child_sizes}")
    print(f"  - Child overlap: {parser.child_overlap}")
    
    nodes = parser.get_nodes([doc])
    
    # Analyze the structure
    parent_chunks = [n for n in nodes if isinstance(n, Chunk)]
    child_nodes = [n for n in nodes if isinstance(n, SymNode)]
    
    print(f"\nResults:")
    print(f"  - Total nodes created: {len(nodes)}")
    print(f"  - Parent chunks: {len(parent_chunks)}")
    print(f"  - Child SymNodes: {len(child_nodes)}")
    
    # Show first parent chunk
    if parent_chunks:
        parent = parent_chunks[0]
        print(f"\nFirst Parent Chunk:")
        print(f"  - ID: {parent.id[:8]}...")
        print(f"  - Length: {len(parent.text)} chars")
        print(f"  - Children count: {len(parent.children_ids)}")
        print(f"  - Text preview: {parent.text[:100]}...")
    
    # Show child distribution
    children_by_size = {}
    for child in child_nodes:
        size = child.metadata.get("child_size", "unknown")
        children_by_size[size] = children_by_size.get(size, 0) + 1
    
    print(f"\nChild SymNodes by size:")
    for size, count in sorted(children_by_size.items()):
        print(f"  - {size} chars: {count} nodes")
    
    # Example 2: Parent-Child Relationships
    print("\n" + "="*80)
    print("Example 2: Parent-Child Relationships")
    print("="*80)
    
    if parent_chunks and child_nodes:
        parent = parent_chunks[0]
        parent_children = [c for c in child_nodes if c.parent_id == parent.id]
        
        print(f"\nParent Chunk #{parent.chunk_index}:")
        print(f"  - Has {len(parent_children)} children")
        
        # Show a few children
        for i, child in enumerate(parent_children[:3]):
            print(f"\n  Child #{i} (size={child.metadata['child_size']}):")
            print(f"    - Text: {child.text[:60]}...")
            print(f"    - Requires resolution: {child.requires_parent_resolution()}")
    
    # Example 3: Sequential Linking
    print("\n" + "="*80)
    print("Example 3: Sequential Linking of Parent Chunks")
    print("="*80)
    
    print(f"\nParent chunk sequence:")
    for i, parent in enumerate(parent_chunks[:5]):  # Show first 5
        prev = "None" if parent.previous_id is None else parent.previous_id[:8] + "..."
        next_ = "None" if parent.next_id is None else parent.next_id[:8] + "..."
        print(f"  Chunk {i}: prev={prev}, next={next_}")
    
    # Example 4: Comparison with SimpleNodeParser
    print("\n" + "="*80)
    print("Example 4: Comparison with SimpleNodeParser")
    print("="*80)
    
    simple_parser = SimpleNodeParser(chunk_size=500, overlap=50)
    simple_nodes = simple_parser.get_nodes([doc])
    
    print(f"\nSimpleNodeParser:")
    print(f"  - Total nodes: {len(simple_nodes)}")
    print(f"  - All are Chunk objects")
    print(f"  - No hierarchical structure")
    
    print(f"\nHierarchicalNodeParser:")
    print(f"  - Total nodes: {len(nodes)}")
    print(f"  - Parent chunks: {len(parent_chunks)}")
    print(f"  - Child SymNodes: {len(child_nodes)}")
    print(f"  - Ratio: {len(child_nodes)/len(parent_chunks):.1f} children per parent")
    
    print(f"\nStorage comparison:")
    print(f"  - Simple: {len(simple_nodes)} nodes")
    print(f"  - Hierarchical: {len(nodes)} nodes ({len(nodes)/len(simple_nodes):.1f}x)")
    
    # Example 5: Custom Metadata
    print("\n" + "="*80)
    print("Example 5: Custom Metadata Propagation")
    print("="*80)
    
    custom_metadata = {
        "source": "ai_overview.txt",
        "author": "John Doe",
        "category": "technology"
    }
    
    nodes_with_metadata = parser.get_nodes([doc], metadata=custom_metadata)
    
    print(f"\nCustom metadata added to all nodes:")
    sample_node = nodes_with_metadata[0]
    for key, value in custom_metadata.items():
        print(f"  - {key}: {sample_node.metadata.get(key)}")
    
    # Example 6: Multiple Child Size Levels
    print("\n" + "="*80)
    print("Example 6: Multiple Granularity Levels")
    print("="*80)
    
    multi_level_parser = HierarchicalNodeParser(
        chunk_size=800,
        child_sizes=[400, 200, 100, 50]  # 4 levels!
    )
    
    multi_nodes = multi_level_parser.get_nodes([doc])
    multi_children = [n for n in multi_nodes if isinstance(n, SymNode)]
    
    print(f"\nMulti-level parser (4 child sizes):")
    print(f"  - Total nodes: {len(multi_nodes)}")
    print(f"  - Child nodes: {len(multi_children)}")
    
    # Count by size
    by_size = {}
    for child in multi_children:
        size = child.metadata.get("child_size")
        by_size[size] = by_size.get(size, 0) + 1
    
    print(f"\n  Distribution by size:")
    for size in sorted(by_size.keys(), reverse=True):
        count = by_size[size]
        print(f"    - {size:4d} chars: {count:3d} nodes")
    
    print("\n" + "="*80)
    print("Example completed!")
    print("="*80)
    
    # Usage tips
    print("\n📚 Usage Tips:")
    print("  1. Use larger parents (4096+) for context-rich retrieval")
    print("  2. Use multiple child sizes for different search granularities")
    print("  3. Smaller children (256-512) work well for precise semantic search")
    print("  4. Index only children in vector store, retrieve parents for context")
    print("  5. Adjust overlap based on content: higher for technical docs")


if __name__ == "__main__":
    main()
