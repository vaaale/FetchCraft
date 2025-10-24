"""Test that Node properties are properly persisted in the vector store."""

import asyncio
from qdrant_client import QdrantClient
from rag_framework import VectorIndex, QdrantVectorStore, Node, Chunk


class MockEmbeddings:
    """Mock embeddings for testing."""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    async def embed_query(self, text: str):
        """Return a mock embedding."""
        return [0.1] * self.dimension
    
    async def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [[0.1] * self.dimension for _ in texts]
    
    async def aget_dimension(self):
        """Get dimension."""
        return self.dimension


async def test_node_persistence():
    """Test that all Node properties are stored and retrieved correctly."""
    
    # Create in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Create embeddings (mock for testing)
    embeddings = MockEmbeddings(dimension=384)
    
    # Create vector store with Node as document class
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_size=384
    )
    
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings,
        index_id="test_index"
    )
    
    # Create a parent node
    parent_node = Node(
        text="This is a parent document",
        metadata={"type": "parent", "category": "test"},
        embedding=[0.1] * 384
    )
    
    # Create child nodes with relationships
    node1 = Node(
        text="First child node",
        metadata={"type": "child", "order": 1},
        embedding=[0.2] * 384,
        parent_id=parent_node.id
    )
    
    node2 = Node(
        text="Second child node",
        metadata={"type": "child", "order": 2},
        embedding=[0.3] * 384,
        parent_id=parent_node.id,
        previous_id=node1.id
    )
    
    node1.next_id = node2.id
    
    # Add nodes to index
    node_ids = await index.add_documents([parent_node, node1, node2])
    
    print("✓ Added 3 nodes to the index")
    print(f"  Parent ID: {parent_node.id}")
    print(f"  Node1 ID: {node1.id}")
    print(f"  Node2 ID: {node2.id}")
    print()
    
    # Retrieve and verify parent node
    retrieved_parent = await index.get_document(parent_node.id)
    assert retrieved_parent is not None, "Parent node not found"
    assert retrieved_parent.text == parent_node.text, "Parent text mismatch"
    assert retrieved_parent.metadata["type"] == "parent", "Parent metadata mismatch"
    assert retrieved_parent.metadata["category"] == "test", "Parent metadata mismatch"
    assert retrieved_parent.embedding is not None, "Parent embedding not retrieved"
    print("✓ Parent node retrieved correctly")
    print(f"  Text: {retrieved_parent.text}")
    print(f"  Metadata: {retrieved_parent.metadata}")
    print()
    
    # Retrieve and verify first child node
    retrieved_node1 = await index.get_document(node1.id)
    assert retrieved_node1 is not None, "Node1 not found"
    assert retrieved_node1.text == node1.text, "Node1 text mismatch"
    assert retrieved_node1.parent_id == parent_node.id, "Parent relationship not preserved"
    assert retrieved_node1.next_id == node2.id, "Next relationship not preserved"
    assert retrieved_node1.previous_id is None, "Previous should be None"
    print("✓ First child node retrieved correctly")
    print(f"  Text: {retrieved_node1.text}")
    print(f"  Parent ID: {retrieved_node1.parent_id}")
    print(f"  Next ID: {retrieved_node1.next_id}")
    print(f"  Previous ID: {retrieved_node1.previous_id}")
    print()
    
    # Retrieve and verify second child node
    retrieved_node2 = await index.get_document(node2.id)
    assert retrieved_node2 is not None, "Node2 not found"
    assert retrieved_node2.text == node2.text, "Node2 text mismatch"
    assert retrieved_node2.parent_id == parent_node.id, "Parent relationship not preserved"
    assert retrieved_node2.previous_id == node1.id, "Previous relationship not preserved"
    assert retrieved_node2.next_id is None, "Next should be None"
    print("✓ Second child node retrieved correctly")
    print(f"  Text: {retrieved_node2.text}")
    print(f"  Parent ID: {retrieved_node2.parent_id}")
    print(f"  Next ID: {retrieved_node2.next_id}")
    print(f"  Previous ID: {retrieved_node2.previous_id}")
    print()
    
    # Test with Chunk (which inherits from Node)
    chunk = Chunk.from_text(
        text="This is a chunk with extra properties",
        chunk_index=0,
        start_char_idx=0,
        end_char_idx=37,
        metadata={"source": "test.txt"}
    )
    chunk.embedding = [0.4] * 384
    
    chunk_ids = await index.add_documents([chunk])
    print("✓ Added Chunk to the index")
    
    # Retrieve chunk and verify all properties
    retrieved_chunk = await index.get_document(chunk.id)
    assert retrieved_chunk is not None, "Chunk not found"
    assert retrieved_chunk.text == chunk.text, "Chunk text mismatch"
    assert retrieved_chunk.chunk_index == 0, "Chunk index not preserved"
    assert retrieved_chunk.start_char_idx == 0, "Start char index not preserved"
    assert retrieved_chunk.end_char_idx == 37, "End char index not preserved"
    assert retrieved_chunk.metadata["source"] == "test.txt", "Chunk metadata mismatch"
    assert retrieved_chunk.metadata["chunk"] == True, "Chunk metadata flag missing"
    print("✓ Chunk retrieved correctly with all properties")
    print(f"  Text: {retrieved_chunk.text}")
    print(f"  Chunk index: {retrieved_chunk.chunk_index}")
    print(f"  Char range: {retrieved_chunk.start_char_idx}-{retrieved_chunk.end_char_idx}")
    print(f"  Metadata: {retrieved_chunk.metadata}")
    print()
    
    # Search and verify results
    query_embedding = [0.25] * 384
    results = await index.search(query_embedding, k=5)
    
    print(f"✓ Search returned {len(results)} results")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.3f} | Type: {type(doc).__name__} | Text: {doc.text[:40]}...")
    print()
    
    print("="*60)
    print("All Node properties are correctly persisted! ✓")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_node_persistence())
