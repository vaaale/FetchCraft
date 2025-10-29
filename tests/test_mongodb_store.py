"""
Tests for MongoDB document store.
"""

import pytest

from fetchcraft.document_store import MongoDBDocumentStore
from fetchcraft.node import DocumentNode, Chunk

# Check if motor is available
try:
    import motor.motor_asyncio

    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_add_and_get_document():
    """Test adding and retrieving a document from MongoDB."""
    # Use in-memory MongoDB for testing (mongomock with motor)
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Create a document
        doc = DocumentNode.from_text(
            text="Hello world",
            metadata={"source": "test"}
        )

        # Add document
        doc_id = await store.add_document(doc)
        assert doc_id == doc.id

        # Retrieve document
        retrieved = await store.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.id == doc.id
        assert retrieved.text == "Hello world"
        assert retrieved.metadata["source"] == "test"
        assert isinstance(retrieved, DocumentNode)

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_add_multiple_documents():
    """Test adding multiple documents at once."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Create multiple documents
        doc1 = DocumentNode.from_text("Document 1", metadata={"index": 1})
        doc2 = DocumentNode.from_text("Document 2", metadata={"index": 2})
        doc3 = DocumentNode.from_text("Document 3", metadata={"index": 3})

        # Add all documents
        doc_ids = await store.add_documents([doc1, doc2, doc3])
        assert len(doc_ids) == 3

        # Retrieve all
        retrieved = await store.get_documents(doc_ids)
        assert len(retrieved) == 3
        assert all(isinstance(d, DocumentNode) for d in retrieved)

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_update_document():
    """Test updating a document."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Create and add document
        doc = DocumentNode.from_text("Original text", metadata={"version": 1})
        await store.add_document(doc)

        # Update document
        doc.text = "Updated text"
        doc.metadata["version"] = 2
        success = await store.update_document(doc)
        assert success

        # Retrieve and verify
        retrieved = await store.get_document(doc.id)
        assert retrieved.text == "Updated text"
        assert retrieved.metadata["version"] == 2

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_delete_document():
    """Test deleting a document."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Create and add document
        doc = DocumentNode.from_text("To be deleted")
        await store.add_document(doc)

        # Verify it exists
        assert await store.document_exists(doc.id)

        # Delete it
        success = await store.delete_document(doc.id)
        assert success

        # Verify it's gone
        assert not await store.document_exists(doc.id)
        retrieved = await store.get_document(doc.id)
        assert retrieved is None

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_list_documents():
    """Test listing documents with pagination."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Add multiple documents
        docs = [
            DocumentNode.from_text(f"Document {i}", metadata={"index": i})
            for i in range(10)
        ]
        await store.add_documents(docs)

        # List first 5
        first_batch = await store.list_documents(limit=5, offset=0)
        assert len(first_batch) == 5

        # List next 5
        second_batch = await store.list_documents(limit=5, offset=5)
        assert len(second_batch) == 5

        # Count all
        count = await store.count_documents()
        assert count == 10

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_mongodb_store_get_by_doc_id():
    """Test retrieving all nodes belonging to a document."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()

        store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            client=client
        )

        # Create document and chunks
        doc = DocumentNode.from_text("Full document text")
        chunk1 = Chunk.from_text("Chunk 1", chunk_index=0)
        chunk2 = Chunk.from_text("Chunk 2", chunk_index=1)

        # Set doc_id on chunks
        chunk1.doc_id = doc.id
        chunk2.doc_id = doc.id

        # Add all to store
        await store.add_documents([doc, chunk1, chunk2])

        # Retrieve all by doc_id
        all_nodes = await store.get_documents_by_doc_id(doc.id)
        assert len(all_nodes) == 3

        # Verify we got the document and both chunks
        doc_nodes = [n for n in all_nodes if isinstance(n, DocumentNode)]
        chunk_nodes = [n for n in all_nodes if isinstance(n, Chunk)]
        assert len(doc_nodes) == 1
        assert len(chunk_nodes) == 2

        await store.close()
    except ImportError:
        pytest.skip("mongomock-motor not installed")
