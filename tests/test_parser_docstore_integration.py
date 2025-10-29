"""
Tests for parser integration with DocumentStore.
"""

import pytest
from pathlib import Path
from fetchcraft import DocumentNode, TextFileDocumentParser

# Check if motor is available
try:
    import motor.motor_asyncio
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_parser_with_docstore():
    """Test parser automatically stores DocumentNode in doc_store."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        from fetchcraft import MongoDBDocumentStore
        
        # Create mock MongoDB client and store
        client = AsyncMongoMockClient()
        doc_store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_docs",
            client=client
        )
        
        # Create parser with doc_store
        parser = TextFileDocumentParser(doc_store=doc_store)
        
        # Create temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\n")
            temp_path = Path(f.name)
        
        try:
            # Parse file
            nodes = await parser.from_file(temp_path)
            
            # Verify nodes were returned
            assert len(nodes) > 0
            doc_node = nodes[0]
            assert isinstance(doc_node, DocumentNode)
            
            # Verify DocumentNode was stored in doc_store
            retrieved = await doc_store.get_document(doc_node.id)
            assert retrieved is not None
            assert retrieved.id == doc_node.id
            assert retrieved.text == doc_node.text
            assert isinstance(retrieved, DocumentNode)
            
        finally:
            # Clean up temp file
            temp_path.unlink()
            await doc_store.close()
            
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_parser_with_docstore_override():
    """Test parser doc_store can be overridden per call."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        from fetchcraft import MongoDBDocumentStore
        
        # Create two mock stores
        client = AsyncMongoMockClient()
        
        store1 = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="store1",
            client=client
        )
        
        store2 = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="store2",
            client=client
        )
        
        # Create parser with store1
        parser = TextFileDocumentParser(doc_store=store1)
        
        # Create temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document")
            temp_path = Path(f.name)
        
        try:
            # Parse with store2 override
            nodes = await parser.from_file(temp_path, doc_store=store2)
            doc_node = nodes[0]
            
            # Verify it was stored in store2, not store1
            assert await store2.document_exists(doc_node.id)
            assert not await store1.document_exists(doc_node.id)
            
        finally:
            temp_path.unlink()
            await store1.close()
            await store2.close()
            
    except ImportError:
        pytest.skip("mongomock-motor not installed")


@pytest.mark.skipif(not MOTOR_AVAILABLE, reason="motor not installed")
@pytest.mark.asyncio
async def test_parser_parse_directory_async_with_docstore():
    """Test parsing directory with doc_store."""
    try:
        from mongomock_motor import AsyncMongoMockClient
        from fetchcraft import MongoDBDocumentStore
        
        # Create mock store
        client = AsyncMongoMockClient()
        doc_store = MongoDBDocumentStore(
            database_name="test_db",
            collection_name="test_docs",
            client=client
        )
        
        parser = TextFileDocumentParser(doc_store=doc_store)
        
        # Create temporary directory with files
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple test files
            (temp_path / "file1.txt").write_text("Document 1 content")
            (temp_path / "file2.txt").write_text("Document 2 content")
            (temp_path / "file3.txt").write_text("Document 3 content")
            
            # Parse directory
            results = await parser.parse_directory(temp_path)
            
            # Verify all files were parsed
            assert len(results) == 3
            
            # Verify all DocumentNodes were stored
            doc_count = await doc_store.count_documents()
            assert doc_count == 3
            
            # Verify we can retrieve them
            for file_path, nodes in results.items():
                doc_node = nodes[0]
                retrieved = await doc_store.get_document(doc_node.id)
                assert retrieved is not None
                assert isinstance(retrieved, DocumentNode)
        
        await doc_store.close()
            
    except ImportError:
        pytest.skip("mongomock-motor not installed")


