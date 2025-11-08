"""
Unit tests for FilesystemDocumentSource
"""

import tempfile
from pathlib import Path

import pytest

from fetchcraft.node import DocumentNode, NodeType
from fetchcraft.parsing.filesystem import FilesystemDocumentParser


class TestFilesystemDocumentSource:
    """Test suite for FilesystemDocumentSource"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create test files
            (tmp_path / "file1.txt").write_text("This is file 1 content.")
            (tmp_path / "file2.txt").write_text("This is file 2 content.")
            (tmp_path / "README.md").write_text("# README\nThis is a readme file.")
            
            # Create subdirectory with files
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_text("This is file 3 in subdirectory.")
            (subdir / "data.json").write_text('{"key": "value"}')
            
            # Create nested subdirectory
            nested_dir = subdir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file4.txt").write_text("This is file 4 in nested directory.")
            
            yield tmp_path
    
    @pytest.mark.asyncio
    async def test_from_directory_all_files(self, temp_dir):
        """Test reading all files from directory"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="*",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get 3 files from root directory only
        assert len(documents) == 3
        assert all(isinstance(doc, DocumentNode) for doc in documents)
        
        # Check filenames
        filenames = {doc.metadata["filename"] for doc in documents}
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames
        assert "README.md" in filenames
    
    @pytest.mark.asyncio
    async def test_from_directory_recursive(self, temp_dir):
        """Test recursive directory traversal"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="*.txt",
            recursive=True
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get all .txt files recursively (file1, file2, file3, file4)
        assert len(documents) == 4
        
        filenames = {doc.metadata["filename"] for doc in documents}
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames
        assert "file3.txt" in filenames
        assert "file4.txt" in filenames
    
    @pytest.mark.asyncio
    async def test_from_directory_pattern_filtering(self, temp_dir):
        """Test pattern-based file filtering"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="*.md",
            recursive=True
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get only .md files
        assert len(documents) == 1
        assert documents[0].metadata["filename"] == "README.md"
    
    @pytest.mark.asyncio
    async def test_from_file(self, temp_dir):
        """Test reading a single file"""
        file_path = temp_dir / "file1.txt"
        source = FilesystemDocumentParser.from_file(file_path)
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get exactly one document
        assert len(documents) == 1
        assert documents[0].text == "This is file 1 content."
        assert documents[0].metadata["filename"] == "file1.txt"
    
    @pytest.mark.asyncio
    async def test_document_metadata(self, temp_dir):
        """Test that document metadata is properly set"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="file1.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        doc = documents[0]
        metadata = doc.metadata
        
        # Check required metadata fields
        assert "parsing" in metadata
        assert metadata["filename"] == "file1.txt"
        assert "file_size" in metadata
        assert metadata["file_size"] > 0
        assert "total_length" in metadata
        assert metadata["total_length"] == len(doc.text)
    
    @pytest.mark.asyncio
    async def test_custom_metadata(self, temp_dir):
        """Test that custom metadata is added to documents"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="file1.txt",
            recursive=False
        )
        
        custom_metadata = {
            "author": "Test Author",
            "category": "test"
        }
        
        documents = []
        async for doc in source.get_documents(metadata=custom_metadata):
            documents.append(doc)
        
        doc = documents[0]
        
        # Custom metadata should be present
        assert doc.metadata["author"] == "Test Author"
        assert doc.metadata["category"] == "test"
        
        # Default metadata should also be present
        assert doc.metadata["filename"] == "file1.txt"
    
    @pytest.mark.asyncio
    async def test_document_text_content(self, temp_dir):
        """Test that document text is correctly read"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="README.md",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        doc = documents[0]
        assert "# README" in doc.text
        assert "This is a readme file." in doc.text
    
    @pytest.mark.asyncio
    async def test_doc_id_is_set(self, temp_dir):
        """Test that doc_id is set on DocumentNode"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="file1.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        doc = documents[0]
        assert doc.doc_id is not None
        assert doc.doc_id == doc.id  # DocumentNode sets doc_id to its own id
    
    @pytest.mark.asyncio
    async def test_encoding_utf8(self, temp_dir):
        """Test reading UTF-8 encoded files"""
        # Create a file with UTF-8 content
        utf8_file = temp_dir / "utf8.txt"
        utf8_file.write_text("Hello 世界 🌍", encoding="utf-8")
        
        source = FilesystemDocumentParser.from_file(utf8_file)
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        assert len(documents) == 1
        assert "世界" in documents[0].text
        assert "🌍" in documents[0].text
    
    @pytest.mark.asyncio
    async def test_encoding_fallback(self, temp_dir):
        """Test that ISO-8859-1 fallback works"""
        # Create a file with ISO-8859-1 content
        iso_file = temp_dir / "iso.txt"
        iso_file.write_bytes(b"Hello \xe9")  # é in ISO-8859-1
        
        source = FilesystemDocumentParser.from_file(iso_file)
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should successfully read with fallback encoding
        assert len(documents) == 1
        assert "Hello" in documents[0].text
    
    @pytest.mark.asyncio
    async def test_empty_directory(self, temp_dir):
        """Test reading from empty directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        source = FilesystemDocumentParser.from_directory(
            directory=empty_dir,
            pattern="*.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should return no documents
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_no_matching_files(self, temp_dir):
        """Test when no files match the pattern"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="*.nonexistent",
            recursive=True
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should return no documents
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_empty_file(self, temp_dir):
        """Test reading an empty file"""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        source = FilesystemDocumentParser.from_file(empty_file)
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        assert len(documents) == 1
        assert documents[0].text == ""
        assert documents[0].metadata["total_length"] == 0
    
    @pytest.mark.asyncio
    async def test_multiple_documents_order(self, temp_dir):
        """Test that documents are yielded (order may vary by filesystem)"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="file*.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get file1.txt and file2.txt
        assert len(documents) == 2
        filenames = {doc.metadata["filename"] for doc in documents}
        assert filenames == {"file1.txt", "file2.txt"}
    
    @pytest.mark.asyncio
    async def test_subdirectory_pattern(self, temp_dir):
        """Test reading files from specific subdirectory"""
        subdir = temp_dir / "subdir"
        source = FilesystemDocumentParser.from_directory(
            directory=subdir,
            pattern="*.txt",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get only file3.txt from subdir (not nested)
        assert len(documents) == 1
        assert documents[0].metadata["filename"] == "file3.txt"
    
    @pytest.mark.asyncio
    async def test_json_file_reading(self, temp_dir):
        """Test reading non-text files like JSON"""
        source = FilesystemDocumentParser.from_directory(
            directory=temp_dir,
            pattern="*.json",
            recursive=True
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        assert len(documents) == 1
        assert documents[0].metadata["filename"] == "data.json"
        assert '{"key": "value"}' in documents[0].text


class TestFilesystemDocumentSourceConstructors:
    """Test the different constructor methods"""
    
    def test_from_directory_constructor(self):
        """Test from_directory class method"""
        source = FilesystemDocumentParser.from_directory(
            directory=Path("/tmp"),
            pattern="*.txt",
            recursive=True
        )
        
        assert source.directory == Path("/tmp")
        assert source.pattern == "*.txt"
        assert source.recursive is True
        assert source.fs is None
    
    def test_from_file_constructor(self):
        """Test from_file class method"""
        file_path = Path("/tmp/test.txt")
        source = FilesystemDocumentParser.from_file(file_path)
        
        assert source.directory == Path("/tmp")
        assert source.pattern == file_path.name  # Should be filename, not full path
        assert source.recursive is False
        assert source.fs is None
    
    def test_from_fs_constructor(self):
        """Test from_fs class method with fsspec filesystem"""
        import fsspec
        fs = fsspec.filesystem('memory')
        
        source = FilesystemDocumentParser.from_fs(
            fs=fs,
            directory=Path("/data"),
            pattern="*.csv",
            recursive=False
        )
        
        assert source.fs is fs
        assert source.directory == Path("/data")
        assert source.pattern == "*.csv"
        assert source.recursive is False


class TestFilesystemDocumentSourceErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self):
        """Test error handling for nonexistent directory"""
        source = FilesystemDocumentParser.from_file(
            Path("/nonexistent/file.txt")
        )
        
        # Glob on nonexistent directory returns no results (doesn't raise)
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get no documents from nonexistent location
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_encoding(self, tmp_path):
        """Test handling of binary files"""
        # Create a binary file
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x80\x81\x82\x83')
        
        source = FilesystemDocumentParser.from_file(binary_file)
        
        # ISO-8859-1 can decode any byte sequence, so this will succeed
        # (even if the result is not meaningful text)
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        assert len(documents) == 1
        # The text may not be meaningful, but it should be decodable
        assert documents[0].text == '\x80\x81\x82\x83'


class TestFilesystemDocumentSourceIntegration:
    """Integration tests for real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_mixed_file_types(self, tmp_path):
        """Test reading directory with mixed file types"""
        # Create various file types
        (tmp_path / "doc.txt").write_text("Text document")
        (tmp_path / "data.json").write_text('{"data": "json"}')
        (tmp_path / "readme.md").write_text("# Markdown")
        (tmp_path / "config.yaml").write_text("key: value")
        
        source = FilesystemDocumentParser.from_directory(
            directory=tmp_path,
            pattern="*",
            recursive=False
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should read all files
        assert len(documents) == 4
        
        # Each should be a valid DocumentNode
        for doc in documents:
            assert isinstance(doc, DocumentNode)
            assert doc.text
            assert doc.metadata["filename"]
    
    @pytest.mark.asyncio
    async def test_large_directory_structure(self, tmp_path):
        """Test reading from complex directory structure"""
        # Create complex structure
        for i in range(3):
            dir_path = tmp_path / f"level1_{i}"
            dir_path.mkdir()
            (dir_path / f"file_{i}.txt").write_text(f"Level 1 file {i}")
            
            for j in range(2):
                subdir = dir_path / f"level2_{j}"
                subdir.mkdir()
                (subdir / f"file_{i}_{j}.txt").write_text(f"Level 2 file {i}-{j}")
        
        source = FilesystemDocumentParser.from_directory(
            directory=tmp_path,
            pattern="*.txt",
            recursive=True
        )
        
        documents = []
        async for doc in source.get_documents():
            documents.append(doc)
        
        # Should get all 9 files (3 at level 1, 6 at level 2)
        assert len(documents) == 9
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, tmp_path):
        """Test processing documents in batches"""
        # Create multiple files
        for i in range(10):
            (tmp_path / f"doc_{i:02d}.txt").write_text(f"Document {i}")
        
        source = FilesystemDocumentParser.from_directory(
            directory=tmp_path,
            pattern="*.txt",
            recursive=False
        )
        
        # Process in batches
        batch_size = 3
        batch = []
        batch_count = 0
        
        async for doc in source.get_documents():
            batch.append(doc)
            if len(batch) >= batch_size:
                batch_count += 1
                batch = []
        
        # Should have processed multiple batches
        assert batch_count >= 3


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
