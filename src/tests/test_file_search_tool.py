"""
Tests for FileSearchTool.
"""

import pytest
import tempfile
from pathlib import Path
from fetchcraft import FileSearchTool, FileSearchResult


# Check if fsspec is available
try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_tool_creation():
    """Test creating a FileSearchTool."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(
            root_path=temp_dir,
            name="test_search",
            max_results=10
        )
        
        assert tool.name == "test_search"
        assert tool.root_path == temp_dir
        assert tool.max_results == 10
        assert tool.recursive is True


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_tool_from_local_path():
    """Test creating FileSearchTool from local path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool.from_local_path(
            root_path=temp_dir,
            max_results=20,
            allowed_extensions=['.txt', '.md']
        )
        
        assert tool.root_path == temp_dir
        assert tool.max_results == 20
        assert tool.allowed_extensions == ['.txt', '.md']


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_basic():
    """Test basic file search functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "file1.txt").write_text("content 1")
        (temp_path / "file2.txt").write_text("content 2")
        (temp_path / "file3.md").write_text("content 3")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        # Mock context
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for txt files
        result = await tool(ctx, "*.txt", name_only=True)
        
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "file3.md" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_with_extension_filter():
    """Test file search with extension filtering."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create files with different extensions
        (temp_path / "doc1.txt").write_text("text")
        (temp_path / "doc2.md").write_text("markdown")
        (temp_path / "doc3.py").write_text("python")
        
        tool = FileSearchTool(
            root_path=temp_dir,
            allowed_extensions=['.txt', '.md']
        )
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for all files - should only return .txt and .md
        result = await tool(ctx, "*", name_only=True)
        
        assert "doc1.txt" in result
        assert "doc2.md" in result
        assert "doc3.py" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_recursive():
    """Test recursive file search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested directory structure
        subdir = temp_path / "subdir"
        subdir.mkdir()
        
        (temp_path / "root.txt").write_text("root file")
        (subdir / "nested.txt").write_text("nested file")
        
        tool = FileSearchTool(root_path=temp_dir, recursive=True)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search should find both files
        result = await tool(ctx, "*.txt", name_only=True)
        
        assert "root.txt" in result
        assert "nested.txt" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_max_results():
    """Test max_results limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create many files
        for i in range(20):
            (temp_path / f"file{i}.txt").write_text(f"content {i}")
        
        tool = FileSearchTool(root_path=temp_dir, max_results=5)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search should return only 5 files
        result = await tool(ctx, "*.txt", name_only=True)
        
        # Count numbered entries (1. 2. 3. etc.)
        lines = result.split('\n')
        numbered_lines = [line for line in lines if line.strip() and line[0].isdigit()]
        assert len(numbered_lines) == 5  # Should be limited to max_results


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_result():
    """Test FileSearchResult class."""
    result = FileSearchResult(
        path="/data/test.txt",
        size=1024,
        file_type=".txt",
        metadata={"source": "test"}
    )
    
    assert result.path == "/data/test.txt"
    assert result.size == 1024
    assert result.file_type == ".txt"
    assert result.metadata["source"] == "test"


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_no_results():
    """Test file search when no files match."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create one file
        (temp_path / "file.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for non-existent pattern
        result = await tool(ctx, "*.xyz", name_only=True)
        
        assert "No files found" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_custom_formatter():
    """Test file search with custom formatter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("content")
        
        def custom_formatter(results):
            return f"Custom: Found {len(results)} files"
        
        tool = FileSearchTool(
            root_path=temp_dir,
            formatter=custom_formatter
        )
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, "*.txt")
        
        assert "Custom: Found 1 files" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_invalid_root():
    """Test that FileSearchTool raises error for invalid root path."""
    with pytest.raises(ValueError, match="Root path does not exist"):
        FileSearchTool(root_path="/nonexistent/path/12345")


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_name_only_vs_full_path():
    """Test difference between name_only and full path search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested structure
        subdir = temp_path / "specific_dir"
        subdir.mkdir()
        
        (temp_path / "file.txt").write_text("root")
        (subdir / "file.txt").write_text("nested")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # name_only=True should find both
        result_name = await tool(ctx, "file.txt", name_only=True)
        assert result_name.count("file.txt") == 2
        
        # name_only=False with specific path should find one
        result_path = await tool(ctx, "specific_dir/file.txt", name_only=False)
        assert result_path.count("file.txt") == 1
        assert "specific_dir" in result_path
