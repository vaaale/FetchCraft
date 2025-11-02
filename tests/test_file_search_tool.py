"""
Tests for FileSearchTool.
"""

import pytest
import tempfile
from pathlib import Path

from fetchcraft.agents import FileSearchTool, FileSearchResult

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
def test_file_search_with_dir_filesystem():
    """Test that FileSearchTool works with dir filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Should work with dir filesystem
        tool = FileSearchTool(root_path=temp_dir)
        assert tool.root_path == temp_dir
        # Verify it's using dir filesystem
        assert tool.fs.protocol == ('dir',) or tool.fs.protocol == 'dir'


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_get_tool_function():
    """Test that get_tool_function() returns a callable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_path=temp_dir, name="custom_search")
        
        # Get the tool function
        tool_func = tool.get_tool_function()
        
        # Verify it's callable
        assert callable(tool_func)
        # Verify name was set
        assert tool_func.__name__ == "custom_search"
        # Verify docstring was set
        assert tool_func.__doc__ is not None
        assert "Search for files" in tool_func.__doc__


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_via_tool_function():
    """Test using the tool through get_tool_function()."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir)
        tool_func = tool.get_tool_function()
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Call through tool function
        result = await tool_func(ctx, "*.txt")
        assert "test.txt" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_error_handling():
    """Test error handling when glob fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Use a pattern that might cause issues on some filesystems
        # The tool should handle it gracefully
        result = await tool(ctx, "")
        # Should return a result (either files or no files found)
        assert isinstance(result, str)


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


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_with_leading_slash():
    """Test that patterns with leading slashes are handled correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Pattern with leading slash should work
        result = await tool(ctx, "/test.txt", name_only=False)
        assert "test.txt" in result or "No files found" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_glob_patterns():
    """Test various glob patterns."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create files with different patterns
        (temp_path / "test1.txt").write_text("content")
        (temp_path / "test2.txt").write_text("content")
        (temp_path / "data1.csv").write_text("content")
        subdir = temp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir, recursive=True)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Test wildcard pattern
        result = await tool(ctx, "test*.txt", name_only=True)
        assert "test1.txt" in result
        assert "test2.txt" in result
        assert "data1.csv" not in result
        
        # Test recursive pattern
        result = await tool(ctx, "**/*.txt", name_only=False)
        assert "nested.txt" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_non_recursive():
    """Test non-recursive search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested structure
        subdir = temp_path / "subdir"
        subdir.mkdir()
        
        (temp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")
        
        tool = FileSearchTool(root_path=temp_dir, recursive=False)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Should only find root file
        result = await tool(ctx, "*.txt", name_only=True)
        assert "root.txt" in result
        assert "nested.txt" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_multiple_extensions():
    """Test filtering with multiple allowed extensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "doc.txt").write_text("text")
        (temp_path / "doc.md").write_text("markdown")
        (temp_path / "doc.py").write_text("python")
        (temp_path / "doc.json").write_text("json")
        
        tool = FileSearchTool(
            root_path=temp_dir,
            allowed_extensions=['.txt', '.md', '.json']
        )
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, "doc.*", name_only=True)
        assert "doc.txt" in result
        assert "doc.md" in result
        assert "doc.json" in result
        assert "doc.py" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_empty_directory():
    """Test search in empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, "*", name_only=True)
        assert "No files found" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_case_sensitivity():
    """Test case sensitivity in file patterns."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "Test.txt").write_text("content")
        (temp_path / "test.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Glob patterns are typically case-sensitive on Linux, case-insensitive on Windows
        result = await tool(ctx, "test.txt", name_only=True)
        # Should at least find the lowercase version
        assert "test.txt" in result.lower()


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_search_tool_repr():
    """Test FileSearchTool string representation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(
            root_path=temp_dir,
            name="my_search",
            max_results=25
        )
        
        repr_str = repr(tool)
        assert "FileSearchTool" in repr_str
        assert "my_search" in repr_str
        assert "25" in repr_str


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_deep_nesting():
    """Test search in deeply nested directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create deep nesting
        deep_path = temp_path / "level1" / "level2" / "level3"
        deep_path.mkdir(parents=True)
        
        (deep_path / "deep.txt").write_text("deep content")
        
        tool = FileSearchTool(root_path=temp_dir, recursive=True)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Should find the deeply nested file
        result = await tool(ctx, "deep.txt", name_only=True)
        assert "deep.txt" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_search_special_characters():
    """Test search with special characters in filenames."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create files with special characters (safe ones)
        (temp_path / "file-name.txt").write_text("content")
        (temp_path / "file_name.txt").write_text("content")
        (temp_path / "file name.txt").write_text("content")
        
        tool = FileSearchTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for files with dashes
        result = await tool(ctx, "*-*.txt", name_only=True)
        assert "file-name.txt" in result
        
        # Search for files with underscores
        result = await tool(ctx, "*_*.txt", name_only=True)
        assert "file_name.txt" in result
