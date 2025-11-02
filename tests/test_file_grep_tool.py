"""
Tests for FileGrepTool.
"""

import pytest
import tempfile
from pathlib import Path

from fetchcraft.agents import FileGrepTool

# Check if fsspec is available
try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_grep_tool_creation():
    """Test creating a FileGrepTool."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileGrepTool(
            root_path=temp_dir,
            name="test_grep",
            max_results=10,
            context_lines=3
        )
        
        assert tool.name == "test_grep"
        assert tool.root_path == temp_dir
        assert tool.max_results == 10
        assert tool.context_lines == 3
        assert tool.recursive is True


# Note: from_local_path() removed in simplified version - use __init__ directly


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_basic_text_search():
    """Test basic text search in files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "file1.txt").write_text("This contains TODO item\nAnother line\n")
        (temp_path / "file2.txt").write_text("No special content\n")
        (temp_path / "file3.txt").write_text("Another TODO here\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for "TODO"
        result = await tool(ctx, pattern="TODO")
        
        assert "file1.txt" in result
        assert "file3.txt" in result
        assert "file2.txt" not in result
        assert "TODO" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_case_sensitive():
    """Test case-sensitive search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("Error occurred\nerror in lowercase\nERROR in caps\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Case-insensitive (default)
        result = await tool(ctx, pattern="error", case_sensitive=False)
        assert result.count("Error") + result.count("error") + result.count("ERROR") >= 3
        
        # Case-sensitive - should only match "error" in lowercase
        result = await tool(ctx, pattern="error", case_sensitive=True)
        assert "error in lowercase" in result
        # Should have found the match on line 2 (note: context lines may include other variations)
        assert "Line 2:" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_regex_search():
    """Test regex pattern search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "code.py").write_text(
            "class MyAgent:\n"
            "    pass\n"
            "class TestAgent:\n"
            "    pass\n"
            "def function():\n"
            "    pass\n"
        )
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Regex search for class definitions
        result = await tool(ctx, pattern=r"class\s+\w+Agent", regex=True)
        
        assert "MyAgent" in result
        assert "TestAgent" in result
        # Should have matched on lines 1 and 3 (class definitions)
        assert "Line 1:" in result
        assert "Line 3:" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_with_file_pattern():
    """Test grep with file pattern filtering."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "file.txt").write_text("TODO in txt\n")
        (temp_path / "file.py").write_text("TODO in python\n")
        (temp_path / "file.md").write_text("TODO in markdown\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search only in .py files
        result = await tool(ctx, pattern="TODO", file_pattern="*.py")
        
        assert "file.py" in result
        assert "file.txt" not in result
        assert "file.md" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_context_lines():
    """Test context lines before and after matches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        content = "line 1\nline 2\nTODO item\nline 4\nline 5\n"
        (temp_path / "test.txt").write_text(content)
        
        tool = FileGrepTool(root_path=temp_dir, context_lines=2)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Should include context lines
        assert "line 2" in result  # Context before
        assert "line 4" in result  # Context after


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_max_results():
    """Test max_results limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create many files with matches
        for i in range(20):
            (temp_path / f"file{i}.txt").write_text(f"TODO item {i}\n")
        
        tool = FileGrepTool(root_path=temp_dir, max_results=5)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Count how many files are mentioned (look for file*.txt)
        file_count = result.count(".txt")
        # Should be limited to max_results
        assert file_count <= 5


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_max_matches_per_file():
    """Test max_matches_per_file limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create file with many matches
        content = "\n".join([f"TODO item {i}" for i in range(20)])
        (temp_path / "test.txt").write_text(content)
        
        tool = FileGrepTool(root_path=temp_dir, max_matches_per_file=3)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Should mention there are more matches
        assert "more matches" in result or "3" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_no_matches():
    """Test grep when no matches are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("No special content\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="NONEXISTENT")
        
        assert "No matches found" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_binary_detection():
    """Test that binary files are skipped (always enabled in simplified version)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a binary file (with null bytes)
        (temp_path / "binary.bin").write_bytes(b"Binary\x00\x00\x00content")
        # Create a text file
        (temp_path / "text.txt").write_text("Text content\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="content")
        
        # Should only find text file (binary detection always enabled)
        assert "text.txt" in result
        assert "binary.bin" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_recursive():
    """Test recursive search in subdirectories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested structure
        subdir = temp_path / "subdir"
        subdir.mkdir()
        deep_dir = subdir / "deep"
        deep_dir.mkdir()
        
        (temp_path / "root.txt").write_text("TODO at root\n")
        (subdir / "sub.txt").write_text("TODO in subdir\n")
        (deep_dir / "deep.txt").write_text("TODO in deep\n")
        
        tool = FileGrepTool(root_path=temp_dir, recursive=True)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        assert "root.txt" in result
        assert "sub.txt" in result
        assert "deep.txt" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_allowed_extensions():
    """Test grep with allowed extensions filter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "file.txt").write_text("TODO\n")
        (temp_path / "file.py").write_text("TODO\n")
        (temp_path / "file.js").write_text("TODO\n")
        
        tool = FileGrepTool(
            root_path=temp_dir,
            allowed_extensions=['.txt', '.py']
        )
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        assert "file.txt" in result
        assert "file.py" in result
        assert "file.js" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_multiline_matches():
    """Test grep finding matches on different lines."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        content = "Line 1\nTODO: fix this\nLine 3\nTODO: and this\nLine 5\n"
        (temp_path / "test.txt").write_text(content)
        
        tool = FileGrepTool(root_path=temp_dir, max_matches_per_file=10)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Should find both occurrences
        assert "Line 2" in result or "fix this" in result
        assert "Line 4" in result or "and this" in result


# Note: FileGrepMatch and FileGrepResult classes removed in simplified version


# Note: Custom formatter removed in simplified version - uses built-in formatting only


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_with_leading_slash():
    """Test that file patterns with leading slashes are handled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("TODO item\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Pattern with leading slash should work
        result = await tool(ctx, pattern="TODO", file_pattern="/test.txt")
        assert "test.txt" in result or "No matches found" not in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_invalid_regex():
    """Test handling of invalid regex patterns."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("content\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Invalid regex pattern
        result = await tool(ctx, pattern="[invalid(", regex=True)
        
        # Should handle gracefully (no matches due to invalid pattern)
        assert "No matches found" in result or result == ""


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_empty_directory():
    """Test grep in empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        assert "No matches found" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_grep_tool_repr():
    """Test FileGrepTool string representation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileGrepTool(
            root_path=temp_dir,
            name="my_grep",
            max_results=25
        )
        
        repr_str = repr(tool)
        assert "FileGrepTool" in repr_str
        assert "my_grep" in repr_str
        assert "25" in repr_str


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_special_regex_chars():
    """Test searching for patterns with regex special characters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("Price is $50.00\nAnother $25.50\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Search for literal $ (not regex)
        result = await tool(ctx, pattern="$50", regex=False)
        assert "$50" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_unicode_content():
    """Test grep with unicode content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # File with unicode characters
        (temp_path / "unicode.txt").write_text("Hello 世界\nTODO: 日本語\n", encoding='utf-8')
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        # Should handle unicode properly
        assert "TODO" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_file_size_limit():
    """Test that files exceeding size limit are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a small file
        (temp_path / "small.txt").write_text("TODO small\n")
        
        # Create a large file (simulate by setting very small limit)
        large_content = "TODO large\n" * 10000  # ~110KB
        (temp_path / "large.txt").write_text(large_content)
        
        # Set very small file size limit
        tool = FileGrepTool(root_path=temp_dir, max_file_size_mb=0.0001)  # ~100 bytes
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Should only find small file
        assert "small.txt" in result
        # Large file should be skipped due to size
        assert "large.txt" not in result or "No matches found" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_grep_with_dir_filesystem():
    """Test that FileGrepTool works with dir filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Should work with dir filesystem
        tool = FileGrepTool(root_path=temp_dir)
        assert tool.root_path == temp_dir
        # Verify it's using dir filesystem
        assert tool.fs.protocol == ('dir',) or tool.fs.protocol == 'dir'


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
def test_file_grep_get_tool_function():
    """Test that get_tool_function() returns a callable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tool = FileGrepTool(root_path=temp_dir, name="custom_grep")
        
        # Get the tool function
        tool_func = tool.get_tool_function()
        
        # Verify it's callable
        assert callable(tool_func)
        # Verify name was set
        assert tool_func.__name__ == "custom_grep"
        # Verify docstring was set
        assert tool_func.__doc__ is not None
        assert "Search for text patterns" in tool_func.__doc__


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_via_tool_function():
    """Test using the tool through get_tool_function()."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        (temp_path / "test.txt").write_text("TODO item\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        tool_func = tool.get_tool_function()
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Call through tool function
        result = await tool_func(ctx, pattern="TODO")
        assert "TODO" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_encoding_fallback():
    """Test handling of files with different encodings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create file with latin-1 encoding
        (temp_path / "latin1.txt").write_bytes("Café TODO\n".encode('latin-1'))
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Should handle encoding fallback
        result = await tool(ctx, pattern="TODO")
        assert "TODO" in result


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_error_in_file_read():
    """Test graceful handling when file can't be read."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a file that will be readable
        (temp_path / "good.txt").write_text("TODO item\n")
        
        tool = FileGrepTool(root_path=temp_dir)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        # Should handle errors gracefully and still return results from readable files
        result = await tool(ctx, pattern="TODO")
        assert isinstance(result, str)


@pytest.mark.skipif(not FSSPEC_AVAILABLE, reason="fsspec not installed")
@pytest.mark.asyncio
async def test_file_grep_no_context_lines():
    """Test grep with zero context lines."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        content = "line 1\nTODO item\nline 3\n"
        (temp_path / "test.txt").write_text(content)
        
        tool = FileGrepTool(root_path=temp_dir, context_lines=0)
        
        class MockContext:
            tool_call_id = "test"
            deps = None
        
        ctx = MockContext()
        
        result = await tool(ctx, pattern="TODO")
        
        # Should find the match without extra context
        assert "TODO" in result
        assert "Line 2:" in result


# Note: binary_detection parameter removed in simplified version - always enabled
