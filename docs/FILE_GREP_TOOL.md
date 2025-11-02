# FileGrepTool Documentation

A powerful tool for searching file contents using grep-like functionality with fsspec filesystem support.

## Overview

`FileGrepTool` enables AI agents to search for text patterns within files, similar to the Unix `grep` command. It supports:

- **Text and regex patterns**: Search for literal text or complex regular expressions
- **Case-sensitive/insensitive search**: Flexible search options
- **Context lines**: Shows lines before and after matches for better understanding
- **File filtering**: Search within specific file types or patterns
- **Binary file detection**: Automatically skips binary files
- **File size limits**: Prevents loading huge files
- **Multiple filesystem support**: Works with local, S3, GCS, and other fsspec-compatible filesystems

## Features

- ✅ **Grep-like functionality**: Search file contents with patterns
- ✅ **Regular expression support**: Complex pattern matching
- ✅ **Context lines**: See surrounding lines for better context
- ✅ **File filtering**: Use glob patterns to filter which files to search
- ✅ **Safe defaults**: Binary detection, file size limits, encoding handling
- ✅ **Configurable limits**: Control max results, matches per file, context lines
- ✅ **fsspec integration**: Works with any fsspec-compatible filesystem

## Installation

```bash
# FileGrepTool requires fsspec and pydantic-ai
pip install fsspec pydantic-ai
```

## Basic Usage

### Simple Text Search

```python
from fetchcraft.agents import FileGrepTool

# Create tool for local filesystem
tool = FileGrepTool(
    root_path="/path/to/documents",
    context_lines=2  # Show 2 lines before/after each match
)

# Use with an agent
from pydantic_ai import Agent, Tool

agent = Agent(
    model="openai:gpt-4",
    tools=[Tool(tool.get_tool_function(), takes_ctx=True)]
)

# Agent can now search file contents
result = await agent.run("Find all TODO comments in Python files")
```

### Direct Usage (Without Agent)

```python
import asyncio
from fetchcraft.agents import FileGrepTool

async def search_files():
    tool = FileGrepTool(root_path="/path/to/code")
    
    # Mock context for testing
    class MockContext:
        tool_call_id = "test"
        deps = None
    
    # Search for pattern
    result = await tool(
        ctx=MockContext(),
        pattern="TODO",
        file_pattern="*.py",  # Only search Python files
        case_sensitive=False,
        regex=False
    )
    print(result)

asyncio.run(search_files())
```

## Configuration Options

### Constructor Parameters

```python
FileGrepTool(
    root_path=".",                      # Root directory to search
    fs=None,                            # Optional fsspec filesystem
    name="grep_files",                  # Tool name for the agent
    description=None,                   # Custom description
    formatter=None,                     # Custom result formatter
    max_results=50,                     # Max files to return results from
    max_matches_per_file=10,            # Max matches per file
    context_lines=2,                    # Lines before/after match
    allowed_extensions=None,            # e.g., ['.py', '.txt']
    recursive=True,                     # Search subdirectories
    binary_detection=True,              # Skip binary files
    max_file_size_mb=10.0              # Max file size to search
)
```

### Search Parameters

When calling the tool, you can specify:

```python
await tool(
    ctx=context,
    pattern="search_text",              # Pattern to search for
    file_pattern="*.py",                # Optional: filter files
    case_sensitive=False,               # Case sensitivity
    regex=False                         # Use regex pattern
)
```

## Examples

### Example 1: Find TODO Comments

```python
from fetchcraft.agents import FileGrepTool

tool = FileGrepTool(
    root_path="/project/src",
    allowed_extensions=['.py', '.js', '.ts']
)

# Find all TODO comments
result = await tool(
    ctx=context,
    pattern="TODO",
    case_sensitive=False
)
```

### Example 2: Regex Search

```python
# Find class definitions in Python files
result = await tool(
    ctx=context,
    pattern=r"class\s+\w+Agent",
    file_pattern="*.py",
    regex=True
)

# Find error patterns
result = await tool(
    ctx=context,
    pattern=r"(error|exception|failed)",
    file_pattern="**/*.log",
    regex=True,
    case_sensitive=False
)
```

### Example 3: Case-Sensitive Search

```python
# Find exact matches (e.g., environment variables)
result = await tool(
    ctx=context,
    pattern="API_KEY",
    case_sensitive=True
)
```

### Example 4: Search Specific File Types

```python
# Search only in markdown files
result = await tool(
    ctx=context,
    pattern="configuration",
    file_pattern="**/*.md"
)

# Search in multiple levels
result = await tool(
    ctx=context,
    pattern="import",
    file_pattern="src/**/*.py"
)
```

### Example 5: Remote Filesystems

```python
import fsspec

# Search in S3
s3_fs = fsspec.filesystem('s3', anon=False)
tool = FileGrepTool(
    root_path="my-bucket/documents",
    fs=s3_fs
)

# Search in Google Cloud Storage
gcs_fs = fsspec.filesystem('gcs')
tool = FileGrepTool(
    root_path="my-bucket/data",
    fs=gcs_fs
)
```

## Use Cases

### 1. Code Analysis

```python
# Find security issues
await tool(ctx, pattern=r"password\s*=\s*['\"]", regex=True)

# Find deprecated API usage
await tool(ctx, pattern="@deprecated", file_pattern="**/*.java")

# Find debug statements
await tool(ctx, pattern=r"console\.log|print\(", regex=True)
```

### 2. Documentation Search

```python
# Find configuration documentation
await tool(ctx, pattern="configuration", file_pattern="**/*.md")

# Find API references
await tool(ctx, pattern=r"API.*endpoint", regex=True, file_pattern="docs/**/*.md")
```

### 3. Log Analysis

```python
# Find error logs
await tool(ctx, pattern="ERROR", file_pattern="**/*.log", case_sensitive=True)

# Find specific request patterns
await tool(ctx, pattern=r"POST /api/.*", regex=True, file_pattern="access.log")
```

### 4. Configuration Management

```python
# Find environment variables
await tool(ctx, pattern=r"\$\{.*\}", regex=True, file_pattern="*.yaml")

# Find hardcoded values
await tool(ctx, pattern="localhost", case_sensitive=False)
```

## Output Format

The default formatter returns results in this format:

```
Found matches in 3 file(s) (15 total matches):

📄 src/agents/pydantic_agent.py (5 matches):
  Line 42: # TODO: Implement caching
    class PydanticAgent:
    # TODO: Implement caching
    def __init__(self):
  Line 156: # TODO: Add validation
    ... and 3 more matches

📄 src/tools/retriever_tool.py (10 matches):
  Line 23: # TODO: Support batch queries
    ... and 9 more matches
```

## Custom Formatting

You can provide a custom formatter:

```python
def custom_formatter(results):
    """Custom formatter for grep results."""
    output = []
    for file_result in results:
        output.append(f"File: {file_result.path}")
        for match in file_result.matches:
            output.append(f"  {match.line_number}: {match.match_text}")
    return "\n".join(output)

tool = FileGrepTool(
    root_path="/path",
    formatter=custom_formatter
)
```

## Combining with FileSearchTool

Use both tools together for powerful file operations:

```python
from fetchcraft.agents import FileSearchTool, FileGrepTool
from pydantic_ai import Agent, Tool

# Tool for finding files by name
file_search = FileSearchTool(root_path="/project")

# Tool for searching file contents
file_grep = FileGrepTool(root_path="/project")

# Create agent with both tools
agent = Agent(
    model="openai:gpt-4",
    tools=[
        Tool(file_search.get_tool_function(), takes_ctx=True),
        Tool(file_grep.get_tool_function(), takes_ctx=True)
    ],
    system_prompt="""You can search for files by name and by content.
    Use find_files to locate files by filename.
    Use grep_files to search within file contents."""
)

# Agent can now intelligently use both tools
result = await agent.run(
    "Find all Python test files and show me which ones test the Agent class"
)
```

## Performance Considerations

### File Size Limits

- Default max file size: 10 MB
- Large files are automatically skipped
- Adjust with `max_file_size_mb` parameter

### Binary Detection

- Binary files are automatically detected and skipped
- Uses null byte detection in first 8KB
- Disable with `binary_detection=False`

### Result Limits

- `max_results`: Limits number of files returned (default: 50)
- `max_matches_per_file`: Limits matches per file (default: 10)
- Prevents overwhelming output and excessive processing

### Encoding Handling

- Tries UTF-8 first, falls back to Latin-1
- Logs warnings for files that can't be decoded
- Skips files with persistent encoding issues

## Best Practices

1. **Use specific file patterns** to reduce search scope
2. **Set appropriate limits** based on your use case
3. **Use regex carefully** - test patterns before deploying
4. **Consider file size limits** for large codebases
5. **Combine with FileSearchTool** for comprehensive file operations
6. **Use context lines** to understand matches better
7. **Filter by extension** when searching specific file types

## Comparison: FileGrepTool vs FileSearchTool

| Feature | FileGrepTool | FileSearchTool |
|---------|--------------|----------------|
| **Searches** | File contents | File names/paths |
| **Pattern Type** | Text/Regex in content | Glob patterns for names |
| **Use Case** | Find text within files | Find files by name |
| **Returns** | Matches with line numbers | File paths with metadata |
| **Best For** | Code search, log analysis | File discovery, navigation |

## Troubleshooting

### No matches found

- Check the root_path is correct
- Verify file_pattern includes the files you want
- Test your regex pattern separately
- Check allowed_extensions filter

### Binary files being searched

- Ensure `binary_detection=True` (default)
- Some text files may be detected as binary if they contain null bytes

### Performance issues

- Reduce `max_results` and `max_matches_per_file`
- Use more specific `file_pattern`
- Reduce `max_file_size_mb`
- Use `allowed_extensions` to filter file types

### Encoding errors

- Files are automatically tried with UTF-8 and Latin-1
- Check logs for files that couldn't be decoded
- Some files may have unusual encodings

## Advanced Usage

### Custom Filesystem Backends

```python
# Search in ZIP archive
zip_fs = fsspec.filesystem('zip', fo='archive.zip')
tool = FileGrepTool(root_path=".", fs=zip_fs)

# Search via SFTP
sftp_fs = fsspec.filesystem('sftp', host='server.com', username='user')
tool = FileGrepTool(root_path="/remote/path", fs=sftp_fs)
```

### Integration with RAG Systems

```python
# Use grep results as context for RAG
grep_tool = FileGrepTool(root_path="/docs")
search_result = await grep_tool(ctx, pattern="authentication")

# Feed results to vector store or LLM
# ... (integrate with your RAG pipeline)
```

## API Reference

See the inline documentation in `file_grep_tool.py` for complete API details.

## See Also

- [FileSearchTool Documentation](FILE_SEARCH_TOOL.md)
- [Agent Tools Overview](AGENT_TOOLS.md)
- [fsspec Documentation](https://filesystem-spec.readthedocs.io/)
