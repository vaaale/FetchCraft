"""
Test script for FileGrepTool.
"""
import asyncio

import fsspec

from fetchcraft.agents import FileGrepTool


async def main():
    """Test FileGrepTool with various search patterns."""
    
    # Create filesystem
    fs = fsspec.filesystem("file")
    
    # Create grep tool
    tool = FileGrepTool(
        root_path="docs",
        fs=fs,
        context_lines=2,
        max_matches_per_file=5
    )
    
    # Mock context for testing
    class MockContext:
        tool_call_id = "test"
        deps = None
    
    ctx = MockContext()
    
    print("="*70)
    print("Test 1: Simple text search")
    print("="*70)
    result = await tool(ctx, pattern="TODO", file_pattern="*")
    print(result)
    print("\n")
    
    print("="*70)
    print("Test 2: Regex search")
    print("="*70)
    result = await tool(ctx, pattern=r"class\s+\w+Agent", regex=True, file_pattern="*.py")
    print(result)
    print("\n")
    
    print("="*70)
    print("Test 3: Case-sensitive search")
    print("="*70)
    result = await tool(ctx, pattern="Error", case_sensitive=True)
    print(result)
    print("\n")
    
    print("="*70)
    print("Test 4: Search in all files")
    print("="*70)
    result = await tool(ctx, pattern="import")
    print(result)


if __name__ == '__main__':
    asyncio.run(main())
