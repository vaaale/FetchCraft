"""
Example: Using FileSearchTool with Agents

This example demonstrates how to use the FileSearchTool to give agents
controlled filesystem access for searching files.
"""

import asyncio
from pathlib import Path

from fetchcraft.agents import FileSearchTool

try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None


async def basic_usage():
    """
    Basic usage of FileSearchTool.
    """
    print("=" * 60)
    print("Basic FileSearchTool Usage")
    print("=" * 60)
    
    # Create tool with local filesystem access
    tool = FileSearchTool(
        root_path="",  # Current directory
        name="search_files",
        max_results=10
    )
    
    # Create a mock context (for demonstration)
    class MockContext:
        tool_call_id = "test_call"
        deps = None
    
    ctx = MockContext()
    
    # Search for Python files
    print("\n1. Searching for Python files (*.py):")
    result = await tool(ctx, "*.py", name_only=True)
    print(result)
    
    # Search for specific pattern
    print("\n2. Searching for test files (test_*.py):")
    result = await tool(ctx, "test_*.py", name_only=True)
    print(result)
    
    # Search with full path pattern
    print("\n3. Searching in specific directory (src/**/*.py):")
    result = await tool(ctx, "src/**/*.py", name_only=False)
    print(result)


async def with_agent():
    """
    Using FileSearchTool with a pydantic-ai agent.
    """
    if not PYDANTIC_AI_AVAILABLE:
        print("pydantic-ai not installed. Skipping agent example.")
        return
    
    print("\n" + "=" * 60)
    print("FileSearchTool with Agent")
    print("=" * 60)
    
    # Create file search tool
    file_tool = FileSearchTool(
        root_path="",
        name="search_files",
        description="Search for files in the project directory",
        max_results=20,
        allowed_extensions=['.py', '.txt', '.md'],  # Only allow specific types
        recursive=True
    )
    
    # Create agent with file search capability
    agent = Agent(
        'openai:gpt-4',
        tools=[file_tool.get_tool_function()],
        system_prompt="""You are a helpful assistant with access to the filesystem.
You can search for files using glob patterns. Help users find files they need."""
    )
    
    # Ask agent to find files
    result = await agent.run("Find all Python test files in the project")
    print(f"\nAgent response: {result.data}")


async def multiple_tools():
    """
    Using FileSearchTool alongside other tools (e.g., RetrieverTool).
    """
    if not PYDANTIC_AI_AVAILABLE:
        print("pydantic-ai not installed. Skipping multi-tool example.")
        return
    
    print("\n" + "=" * 60)
    print("Multiple Tools: FileSearch + Document Retrieval")
    print("=" * 60)
    
    # Create file search tool
    file_tool = FileSearchTool(
        root_path="./documents",
        name="search_files",
        max_results=10
    )
    
    # Note: You would also create a RetrieverTool here
    # retriever_tool = RetrieverTool.from_retriever(...)
    
    # Create agent with multiple tools
    agent = Agent(
        'openai:gpt-4',
        tools=[
            file_tool.get_tool_function(),
            # retriever_tool.get_tool_function(),  # Add retriever
        ],
        system_prompt="""You are a research assistant with two capabilities:
1. Search for files in the filesystem
2. Search for relevant information in indexed documents

Use the appropriate tool based on the user's request."""
    )
    
    result = await agent.run("Find all markdown files")
    print(f"\nAgent response: {result.data}")


async def constrained_search():
    """
    Using FileSearchTool with constraints for security.
    """
    print("\n" + "=" * 60)
    print("Constrained File Search (Security)")
    print("=" * 60)
    
    # Create tool with strict constraints
    tool = FileSearchTool(
        root_path="/data/public",  # Only allow access to specific directory
        allowed_extensions=['.txt', '.md', '.pdf'],  # Only allow specific file types
        max_results=20,  # Limit results
        recursive=True  # Allow recursive search within root
    )
    
    print(f"\nTool configured with:")
    print(f"  Root: {tool.root_path}")
    print(f"  Allowed extensions: {tool.allowed_extensions}")
    print(f"  Max results: {tool.max_results}")
    print(f"  Recursive: {tool.recursive}")
    
    # The tool will ONLY search within root_path
    # and ONLY return files with allowed extensions
    # This provides controlled access for the agent


async def custom_formatter():
    """
    Using FileSearchTool with custom result formatter.
    """
    print("\n" + "=" * 60)
    print("Custom Formatter")
    print("=" * 60)
    
    def custom_format(results):
        """Custom formatter that shows file sizes in MB."""
        if not results:
            return "No files found."
        
        output = [f"Found {len(results)} file(s):\n"]
        for result in results:
            size_mb = result.size / (1024 * 1024) if result.size else 0
            output.append(f"  📄 {result.path} ({size_mb:.2f} MB)")
        
        return "\n".join(output)
    
    # Create tool with custom formatter
    tool = FileSearchTool(
        root_path="",
        formatter=custom_format,
        max_results=5
    )
    
    class MockContext:
        tool_call_id = "test"
        deps = None
    
    ctx = MockContext()
    result = await tool(ctx, "*.py")
    print(result)


async def practical_example():
    """
    Practical example: Agent finds and lists configuration files.
    """
    print("\n" + "=" * 60)
    print("Practical Example: Finding Configuration Files")
    print("=" * 60)
    
    # Create tool for finding config files
    config_tool = FileSearchTool(
        root_path="",
        name="find_config",
        description="Find configuration files in the project",
        allowed_extensions=['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
        max_results=50
    )
    
    class MockContext:
        tool_call_id = "config_search"
        deps = None
    
    ctx = MockContext()
    
    # Search for common config file patterns
    patterns = ['*.json', '*.yaml', '*.toml', '*.ini']
    
    for pattern in patterns:
        print(f"\nSearching for {pattern}:")
        result = await config_tool(ctx, pattern)
        print(result)


async def main():
    """Run all examples."""
    
    # Basic usage
    await basic_usage()
    
    # Custom formatter
    await custom_formatter()
    
    # Constrained search (demonstration)
    # await constrained_search()  # Requires specific directory
    
    # Practical example
    await practical_example()
    
    # Agent examples (require pydantic-ai)
    # await with_agent()
    # await multiple_tools()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
