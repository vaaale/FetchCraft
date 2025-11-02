"""
Example demonstrating FileSearchTool and FileGrepTool with an agent.

This example shows how to use both file tools together to allow an agent
to search for files by name and by content.
"""
import asyncio
import os

from pydantic_ai import Agent, Tool

from fetchcraft.agents import FileSearchTool, FileGrepTool


async def main():
    """Run the file tools example."""
    
    # Configuration
    root_path = "docs"
    llm_model = os.getenv("LLM_MODEL", "openai:gpt-4-turbo")
    
    print("="*70)
    print("File Tools Example")
    print("="*70)
    print(f"Root path: {root_path}")
    print(f"Model: {llm_model}")
    print("="*70 + "\n")
    
    # Create file search tool (for finding files by name)
    file_search = FileSearchTool(
        root_path=root_path,
        name="find_files",
        max_results=20,
        recursive=True
    )
    
    # Create file grep tool (for finding files by content)
    file_grep = FileGrepTool(
        root_path=root_path,
        name="search_file_contents",
        max_results=10,
        max_matches_per_file=5,
        context_lines=2,
        recursive=True
    )
    
    # Create tools for the agent
    tools = [
        Tool(file_search.get_tool_function(), takes_ctx=True),
        Tool(file_grep.get_tool_function(), takes_ctx=True)
    ]
    
    # Create agent with both tools
    agent = Agent(
        model=llm_model,
        tools=tools,
        system_prompt="""You are a helpful assistant that can search for files by name and content.

Use the find_files tool to search for files by filename or path.
Use the search_file_contents tool to search within file contents.

When asked to find information:
1. First try to identify relevant files using find_files
2. Then search within those files using search_file_contents
3. Provide a clear summary of what you found

Always cite the specific files and line numbers where information was found.
"""
    )
    
    # Example queries
    queries = [
        "Find all Markdown files in the directory",
        "Search for any TODO comments in the codebase",
        "Find files that contain the word 'configuration' or 'config'",
        "Look for any markdown files and tell me what they're about based on their content"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        try:
            result = await agent.run(query)
            print(f"\n{result.output}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("="*70)
    print("Interactive mode - enter your queries (or 'quit' to exit)")
    print("="*70)
    
    while True:
        try:
            user_query = input("\nYour query: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            result = await agent.run(user_query)
            print(f"\n{result.output}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
