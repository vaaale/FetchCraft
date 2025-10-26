# Agent Implementation - ReAct Agent with pydantic-ai

## Overview

The RAG framework now includes an AI agent system using pydantic-ai, allowing for conversational question-answering with automatic retrieval augmentation.

## ✅ Implementation Complete

### Features Implemented

1. **BaseAgent Abstract Class** ✅
   - Abstract interface for all agent types
   - Async-first design with sync wrapper
   - Consistent API across agent implementations

2. **ReActAgent Implementation** ✅
   - Uses pydantic-ai for LLM integration
   - Automatic tool registration for retriever
   - Customizable system prompts
   - Support for multiple LLM providers (OpenAI, Anthropic, etc.)

3. **Seamless Integration** ✅
   - Works directly with VectorIndex retrievers
   - Leverages existing retrieval infrastructure
   - Supports SymNode parent resolution

4. **Documentation** ✅
   - Complete usage guide (`AGENT_USAGE.md`)
   - Working examples (`src/examples/agent_example.py`)
   - API reference and best practices

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User                              │
│                          │                               │
│                    agent.query("question")               │
└──────────────────────────┼──────────────────────────────┘
                           ▼
                    ┌──────────────┐
                    │  ReActAgent  │
                    │              │
                    │ - pydantic-ai│
                    │ - LLM model  │
                    └──────┬───────┘
                           │
                ┌──────────┼──────────┐
                ▼                     ▼
         ┌─────────────┐      ┌─────────────┐
         │  Retriever  │      │  LLM (GPT)  │
         │             │      │             │
         │ .retrieve() │      │  Reasoning  │
         └──────┬──────┘      └─────────────┘
                │
                ▼
         ┌─────────────┐
         │VectorIndex  │
         │             │
         │  Documents  │
         └─────────────┘
```

## Usage Example (As Requested)

```python
import asyncio
from rag_framework import (
    VectorIndex,
    ReActAgent,
    RetrieverTool,
    # ... other imports
)

async def main():
    # Setup index with documents
    index: VectorIndex = ...  # Your configured index
    
    # Create retriever and tool
    retriever = index.as_retriever(top_k=2)
    retriever_tool = RetrieverTool.from_retriever(retriever)
    
    # Create agent from retriever tool
    agent = ReActAgent.create(retriever_tool=retriever_tool)
    
    # Query the agent
    answer = await agent.query("How old was Bill Gates when he died?")
    print(answer)
    # Output: "Based on the retrieved documents, there is no information 
    #          indicating that Bill Gates has died. Bill Gates was born 
    #          on October 28, 1955, and is still alive."

asyncio.run(main())
```

## API

### Creating a RetrieverTool

```python
# From a retriever
retriever = index.as_retriever(top_k=2)
retriever_tool = RetrieverTool.from_retriever(retriever)

# With custom settings
retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="search_knowledge_base",  # Custom tool name
    description="Search for relevant documents",  # Custom description
    formatter=custom_formatter  # Custom result formatter
)
```

### Creating an Agent

```python
agent = ReActAgent.create(
    retriever_tool=retriever_tool,
    model="openai:gpt-4o-mini",  # Optional, default: "openai:gpt-4"
    system_prompt="Custom prompt"  # Optional
)
```

### Querying an Agent

```python
# Async
answer = await agent.query("Your question here")

# Sync (if no event loop is running)
answer = agent.query_sync("Your question here")
```

## File Structure

```
src/rag_framework/agents/
├── __init__.py              # Package exports
├── base.py                  # BaseAgent abstract class
├── retriever_tool.py        # RetrieverTool implementation
└── react_agent.py           # ReActAgent implementation

src/examples/
└── agent_example.py         # Complete working examples

Documentation/
├── AGENT_USAGE.md           # Complete usage guide
└── AGENT_IMPLEMENTATION.md  # This file
```

## Dependencies

Added to `pyproject.toml`:

```toml
dependencies = [
    ...
    "pydantic-ai>=0.0.14",
]
```

## How It Works

### RetrieverTool

The `RetrieverTool` is a wrapper around a `Retriever` that makes it compatible with pydantic-ai agents:

1. **Wraps Retriever**: Takes any `Retriever` instance
2. **Provides Tool Function**: Generates an async function that agents can call
3. **Formats Results**: Converts retrieval results into LLM-friendly text
4. **Customizable**: Supports custom names, descriptions, and formatters

**Key Features**:
- Separates retrieval logic from agent logic
- Allows custom formatting of results
- Enables tool name/description customization
- Reusable across different agents

### The ReAct Pattern

1. **Receive Question**: User asks a question
2. **Reason**: LLM decides what information is needed
3. **Act**: LLM calls `search_documents` tool via RetrieverTool
4. **Observe**: LLM examines retrieved documents
5. **Reason**: LLM formulates answer based on documents
6. **Respond**: Returns final answer to user

### Tool Registration

The agent automatically registers the RetrieverTool:

```python
# RetrieverTool generates the tool function
tool_func = retriever_tool.get_tool_function()

# Agent registers it
agent._agent.tool(tool_func)
```

The LLM can call this tool automatically when it needs information.

## Features

### ✅ Automatic Retrieval
- Agent decides when to search
- Automatically formats retrieved documents
- Handles no-results cases

### ✅ Multi-Step Reasoning
- Can make multiple retrieval calls
- Combines information from different sources
- Handles complex questions requiring synthesis

### ✅ Customizable Behavior
- Custom system prompts
- Different LLM models
- Configurable retrieval parameters

### ✅ Error Handling
- Graceful handling of missing pydantic-ai
- LLM API error handling
- Retrieval failure handling

## Examples

### Example 1: Basic Q&A

```python
agent = ReActAgent.create(retriever=index.as_retriever(top_k=3))
answer = await agent.query("What is machine learning?")
```

### Example 2: Complex Reasoning

```python
# Agent can combine multiple facts
answer = await agent.query(
    "How many years was Bill Gates the CEO of Microsoft?"
)
# Agent searches for: when he became CEO, when he stepped down
# Then calculates: 2000 - 1975 = 25 years
```

### Example 3: Custom System Prompt

```python
custom_prompt = """You are an expert teacher.
Explain concepts simply and include examples."""

agent = ReActAgent.create(
    retriever=retriever,
    system_prompt=custom_prompt
)
```

## Integration with Existing Features

### Works with SymNode

```python
# Create hierarchical index with SymNodes
parent = Chunk.from_text(text="Long document...")
sym_nodes = parent.create_symbolic_nodes([...])

await index.add_documents([parent])
await index.add_documents(sym_nodes)

# Create retriever with parent resolution
retriever = index.as_retriever(
    top_k=5,
    resolve_parents=True  # Returns parent chunks
)

# Agent gets full context from parent chunks
agent = ReActAgent.create(retriever=retriever)
```

### Works with Any Retriever

```python
# Any retriever implementation works
from rag_framework import Retriever

class CustomRetriever(Retriever):
    async def retrieve(self, query, **kwargs):
        # Your custom retrieval logic
        return results

agent = ReActAgent.create(retriever=CustomRetriever())
```

## LLM Provider Support

### OpenAI (Default)

```python
agent = ReActAgent.create(
    retriever=retriever,
    model="openai:gpt-4o-mini"
)
# Requires: OPENAI_API_KEY environment variable
```

### Anthropic Claude

```python
agent = ReActAgent.create(
    retriever=retriever,
    model="anthropic:claude-3-opus-20240229"
)
# Requires: ANTHROPIC_API_KEY environment variable
```

### Other Providers

pydantic-ai supports many providers. See their documentation for details.

## Best Practices

1. **Choose Appropriate top_k**
   ```python
   # Too few: May miss relevant info
   # Too many: Confuses the LLM
   retriever = index.as_retriever(top_k=3)  # 3-5 is often ideal
   ```

2. **Craft Good System Prompts**
   ```python
   # Be specific about expected behavior
   # Include citation requirements
   # Define handling of missing information
   ```

3. **Use Appropriate Models**
   ```python
   # Development: gpt-3.5-turbo (fast, cheap)
   # Production: gpt-4o-mini (balanced)
   # Complex tasks: gpt-4 (best quality)
   ```

4. **Handle Errors**
   ```python
   try:
       answer = await agent.query(question)
   except Exception as e:
       # Handle API errors, rate limits, etc.
       print(f"Error: {e}")
   ```

## Limitations

1. **Requires pydantic-ai**: Optional dependency
2. **Requires LLM API**: OpenAI or compatible service
3. **API Costs**: Each query consumes tokens
4. **Async-first**: Sync wrapper available but limited
5. **Retrieval Quality**: Answers limited by retrieved documents

## Testing

The implementation includes proper error handling for missing dependencies:

```python
# If pydantic-ai is not installed:
try:
    agent = ReActAgent.create(retriever=retriever)
except ImportError as e:
    print(e)  # "pydantic-ai is required..."
```

## Future Enhancements

Potential additions:
- **Memory/History**: Conversation context across queries
- **Multi-Agent**: Collaboration between specialized agents
- **Custom Tools**: Beyond just document retrieval
- **Streaming**: Stream responses as they're generated
- **Caching**: Cache LLM responses for repeated queries

## Complete Example

See `src/examples/agent_example.py` for:
- Basic agent usage
- Custom system prompts
- Multi-step reasoning
- Error handling
- Different use cases

## Summary

✅ **ReAct agent implemented and ready to use**  
✅ **Seamless integration with VectorIndex**  
✅ **Flexible and extensible architecture**  
✅ **Comprehensive documentation**  
✅ **Working examples provided**  

The agent system adds powerful conversational AI capabilities to the RAG framework while maintaining the clean, intuitive API design.
