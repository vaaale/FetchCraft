# RetrieverTool Update

**Date**: 2025-10-25  
**Change**: Agent API now uses `RetrieverTool` instead of direct `Retriever`

## Summary

The ReActAgent API has been updated to use `RetrieverTool` as an intermediate layer between retrievers and agents. This provides better separation of concerns and more flexibility.

## What Changed

### Before (Old API)

```python
retriever = index.as_retriever(top_k=2)
agent = ReActAgent.create(retriever=retriever)
```

### After (New API)

```python
retriever = index.as_retriever(top_k=2)
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool)
```

## Why This Change?

### Benefits

1. **Separation of Concerns**
   - Retrieval logic stays in `Retriever`
   - Tool integration logic in `RetrieverTool`
   - Agent logic in `ReActAgent`

2. **Customization**
   - Custom tool names
   - Custom descriptions for the LLM
   - Custom result formatters

3. **Reusability**
   - Same RetrieverTool can be used across multiple agents
   - Easy to create tool variations

4. **Extensibility**
   - Easier to add features to tools
   - Cleaner architecture for future enhancements

## RetrieverTool Features

### Basic Usage

```python
retriever = index.as_retriever(top_k=3)
retriever_tool = RetrieverTool.from_retriever(retriever)
```

### Custom Tool Name

```python
retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="search_knowledge_base"
)
```

### Custom Description

```python
retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="search_docs",
    description="Search the technical documentation for relevant information"
)
```

### Custom Formatter

```python
def custom_formatter(results):
    """Format results in a custom way."""
    if not results:
        return "No documents found."
    
    output = ["Found relevant documents:\n"]
    for i, (doc, score) in enumerate(results, 1):
        output.append(f"{i}. [{score:.0%}] {doc.text}")
    
    return "\n".join(output)

retriever_tool = RetrieverTool.from_retriever(
    retriever,
    formatter=custom_formatter
)
```

## Implementation Details

### RetrieverTool Class

**Location**: `src/rag_framework/agents/retriever_tool.py`

**Key Methods**:
- `from_retriever()` - Factory method to create from a retriever
- `get_tool_function()` - Returns async function for agent registration
- `__call__()` - Direct invocation of the tool

**Attributes**:
- `retriever` - The wrapped retriever
- `name` - Tool name (for LLM)
- `description` - Tool description (for LLM)
- `formatter` - Result formatter function

### Integration with ReActAgent

```python
class ReActAgent:
    def __init__(self, retriever_tool: RetrieverTool, ...):
        self.retriever_tool = retriever_tool
        
        # Create pydantic-ai agent
        self._agent = Agent(model=model, system_prompt=prompt)
        
        # Register the tool
        tool_func = self.retriever_tool.get_tool_function()
        self._agent.tool(tool_func)
```

## Migration Guide

### Step 1: Update Imports

```python
# Add RetrieverTool to imports
from rag_framework import (
    ReActAgent,
    RetrieverTool,  # Add this
    # ... other imports
)
```

### Step 2: Create RetrieverTool

```python
# Old code:
# agent = ReActAgent.create(retriever=retriever)

# New code:
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool)
```

### Step 3: Optional Customization

```python
# Take advantage of new features
retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="my_search_tool",
    formatter=my_custom_formatter
)
agent = ReActAgent.create(retriever_tool=retriever_tool)
```

## Examples

### Example 1: Basic Agent

```python
retriever = index.as_retriever(top_k=2)
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool)

answer = await agent.query("What is Python?")
```

### Example 2: Custom Formatter

```python
def json_formatter(results):
    """Format results as JSON-like structure."""
    import json
    docs = [{"text": doc.text, "score": score} for doc, score in results]
    return json.dumps(docs, indent=2)

retriever_tool = RetrieverTool.from_retriever(
    retriever,
    formatter=json_formatter
)
agent = ReActAgent.create(retriever_tool=retriever_tool)
```

### Example 3: Multiple Agents with Same Tool

```python
# Create tool once
retriever_tool = RetrieverTool.from_retriever(retriever)

# Use with different agents
agent1 = ReActAgent.create(
    retriever_tool=retriever_tool,
    model="openai:gpt-4",
    system_prompt="You are a technical expert..."
)

agent2 = ReActAgent.create(
    retriever_tool=retriever_tool,
    model="openai:gpt-3.5-turbo",
    system_prompt="You are a friendly assistant..."
)
```

### Example 4: Domain-Specific Tool

```python
def medical_formatter(results):
    """Format medical document results."""
    if not results:
        return "No medical literature found."
    
    formatted = ["Medical Literature Search Results:\n"]
    for i, (doc, score) in enumerate(results, 1):
        # Extract medical metadata
        specialty = doc.metadata.get("specialty", "General")
        year = doc.metadata.get("year", "Unknown")
        
        formatted.append(
            f"{i}. [{specialty}, {year}] "
            f"Relevance: {score:.0%}\n"
            f"   {doc.text[:200]}..."
        )
    
    return "\n".join(formatted)

retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="search_medical_literature",
    description="Search medical literature database for clinical information",
    formatter=medical_formatter
)

agent = ReActAgent.create(
    retriever_tool=retriever_tool,
    system_prompt="You are a medical research assistant..."
)
```

## Backwards Compatibility

**Breaking Change**: The old API (`retriever=retriever`) is no longer supported.

**Reason**: Clean architecture and better extensibility for future features.

**Migration Effort**: Minimal - just add one line to create the RetrieverTool.

## Files Changed

### Core Implementation
- ✅ `src/rag_framework/agents/retriever_tool.py` - NEW
- ✅ `src/rag_framework/agents/react_agent.py` - Updated to use RetrieverTool
- ✅ `src/rag_framework/agents/__init__.py` - Export RetrieverTool
- ✅ `src/rag_framework/__init__.py` - Export RetrieverTool

### Examples
- ✅ `src/examples/agent_example.py` - All examples updated

### Documentation
- ✅ `AGENT_USAGE.md` - Updated with RetrieverTool API
- ✅ `AGENT_IMPLEMENTATION.md` - Added RetrieverTool section
- ✅ `RETRIEVER_TOOL_UPDATE.md` - This document

## Testing

### Verification

```python
# Test RetrieverTool creation
retriever = index.as_retriever()
retriever_tool = RetrieverTool.from_retriever(retriever)
assert retriever_tool.name == "search_documents"

# Test tool function generation
tool_func = retriever_tool.get_tool_function()
assert callable(tool_func)

# Test direct invocation
result = await retriever_tool(None, "test query")
assert isinstance(result, str)

# Test with agent
agent = ReActAgent.create(retriever_tool=retriever_tool)
assert agent.retriever_tool == retriever_tool
```

## Future Enhancements

Potential additions to RetrieverTool:

1. **Multiple Retriever Support**
   ```python
   tool = RetrieverTool.from_retrievers([retriever1, retriever2])
   ```

2. **Result Filtering**
   ```python
   tool = RetrieverTool.from_retriever(
       retriever,
       filter_func=lambda doc, score: score > 0.8
   )
   ```

3. **Caching**
   ```python
   tool = RetrieverTool.from_retriever(
       retriever,
       cache=True,
       cache_ttl=3600
   )
   ```

4. **Metadata Extraction**
   ```python
   tool = RetrieverTool.from_retriever(
       retriever,
       extract_metadata=["author", "date", "source"]
   )
   ```

## Conclusion

The RetrieverTool abstraction provides:
- ✅ Cleaner separation of concerns
- ✅ Better customization options
- ✅ More extensible architecture
- ✅ Reusable tool instances
- ✅ Foundation for future enhancements

**Migration is straightforward**: Just add one line to create the RetrieverTool wrapper.
