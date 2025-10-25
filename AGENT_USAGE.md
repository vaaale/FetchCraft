# Agents - Conversational AI with RAG

## Overview

The RAG framework now includes AI agents that can answer questions using retrieval-augmented generation. Agents use the retriever to find relevant information and then reason about it to provide answers.

## ReAct Agent

The `ReActAgent` implements the ReAct (Reasoning and Acting) pattern using pydantic-ai. It can search for information, reason about the results, and provide thoughtful answers.

## Installation

```bash
pip install pydantic-ai
```

Also ensure you have an OpenAI API key or access to an OpenAI-compatible LLM:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Basic Usage

```python
import asyncio
from qdrant_client import QdrantClient
from rag_framework import (
    OpenAIEmbeddings,
    QdrantVectorStore,
    VectorIndex,
    Node,
    ReActAgent,
    RetrieverTool
)

async def main():
    # 1. Setup embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = await embeddings.aget_dimension()
    
    # 2. Create knowledge base
    documents = [
        "Bill Gates was born on October 28, 1955.",
        "Bill Gates co-founded Microsoft in 1975.",
        "Python was created by Guido van Rossum in 1991.",
    ]
    
    nodes = [Node(text=text) for text in documents]
    
    # 3. Setup index
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="knowledge_base",
        vector_size=dimension
    )
    
    index = VectorIndex(
        vector_store=vector_store,
        embeddings=embeddings
    )
    
    await index.add_documents(nodes)
    
    # 4. Create retriever and tool
    retriever = index.as_retriever(top_k=2)
    retriever_tool = RetrieverTool.from_retriever(retriever)
    
    # 5. Create agent with retriever tool
    agent = ReActAgent.create(retriever_tool=retriever_tool)
    
    # 6. Query the agent
    answer = await agent.query("When was Bill Gates born?")
    print(answer)

asyncio.run(main())
```

## API Reference

### RetrieverTool.from_retriever()

```python
retriever_tool = RetrieverTool.from_retriever(
    retriever: Retriever,
    name: str = "search_documents",
    description: Optional[str] = None,
    formatter: Optional[Callable] = None
)
```

**Parameters**:
- `retriever`: A retriever instance (from `index.as_retriever()`)
- `name`: Name of the tool (default: "search_documents")
- `description`: Tool description for the LLM (optional)
- `formatter`: Custom formatter for results (optional)

### ReActAgent.create()

```python
agent = ReActAgent.create(
    retriever_tool: RetrieverTool,
    model: Union[str, Model] = "openai:gpt-4",
    system_prompt: Optional[str] = None,
    **agent_kwargs
)
```

**Parameters**:
- `retriever_tool`: A RetrieverTool instance (from `RetrieverTool.from_retriever()`)
- `model`: LLM model to use (default: "openai:gpt-4")
  - Format: `"provider:model-name"`
  - Examples: `"openai:gpt-4o-mini"`, `"openai:gpt-4"`, `"anthropic:claude-3-opus"`
- `system_prompt`: Custom system prompt for the agent
- `**agent_kwargs`: Additional arguments passed to pydantic-ai Agent

### Agent.query()

```python
answer = await agent.query(
    question: str,
    **kwargs
)
```

**Parameters**:
- `question`: The question to ask
- `**kwargs`: Additional parameters for the LLM

**Returns**: String response from the agent

## Custom System Prompts

You can customize the agent's behavior with system prompts:

```python
custom_prompt = """You are an expert technical assistant.

When answering questions:
1. Use the search_documents tool to find relevant information
2. Provide detailed technical explanations
3. Include code examples when appropriate
4. Cite your sources from the retrieved documents"""

retriever = index.as_retriever(top_k=5)
retriever_tool = RetrieverTool.from_retriever(retriever)

agent = ReActAgent.create(
    retriever_tool=retriever_tool,
    model="openai:gpt-4o-mini",
    system_prompt=custom_prompt
)
```

## Custom Formatters

You can customize how retrieval results are formatted:

```python
def custom_formatter(results):
    """Custom formatter for retrieval results."""
    if not results:
        return "No documents found."
    
    formatted = ["Here are the relevant documents:\n"]
    for i, (doc, score) in enumerate(results, 1):
        formatted.append(f"{i}. [{score:.1%} relevant] {doc.text[:100]}...")
    
    return "\n".join(formatted)

retriever_tool = RetrieverTool.from_retriever(
    retriever,
    formatter=custom_formatter
)

agent = ReActAgent.create(retriever_tool=retriever_tool)
```

## How It Works

The ReAct agent follows this pattern:

1. **Question**: User asks a question
2. **Think**: Agent decides what information it needs
3. **Act**: Agent calls `search_documents` tool to retrieve information
4. **Observe**: Agent examines the retrieved documents
5. **Reason**: Agent formulates an answer based on the documents
6. **Respond**: Agent provides the final answer

## Examples

### Example 1: Basic Q&A

```python
retriever = index.as_retriever(top_k=3)
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool)

answer = await agent.query("What is machine learning?")
# Agent searches documents, finds relevant information, and provides answer
```

### Example 2: Multi-Step Reasoning

```python
# Knowledge base with related facts
documents = [
    "Microsoft was founded in 1975.",
    "Bill Gates was CEO from 1975 to 2000.",
    "Steve Ballmer became CEO in 2000.",
]

# Create agent
retriever = index.as_retriever(top_k=3)
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool)

# Agent can combine multiple facts
answer = await agent.query("How long was Bill Gates the CEO of Microsoft?")
# Answer: "Bill Gates was CEO for 25 years (1975-2000)"
```

### Example 3: Handling Unknown Information

```python
answer = await agent.query("What is Bill Gates' favorite color?")
# Agent response: "I don't have information about Bill Gates' favorite
# color in the retrieved documents."
```

### Example 4: Custom Tool Name and Description

```python
retriever_tool = RetrieverTool.from_retriever(
    retriever,
    name="search_knowledge_base",
    description="Search the company knowledge base for relevant information"
)

agent = ReActAgent.create(retriever_tool=retriever_tool)
```

## Using Different Models

### OpenAI Models

```python
retriever_tool = RetrieverTool.from_retriever(retriever)

# GPT-4o mini (faster, cheaper)
agent = ReActAgent.create(retriever_tool=retriever_tool, model="openai:gpt-4o-mini")

# GPT-4 (more capable)
agent = ReActAgent.create(retriever_tool=retriever_tool, model="openai:gpt-4")

# GPT-3.5 Turbo (faster, cheaper)
agent = ReActAgent.create(retriever_tool=retriever_tool, model="openai:gpt-3.5-turbo")
```

### Anthropic Claude

```python
# Requires ANTHROPIC_API_KEY environment variable
retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool, model="anthropic:claude-3-opus-20240229")
```

### Custom Model Instance

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai

# Create custom model instance
model = OpenAIChatModel(
    "gpt-4-turbo",
    provider=OpenAIProvider(
        openai_client=openai.AsyncOpenAI(
            api_key="your-key",
            base_url="http://your-endpoint/v1"
        )
    )
)

retriever_tool = RetrieverTool.from_retriever(retriever)
agent = ReActAgent.create(retriever_tool=retriever_tool, model=model)
```

## Integration with VectorIndex

The agent seamlessly integrates with the VectorIndex:

```python
# Create index with embeddings
index = VectorIndex(
    vector_store=vector_store,
    embeddings=embeddings
)

# Add documents (auto-embedded)
await index.add_documents(nodes)

# Create retriever with specific settings
retriever = index.as_retriever(
    top_k=5,  # Return top 5 documents
    resolve_parents=True  # Resolve SymNode parents
)

# Create retriever tool
retriever_tool = RetrieverTool.from_retriever(retriever)

# Create agent with the retriever tool
agent = ReActAgent.create(retriever_tool=retriever_tool)

# Query
answer = await agent.query("Your question here")
```

## Advanced Features

### Streaming Responses

For long responses, you can stream the agent's output:

```python
# Using pydantic-ai's streaming capabilities
async for chunk in agent._agent.run_stream("Tell me about Python"):
    print(chunk, end="")
```

### Context Management

Agents automatically manage context through the retriever:

```python
# Agent will retrieve relevant context for each query
q1 = await agent.query("Who founded Microsoft?")
q2 = await agent.query("When did they found it?")
# Each query independently retrieves context
```

### Error Handling

```python
try:
    answer = await agent.query("What is the meaning of life?")
except Exception as e:
    print(f"Agent error: {e}")
    # Handle API errors, retrieval failures, etc.
```

## Best Practices

1. **Set appropriate top_k**: Balance between context and relevance
   ```python
   retriever = index.as_retriever(top_k=3)  # 3-5 is often good
   ```

2. **Use specific questions**: More specific questions get better answers
   ```python
   # Good: "When was Python first released?"
   # Less good: "Tell me about Python"
   ```

3. **Customize system prompts**: Tailor the agent's behavior to your use case
   ```python
   agent = ReActAgent.create(
       retriever=retriever,
       system_prompt="You are a medical assistant. Always cite sources."
   )
   ```

4. **Handle no-answer cases**: Design prompts to handle missing information
   ```python
   # Default prompt already includes:
   # "If the information is not in the retrieved documents, say so"
   ```

5. **Use appropriate models**: Choose based on speed/cost/quality tradeoff
   ```python
   # For testing: gpt-3.5-turbo (fast, cheap)
   # For production: gpt-4o-mini (balanced)
   # For complex reasoning: gpt-4 (best quality)
   ```

## Limitations

1. **Depends on retrieval quality**: Agent answers are only as good as retrieved documents
2. **LLM limitations**: Subject to the capabilities and limitations of the chosen LLM
3. **API costs**: Each query consumes LLM API tokens
4. **Async only**: Agents are async-first (use `agent.query_sync()` for sync if needed)

## Comparison with Direct Retrieval

### Direct Retrieval
```python
results = await retriever.retrieve("query")
# Returns raw documents - you interpret them
```

### Agent-Based
```python
answer = await agent.query("query")
# Returns natural language answer based on documents
```

**Use agents when**:
- You want natural language answers
- You need multi-step reasoning
- You want to combine information from multiple documents

**Use direct retrieval when**:
- You want raw documents
- You're building custom processing logic
- You want to avoid LLM API costs

## Examples

See `src/examples/agent_example.py` for complete working examples including:
- Basic agent usage
- Custom system prompts
- Multi-step reasoning
- Error handling

## Dependencies

- `pydantic-ai>=0.0.14`
- `openai` (for OpenAI models)
- Valid API key for your chosen LLM provider

## Future Enhancements

Potential additions:
- Conversation history/memory
- Tool-augmented agents (web search, calculator, etc.)
- Multi-agent collaboration
- Custom tool registration
- Caching for repeated queries
