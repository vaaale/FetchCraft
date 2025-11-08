# Output Formatters Usage Guide

Output formatters control how agent responses are formatted, particularly how citations are rendered in the final output.

## Available Formatters

### 1. DefaultOutputFormatter

The default formatter that converts markdown-style citations to markdown links.

**Output Format**: Plain markdown text

**Usage**:
```python
from fetchcraft.agents import PydanticAgent, DefaultOutputFormatter

agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=DefaultOutputFormatter()
)
```

**Example Output**:
```markdown
Hybrid search combines dense and sparse vectors [Documentation](https://docs.example.com).

## Benefits
- Better accuracy
- Improved recall
```

### 2. OpenWebUIFormatter

An HTML formatter optimized for Open WebUI and web-based chat interfaces.

**Output Format**: Complete HTML document wrapped in markdown code block (` ```html ... ``` `)

**Features**:
- Returns full HTML page (DOCTYPE, head, body)
- Wrapped in markdown code block for Open WebUI compatibility
- Converts markdown to HTML
- Clickable citation links with superscript numbers
- References section with all sources
- Responsive styling with centered content
- Dark mode support
- Proper HTML escaping for safety
- Can be saved directly as .html file

**Usage**:
```python
from fetchcraft.agents import PydanticAgent, OpenWebUIFormatter

agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=OpenWebUIFormatter()
)
```

**Example Output**:
````markdown
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Response</title>
    <style>
    /* CSS styles embedded here */
    body { font-family: sans-serif; padding: 2rem; background: #f9f9f9; }
    .rag-response { max-width: 900px; margin: 0 auto; background: white; padding: 2rem; }
    </style>
</head>
<body>
    <div class="rag-response">
        <div class="response-content">
            <p>Hybrid search combines dense and sparse vectors 
            <a href="https://docs.example.com" class="citation-link" target="_blank">
                Documentation<sup>[1]</sup>
            </a>.</p>
            
            <h2>Benefits</h2>
            <ul>
                <li>Better accuracy</li>
                <li>Improved recall</li>
            </ul>
        </div>
        
        <div class="references-section">
            <h3 class="references-title">📚 References</h3>
            <ol class="references-list">
                <li class="reference-item">
                    <a href="https://docs.example.com" target="_blank" class="reference-link">
                        Documentation
                    </a>
                    <span class="source-info">(Tool: search_documents, Relevance: 0.95)</span>
                </li>
            </ol>
        </div>
    </div>
</body>
</html>
```
````

## Using with FastAPI Demo

### For Open WebUI Integration

```python
from fetchcraft.agents import PydanticAgent, OpenWebUIFormatter, RetrieverTool
from pydantic_ai import Tool

# Create retriever tool
retriever_tool = RetrieverTool.from_retriever(retriever)
tool_func = retriever_tool.get_tool_function()
tools = [Tool(tool_func, takes_ctx=True, max_retries=3)]

# Create agent with HTML formatter
agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=OpenWebUIFormatter()  # Use HTML formatter
)

# Query returns HTML-formatted response
response = await agent.query("What is hybrid search?")
print(response.response.content)  # HTML output
```

### FastAPI Endpoint Example

```python
from fastapi import FastAPI
from fetchcraft.agents import PydanticAgent, OpenWebUIFormatter

app = FastAPI()

# Global agent with HTML formatter
agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=OpenWebUIFormatter()
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    response = await agent.query(request.messages[-1].content)
    
    # Response is already HTML-formatted
    return ChatCompletionResponse(
        choices=[
            ChatCompletionChoice(
                message=Message(
                    role="assistant",
                    content=response.response.content  # HTML content
                )
            )
        ]
    )
```

## Creating Custom Formatters

You can create your own output formatter by extending the `OutputFormatter` base class:

```python
from fetchcraft.agents.output_formatters import OutputFormatter
from fetchcraft.agents.model import CitationContainer


class CustomFormatter(OutputFormatter):
    """Custom formatter example."""

    def format(self, response: str, citations: CitationContainer) -> str:
        """
        Format the response with citations.
        
        Args:
            response: Raw response text with markdown-style citations
            citations: Container with all available citations
            
        Returns:
            Formatted response string
        """
        # Your custom formatting logic here
        formatted = response

        # Process citations
        for citation in citations.all_citations:
            # Track which citations are used
            if should_include_citation(citation):
                citations.add_cited(citation)
                # Format citation as needed
                formatted = format_citation(formatted, citation)

        return formatted
```

### Custom Formatter Example: JSON Output

```python
import json
import re
from fetchcraft.agents.output_formatters import OutputFormatter
from fetchcraft.agents.model import CitationContainer, Citation


class JSONFormatter(OutputFormatter):
    """Formats output as JSON with structured citations."""

    def format(self, response: str, citations: CitationContainer) -> str:
        used_citations = []

        # Extract citations from response
        def extract_citation(match):
            citation_id = int(match.group("citation_id"))
            citation = citations.citation(citation_id)
            if citation:
                used_citations.append({
                    "id": citation.citation_id,
                    "title": citation.title,
                    "url": citation.url,
                    "score": citation.node.score
                })
                citations.add_cited(citation)
            return match.group("title")  # Remove citation marker

        # Clean response
        clean_response = re.sub(
            r"\[(?P<title>.*?)\]\((?P<citation_id>\d+)\)",
            extract_citation,
            response
        )

        # Return as JSON
        return json.dumps({
            "response": clean_response,
            "citations": used_citations
        }, indent=2)


# Usage
agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=JSONFormatter()
)
```

## Formatter Comparison

| Feature | DefaultFormatter | OpenWebUIFormatter | Custom |
|---------|------------------|-------------------|--------|
| Output Format | Markdown | HTML | Your choice |
| Citation Style | Links | Superscript + refs | Customizable |
| Styling | None | Embedded CSS | Your styles |
| Web Compatible | Partial | Full | Depends |
| References Section | No | Yes | Optional |
| Dark Mode | N/A | Yes | Optional |
| Best For | CLI, Markdown | Web UIs, Open WebUI | Specific needs |

## Tips

### 1. Choosing the Right Formatter

- **DefaultFormatter**: Use for CLI tools, markdown files, or simple text output
- **OpenWebUIFormatter**: Use for web interfaces, Open WebUI, or HTML rendering
- **Custom**: Create when you need specific formatting (JSON, XML, LaTeX, etc.)

### 2. Testing Formatters

```python
# Test with sample response
formatter = OpenWebUIFormatter()
citations = CitationContainer()

# Add test citation
citation = citations.add(
    call_id="test",
    tool_name="search",
    query="test query",
    node=test_node
)

# Test format
sample_response = f"This is a [test]({citation.citation_id}) citation."
output = formatter.format(sample_response, citations)
print(output)
```

### 3. Dynamic Formatter Selection

```python
def create_agent(output_format: str = "default"):
    if output_format == "html":
        formatter = OpenWebUIFormatter()
    elif output_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = DefaultOutputFormatter()
    
    return PydanticAgent.create(
        model="gpt-4-turbo",
        tools=tools,
        output_formatter=formatter
    )
```

## Open WebUI Specific Setup

When using with Open WebUI:

1. **Configure the agent** with `OpenWebUIFormatter()`
2. **Set content type** to HTML in responses
3. **Enable HTML rendering** in Open WebUI settings

```python
# In your FastAPI server
agent = PydanticAgent.create(
    model="gpt-4-turbo",
    tools=tools,
    output_formatter=OpenWebUIFormatter()
)

# Response will include:
# - Styled HTML content
# - Clickable citations
# - References section
# - Dark mode support
```

## Markdown Support in OpenWebUIFormatter

The HTML formatter supports these markdown elements:

- **Headers**: `# H1`, `## H2`, `### H3`
- **Bold**: `**text**` or `__text__`
- **Italic**: `*text*` or `_text_`
- **Code**: `` `inline` `` or ` ```block``` `
- **Lists**: `- item` or `* item`
- **Links**: `[text](url)` (converted to HTML)
- **Paragraphs**: Double line breaks

## Advanced: Citation Preprocessing

```python
class CustomOpenWebUIFormatter(OpenWebUIFormatter):
    """Extended HTML formatter with custom citation preprocessing."""
    
    def format(self, response: str, citations: CitationContainer) -> str:
        # Preprocess citations before formatting
        for citation in citations.all_citations:
            # Add custom metadata
            if not citation.title:
                citation.title = self._generate_title(citation)
            if not citation.url:
                citation.url = self._generate_url(citation)
        
        # Call parent format method
        return super().format(response, citations)
    
    def _generate_title(self, citation):
        """Generate a title from node content."""
        return citation.node.text[:50] + "..."
    
    def _generate_url(self, citation):
        """Generate a URL from metadata."""
        return f"#citation-{citation.citation_id}"
```

## Summary

- Use **DefaultOutputFormatter** for text-based interfaces
- Use **OpenWebUIFormatter** for web interfaces and Open WebUI
- Create **custom formatters** for specific output requirements
- Formatters are configured when creating the agent
- All formatters support proper citation tracking and rendering
