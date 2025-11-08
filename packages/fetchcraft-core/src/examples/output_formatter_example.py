"""
Example demonstrating different output formatters.

This example shows how to use DefaultOutputFormatter and OpenWebUIFormatter
to format agent responses with citations.
"""
import asyncio

from fetchcraft.agents.output_formatters import DefaultOutputFormatter, OpenWebUIFormatter
from fetchcraft.agents.model import CitationContainer, Citation
from fetchcraft.node import Chunk


def create_sample_citations() -> tuple[str, CitationContainer]:
    """Create sample response with citations for testing."""
    
    # Create citation container
    citations = CitationContainer()
    
    # Add sample citations
    node1 = Chunk.from_text(
        "Hybrid search combines dense vector search with sparse keyword matching...",
        chunk_index=0,
        metadata={
            "title": "Hybrid Search Documentation",
            "url": "https://docs.example.com/hybrid-search",
            "parsing": "documentation"
        }
    )
    # Store score in metadata since we can't set it directly
    node1.metadata["score"] = 0.95
    
    node2 = Chunk.from_text(
        "RAG systems use retrieval-augmented generation to improve LLM responses...",
        chunk_index=1,
        metadata={
            "title": "RAG Systems Guide",
            "url": "https://docs.example.com/rag-guide",
            "parsing": "guide"
        }
    )
    node2.metadata["score"] = 0.88
    
    node3 = Chunk.from_text(
        "Vector embeddings represent text as high-dimensional vectors...",
        chunk_index=2,
        metadata={
            "title": "Vector Embeddings Tutorial",
            "url": "https://docs.example.com/embeddings",
            "parsing": "tutorial"
        }
    )
    node3.metadata["score"] = 0.82
    
    # Add citations
    citation1 = citations.add(
        call_id="call_1",
        tool_name="search_documents",
        query="hybrid search",
        node=node1
    )
    
    citation2 = citations.add(
        call_id="call_2",
        tool_name="search_documents",
        query="RAG systems",
        node=node2
    )
    
    citation3 = citations.add(
        call_id="call_3",
        tool_name="search_documents",
        query="vector embeddings",
        node=node3
    )
    
    # Sample response with citations
    response = f"""# Understanding Hybrid Search

Hybrid search is a powerful technique that combines two complementary approaches for finding relevant information [Hybrid Search]({citation1.citation_id}).

## How It Works

The system uses [vector embeddings]({citation3.citation_id}) to understand semantic meaning, while also maintaining keyword-based search capabilities. This dual approach provides:

- **Better accuracy** for semantic queries
- **Improved recall** for specific terms and model numbers
- **Robustness** across different query types

## RAG Integration

[RAG systems]({citation2.citation_id}) leverage hybrid search to retrieve the most relevant context before generating responses. This ensures that the LLM has access to both semantically similar and keyword-matched documents.

### Code Example

```python
# Create hybrid retriever
retriever = index.as_retriever(
    top_k=5,
    enable_hybrid=True
)

# Query returns hybrid results
results = await retriever.retrieve("your query")
```

## Benefits

1. **Semantic understanding**: Captures meaning and context
2. **Keyword precision**: Matches exact terms and phrases
3. **Best of both worlds**: Combines strengths of each approach

This makes hybrid search ideal for technical documentation, product catalogs, and knowledge bases.
"""
    
    return response, citations


def demo_default_formatter():
    """Demonstrate DefaultOutputFormatter."""
    print("="*80)
    print("DEFAULT FORMATTER (Markdown)")
    print("="*80)
    print()
    
    response, citations = create_sample_citations()
    formatter = DefaultOutputFormatter()
    
    formatted = formatter.format(response, citations)
    
    print(formatted)
    print()
    print("-"*80)
    print(f"Citations used: {len(citations.citations)}")
    for i, citation in enumerate(citations.citations, 1):
        print(f"  [{i}] {citation.title} - {citation.url}")
    print()


def demo_openwebui_formatter():
    """Demonstrate OpenWebUIFormatter."""
    print("="*80)
    print("OPEN WEBUI FORMATTER (HTML)")
    print("="*80)
    print()
    
    response, citations = create_sample_citations()
    formatter = OpenWebUIFormatter()
    
    formatted = formatter.format(response, citations)
    
    print(formatted)
    print()
    print("-"*80)
    print(f"Citations used: {len(citations.citations)}")
    for i, citation in enumerate(citations.citations, 1):
        print(f"  [{i}] {citation.title} - {citation.url}")
    print()


def save_html_output():
    """Save HTML output to file for viewing in browser."""
    response, citations = create_sample_citations()
    formatter = OpenWebUIFormatter()
    
    formatted = formatter.format(response, citations)
    
    # Extract HTML from markdown code block
    # The formatter now returns: ```html\n<html>...</html>\n```
    import re
    html_match = re.search(r'```html\n(.*?)\n```', formatted, re.DOTALL)
    if html_match:
        html_doc = html_match.group(1)
    else:
        # Fallback if format is different
        html_doc = formatted
    
    output_file = "output_formatter_example.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_doc)
    
    print(f"✅ HTML output saved to: {output_file}")
    print(f"   Open it in a browser to see the formatted result!")


def main():
    """Run all demos."""
    print("\n")
    print("🎨 OUTPUT FORMATTER EXAMPLES")
    print("="*80)
    print()
    
    # Demo 1: Default formatter
    demo_default_formatter()
    
    print("\n\n")
    
    # Demo 2: OpenWebUI formatter
    demo_openwebui_formatter()
    
    print("\n\n")
    
    # Save HTML file
    print("="*80)
    print("SAVING HTML OUTPUT")
    print("="*80)
    print()
    save_html_output()
    
    print("\n")
    print("="*80)
    print("✨ DEMO COMPLETE!")
    print("="*80)
    print()
    print("Summary:")
    print("  • DefaultFormatter: Plain markdown with citation links")
    print("  • OpenWebUIFormatter: Rich HTML with styled citations and references")
    print("  • HTML file: saved for browser viewing")
    print()


if __name__ == "__main__":
    main()
