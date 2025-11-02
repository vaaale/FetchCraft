"""
OpenWebUI HTML formatter for agent responses.

This formatter converts agent responses to HTML format optimized for Open WebUI,
including proper citation rendering with clickable links and a references section.
"""
import re
import html
from typing import List, Dict, Set

from .base import OutputFormatter
from ..model import CitationContainer, Citation


class OpenWebUIFormatter(OutputFormatter):
    """
    Formats agent responses as HTML for Open WebUI.
    
    Features:
    - Converts markdown-style citations to HTML links
    - Adds a references section with all cited sources
    - Basic markdown-to-HTML conversion (bold, italic, code blocks)
    - Proper HTML escaping for safety
    - Responsive styling compatible with Open WebUI
    """
    
    def format(self, response: str, citations: CitationContainer) -> str:
        """
        Format response as HTML with citations.
        
        Args:
            response: Raw response text with markdown-style citations
            citations: Container with all available citations
            
        Returns:
            HTML-formatted response string
        """
        # Track which citations are actually used
        used_citations: List[Citation] = []
        citation_map: Dict[int, int] = {}  # Maps original ID to display ID
        citation_placeholders: Dict[str, str] = {}  # Maps placeholder to HTML
        
        # Process citations - create placeholders first
        def create_citation_placeholder(match) -> str:
            citation_id = int(match.group("citation_id"))
            title = match.group("title")
            citation = citations.citation(citation_id)
            
            if not citation:
                return match.group(0)  # Return original if citation not found
            
            # Track this citation
            if citation not in used_citations:
                used_citations.append(citation)
                citations.add_cited(citation)
                display_id = len(used_citations)
                citation_map[citation_id] = display_id
            else:
                display_id = citation_map[citation_id]
            
            # Create unique placeholder (using special marker that won't be affected by markdown)
            placeholder = f"{{{{CITATION_{citation_id}_{display_id}}}}}"
            
            # Create HTML link and store it
            citation_title = html.escape(citation.title or title)
            citation_url = citation.url
            
            if citation_url:
                # Clickable link with superscript reference number
                citation_html = (
                    f'<a href="{html.escape(citation_url)}" '
                    f'class="citation-link" '
                    f'title="{citation_title}" '
                    f'target="_blank" '
                    f'rel="noopener noreferrer">'
                    f'{html.escape(title)}'
                    f'<sup>[{display_id}]</sup>'
                    f'</a>'
                )
            else:
                # No URL, just reference number
                citation_html = f'{html.escape(title)}<sup>[{display_id}]</sup>'
            
            citation_placeholders[placeholder] = citation_html
            return placeholder
        
        # Replace all citation patterns with placeholders
        formatted_response = re.sub(
            r"\[(?P<title>.*?)\]\((?P<citation_id>\d+)\)",
            create_citation_placeholder,
            response
        )
        
        # Apply basic markdown-to-HTML conversion
        formatted_response = self._markdown_to_html(formatted_response)
        
        # Replace placeholders with actual HTML citations
        for placeholder, citation_html in citation_placeholders.items():
            formatted_response = formatted_response.replace(placeholder, citation_html)
        
        # Build the final HTML with citations section
        html_parts = [
            '<div class="rag-response">',
            f'<div class="response-content">{formatted_response}</div>'
        ]
        
        # Add references section if there are citations
        if used_citations:
            html_parts.append(self._build_references_section(used_citations))
        
        html_parts.append('</div>')
        
        # Build complete HTML document
        html_body = '\n'.join(html_parts)
        html_document = self._build_html_document(html_body)
        
        # Wrap in markdown code block
        return f"```html\n{html_document}\n```"
    
    def _markdown_to_html(self, text: str) -> str:
        """
        Convert basic markdown to HTML.
        
        Supports:
        - Headers (# ## ###)
        - Bold (**text** or __text__)
        - Italic (*text* or _text_)
        - Code blocks (```code```)
        - Inline code (`code`)
        - Line breaks
        - Lists (- item)
        """
        # Extract placeholders to preserve them
        placeholders = {}
        placeholder_pattern = r'\{\{CITATION_\d+_\d+\}\}'
        
        def save_placeholder(match):
            key = f"XPLACEHOLDERX{len(placeholders)}XPLACEHOLDERX"
            placeholders[key] = match.group(0)
            return key
        
        text = re.sub(placeholder_pattern, save_placeholder, text)
        
        # Escape HTML
        text = html.escape(text)
        
        # Headers (must come before bold/italic to avoid conflicts)
        text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        
        # Code blocks
        text = re.sub(
            r'```(\w+)?\n(.*?)```',
            lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
            text,
            flags=re.DOTALL
        )
        
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Bold (must come before italic)
        text = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', text)
        
        # Italic
        text = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'_([^_]+)_', r'<em>\1</em>', text)
        
        # Unordered lists
        lines = text.split('\n')
        in_list = False
        html_lines = []
        
        for line in lines:
            if re.match(r'^\s*[-*]\s+', line):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                item_text = re.sub(r'^\s*[-*]\s+', '', line)
                html_lines.append(f'<li>{item_text}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        
        if in_list:
            html_lines.append('</ul>')
        
        text = '\n'.join(html_lines)
        
        # Paragraphs (double line breaks)
        text = re.sub(r'\n\n+', '</p><p>', text)
        
        # Single line breaks
        text = re.sub(r'\n', '<br>', text)
        
        # Wrap in paragraph if not already in block element
        if not re.match(r'^\s*<(h[1-6]|ul|ol|pre|div)', text):
            text = f'<p>{text}</p>'
        
        # Restore placeholders
        for key, placeholder in placeholders.items():
            text = text.replace(key, placeholder)

        return text
    
    def _build_references_section(self, citations: List[Citation]) -> str:
        """
        Build the references section HTML.
        
        Args:
            citations: List of citations to include
            
        Returns:
            HTML string for references section
        """
        refs_html = [
            '<div class="references-section">',
            '<h3 class="references-title">📚 References</h3>',
            '<ol class="references-list">'
        ]
        
        for citation in citations:
            title = html.escape(citation.title or f"Document {citation.citation_id}")
            url = citation.url
            
            # Extract source info
            source_info = []
            if citation.tool_name:
                source_info.append(f'Tool: {html.escape(citation.tool_name)}')
            
            # Get score from node attribute or metadata
            score = getattr(citation.node, 'score', None) or citation.node.metadata.get('score')
            if score is not None:
                source_info.append(f'Relevance: {score:.2f}')
            
            source_text = f' <span class="source-info">({", ".join(source_info)})</span>' if source_info else ''
            
            # Create reference entry
            if url:
                refs_html.append(
                    f'<li class="reference-item">'
                    f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" class="reference-link">'
                    f'{title}'
                    f'</a>'
                    f'{source_text}'
                    f'</li>'
                )
            else:
                refs_html.append(
                    f'<li class="reference-item">'
                    f'{title}'
                    f'{source_text}'
                    f'</li>'
                )
        
        refs_html.append('</ol>')
        refs_html.append('</div>')
        
        return '\n'.join(refs_html)
    
    def _build_html_document(self, body_content: str) -> str:
        """
        Build a complete HTML document with the body content.
        
        Args:
            body_content: HTML content for the body
            
        Returns:
            Complete HTML document as string
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Response</title>
    {self._add_styles()}
</head>
<body>
    {body_content}
</body>
</html>"""
    
    def _add_styles(self) -> str:
        """
        Add CSS styles for the HTML output.
        
        Returns:
            CSS style block as HTML string
        """
        return """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 2rem;
    background-color: #f9f9f9;
}

.rag-response {
    max-width: 900px;
    margin: 0 auto;
    background-color: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.response-content {
    margin-bottom: 2rem;
}

.response-content h1 {
    font-size: 1.8em;
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

.response-content h2 {
    font-size: 1.5em;
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

.response-content h3 {
    font-size: 1.3em;
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

.response-content p {
    margin: 0.5em 0;
}

.response-content ul, .response-content ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

.response-content li {
    margin: 0.3em 0;
}

.response-content code {
    background-color: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9em;
}

.response-content pre {
    background-color: #f5f5f5;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    margin: 1em 0;
}

.response-content pre code {
    background: none;
    padding: 0;
}

.citation-link {
    color: #0066cc;
    text-decoration: none;
    border-bottom: 1px dotted #0066cc;
}

.citation-link:hover {
    color: #0052a3;
    border-bottom: 1px solid #0052a3;
}

.citation-link sup {
    font-size: 0.75em;
    font-weight: bold;
    color: #0066cc;
    margin-left: 0.15em;
}

.references-section {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 2px solid #e0e0e0;
}

.references-title {
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #444;
}

.references-list {
    list-style-type: decimal;
    padding-left: 1.5em;
    margin: 0;
}

.reference-item {
    margin: 0.8em 0;
    line-height: 1.5;
}

.reference-link {
    color: #0066cc;
    text-decoration: none;
    font-weight: 500;
}

.reference-link:hover {
    text-decoration: underline;
}

.source-info {
    color: #666;
    font-size: 0.9em;
    font-style: italic;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    
    .rag-response {
        background-color: #2d2d2d;
        color: #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .response-content code {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    .response-content pre {
        background-color: #2d2d2d;
    }
    
    .references-section {
        border-top-color: #444;
    }
    
    .references-title {
        color: #e0e0e0;
    }
    
    .source-info {
        color: #999;
    }
}
</style>
"""
