"""
HTML Formatter for Fetchcraft MCP Server

This module provides utilities to format JSON outputs into pretty HTML.
"""

from typing import Dict, List, Any


class FindFilesHTMLFormatter:
    """Formats find_files JSON output to pretty HTML."""

    @staticmethod
    def format(data: Dict[str, Any]) -> str:
        """
        Transform find_files JSON output to pretty HTML.

        Args:
            data: The JSON output from find_files containing:
                - files: List of file objects with filename, parsing, score, text_preview
                - total: Total number of results
                - offset: Pagination offset

        Returns:
            HTML string wrapped in triple backticks for chat applications
        """
        files = data.get("files", [])
        total = data.get("total", 0)
        offset = data.get("offset", 0)

        # Start building HTML
        html_parts = []
        
        # Add header
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Search Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 12px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 28px;
        }
        .stats {
            color: #666;
            font-size: 14px;
        }
        .file-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .file-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .filename {
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
            word-break: break-all;
        }
        .score-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .source {
            color: #718096;
            font-size: 13px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .source-icon {
            font-size: 16px;
        }
        .preview {
            background: #f7fafc;
            border-left: 3px solid #667eea;
            padding: 12px 15px;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.6;
            color: #4a5568;
            font-family: 'Courier New', monospace;
        }
        .no-results {
            background: white;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            color: #718096;
        }
        .no-results-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">""")

        # Add header with stats
        html_parts.append(f"""
        <div class="header">
            <h1>🔍 Search Results</h1>
            <div class="stats">
                Found <strong>{total}</strong> result{'s' if total != 1 else ''}{f' (showing {len(files)} starting from {offset})' if offset > 0 or len(files) < total else f' (showing {len(files)})'}
            </div>
        </div>""")

        # Add file cards
        if files:
            for idx, file_data in enumerate(files):
                filename = file_data.get("filename", "Unknown")
                source = file_data.get("source", "Unknown")
                score = file_data.get("score", 0.0)
                preview = file_data.get("text_preview", "No preview available")

                # Format score as percentage
                score_percent = round(score * 100, 1)
                
                html_parts.append(f"""
        <div class="file-card">
            <div class="file-header">
                <div class="filename">📄 {FindFilesHTMLFormatter._escape_html(filename)}</div>
                <div class="score-badge">{score_percent}% match</div>
            </div>
            <div class="source">
                <span class="source-icon">📍</span>
                <span>{FindFilesHTMLFormatter._escape_html(source)}</span>
            </div>
            <div class="preview">{FindFilesHTMLFormatter._escape_html(preview)}</div>
        </div>""")
        else:
            html_parts.append("""
        <div class="no-results">
            <div class="no-results-icon">🔍</div>
            <p>No files found matching your query.</p>
        </div>""")

        # Close HTML
        html_parts.append("""
    </div>
</body>
</html>""")

        # Join all parts and wrap in triple backticks
        html_content = "".join(html_parts)

        return f"```html\n{html_content}\n```"
        # return html_content

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters to prevent XSS."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
