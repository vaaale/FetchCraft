import json
from pathlib import Path

from fastmcp import FastMCP

from fetchcraft.mcp.find_files_service import FindFilesService


def _get_frontend_asset_filenames() -> tuple[str | None, str | None]:
    """Get the filenames of built frontend assets (CSS and JS)."""
    package_dir = Path(__file__).parent.parent.parent.parent
    dist_dir = package_dir / "frontend" / "dist"
    assets_dir = dist_dir / "assets"
    
    css_filename = None
    js_filename = None
    
    if assets_dir.exists():
        css_files = list(assets_dir.glob("index-*.css"))
        if css_files:
            css_filename = css_files[0].name
        
        js_files = list(assets_dir.glob("index-*.js"))
        if js_files:
            js_filename = js_files[0].name
    
    return css_filename, js_filename


def add_tools(mcp: FastMCP, find_files_service: FindFilesService, server_url: str = "http://localhost:8003"):
    """Add MCP tools to the server."""
    
    @mcp.tool()
    async def find_files(
        query: str,
        num_results: int = 10,
        offset: int = 0
    ) -> str:
        """
        Find files using semantic search.

        This tool searches for files based on semantic similarity to the query.
        It uses vector embeddings to find relevant files.

        Args:
            query: The search query to find relevant files
            num_results: Number of results to return (1-100)
            offset: Offset for pagination (starting from 0)

        Returns:
            Dictionary containing the list of matching files, total count, and offset
            If format_html=True, returns a dictionary with 'html' key containing the formatted HTML
        """
        try:
            paginated_nodes = await find_files_service.find_files(
                query=query,
                num_results=num_results,
                offset=offset
            )

            # Convert nodes to file results
            files = []
            for node in paginated_nodes:
                # Get filename from metadata
                source = node.node.metadata.get("source", "Unknown")
                filename = node.node.metadata.get("filename", Path(source).name)

                # Get a sample. Using n first paragraphs
                text = node.node.text.replace("\r\n", "\n")
                paragraphs = text.split("\n\n")
                preview = "\n\n".join(paragraphs[:5])

                files.append({
                    "filename": filename,
                    "source": source,
                    "score": float(node.score) if node.score else 0.0,
                    "text_preview": (
                        preview + f" ....\n({max(len(paragraphs) - 5, 0)} more paragraphs)"
                    )
                })

            # Prepare data for frontend
            data = {
                "query": query,
                "files": files,
                "total": len(paginated_nodes),
                "offset": offset,
                "serverUrl": server_url
            }
            
            # Get frontend asset filenames for external loading
            css_filename, js_filename = _get_frontend_asset_filenames()
            
            # Escape the JSON data to prevent issues with </script> tags in content
            json_data = json.dumps(data, ensure_ascii=False)
            
            # Build asset URLs
            css_url = f"{server_url}/assets/{css_filename}" if css_filename else ""
            js_url = f"{server_url}/assets/{js_filename}" if js_filename else ""
            
#             html_content = f"""<!doctype html>
# <html lang="en">
#   <head>
#     <meta charset="UTF-8" />
#     <meta name="viewport" content="width=device-width, initial-scale=1.0" />
#     <title>Fetchcraft File Finder</title>
#     <link rel="stylesheet" crossorigin href="{css_url}">
#     <script>
#       window.__SEARCH_RESULTS__ = {json_data};
#     </script>
#   </head>
#   <body>
#     <div id="root"></div>
#     <script type="module" crossorigin src="{js_url}"></script>
#   </body>
# </html>"""

            html_content = f"""
<html lang="en">
    <head>
    <meta http-equiv="refresh" content="0; url=http://localhost:8003/find-files?query={query}&num_results={num_results}&offset={offset}">
    </head>
    <body>
    </body>
</html>"""


            # Format for MCP with artifact syntax
            response_header = f"Your search results for query: {query}"
            search_id = "_".join(query.split())
            artifact_header = f':::artifact{{identifier="{search_id}" type="text/html" title="File Search Results"}}'
            result = f"{response_header}\n{artifact_header}\n```html\n{html_content}\n```\n"

            return result
        except Exception as e:
            raise RuntimeError(f"Error searching files: {str(e)}")
