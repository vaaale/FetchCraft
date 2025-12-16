"""
Frontend router for serving the web application.

This module provides routes for serving the built frontend application
and static assets.
"""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from importlib_resources import files


def create_frontend_router(frontend_dir: Path | None = None) -> APIRouter:
    """
    Create a router for serving the frontend application.
    
    Args:
        frontend_dir: Path to the frontend dist directory. If None, uses default location.
        
    Returns:
        Configured APIRouter with frontend routes
    """
    router = APIRouter(tags=["frontend"])
    frontend_dir = files("fetchcraft-mcp.frontend") / "dist"

    # Default frontend directory location
    if frontend_dir is None:
        package_root = Path(__file__).parent.parent.parent.parent
        frontend_dir = package_root / "frontend" / "dist"

    # Ensure frontend directory exists
    if not frontend_dir.exists():
        @router.get("/")
        async def frontend_not_built():
            return HTMLResponse(
                content="""
                <html>
                    <head><title>Frontend Not Built</title></head>
                    <body>
                        <h1>Frontend Not Built</h1>
                        <p>The frontend has not been built yet.</p>
                        <p>Please run the following commands:</p>
                        <pre>
cd frontend
npm install
npm run build
                        </pre>
                    </body>
                </html>
                """,
                status_code=503
            )
        return router
    
    # Serve the main index.html at root
    @router.get("/", response_class=HTMLResponse)
    async def serve_index():
        """Serve the frontend index.html."""
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            content="<h1>Frontend index.html not found</h1>",
            status_code=404
        )
    
    # Fallback for SPA routing - serve index.html for any unmatched routes
    # This allows client-side routing to work properly
    @router.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Serve index.html for client-side routing.
        
        This catches all GET requests that aren't handled by other routes
        and serves the index.html to support single-page application routing.
        """
        # Check if this is a request for a static file
        file_path = frontend_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Otherwise, serve index.html for SPA routing
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        
        return HTMLResponse(
            content="<h1>Frontend index.html not found</h1>",
            status_code=404
        )
    
    return router


class CORSStaticFiles(StaticFiles):
    """StaticFiles with CORS headers for iframe compatibility."""
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_with_cors(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"access-control-allow-origin", b"*"))
                    headers.append((b"access-control-allow-methods", b"GET, OPTIONS"))
                    headers.append((b"access-control-allow-headers", b"*"))
                    message["headers"] = headers
                await send(message)
            await super().__call__(scope, receive, send_with_cors)
        else:
            await super().__call__(scope, receive, send)


def mount_static_assets(app, frontend_dir: Path | None = None):
    """
    Mount static assets directory for serving CSS, JS, images, etc.
    
    Args:
        app: FastAPI application instance
        frontend_dir: Path to the frontend dist directory
    """
    if frontend_dir is None:
        frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    
    assets_dir = frontend_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", CORSStaticFiles(directory=str(assets_dir)), name="assets")
