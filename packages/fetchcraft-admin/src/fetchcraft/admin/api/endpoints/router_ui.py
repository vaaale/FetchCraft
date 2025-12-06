"""UI router for serving the frontend application."""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ui"])

# Navigate from src/fetchcraft/admin/api/endpoints to package root
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"


@router.get("/")
async def serve_index():
    """Serve the main index.html file."""
    if not FRONTEND_DIST.exists():
        return JSONResponse({
            "message": "Fetchcraft Admin API V2 is running",
            "note": "Frontend not built",
            "api_docs": "/docs",
        })
    
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {"message": "Frontend not built"},
        status_code=404,
    )


@router.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve SPA - return index.html for all non-API routes."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    if not FRONTEND_DIST.exists():
        return JSONResponse({
            "message": "Fetchcraft Admin API V2 is running",
            "note": "Frontend not built",
            "api_docs": "/docs",
        })
    
    file_path = FRONTEND_DIST / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    
    return JSONResponse(
        {"message": "Frontend not built"},
        status_code=404,
    )
