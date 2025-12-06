"""API endpoints package."""
from fetchcraft.admin.api.endpoints.router_main import router as main_router
from fetchcraft.admin.api.endpoints.router_ui import router as ui_router

__all__ = ["main_router", "ui_router"]
