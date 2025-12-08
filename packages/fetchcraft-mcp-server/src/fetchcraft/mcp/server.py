import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from fetchcraft.mcp.api import create_api_router
from fetchcraft.mcp.find_files_service import FindFilesService
from fetchcraft.mcp.frontend_router import create_frontend_router, mount_static_assets
from fetchcraft.mcp.iframe_middleware import IframeHeadersMiddleware
from fetchcraft.mcp.mcp_api import add_tools
from fetchcraft.mcp.query_service import QueryService
from fetchcraft.mcp.settings import MCPServerSettings
from fetchcraft.mcp.setup import setup_logging


def configure_fetchcraft_mcp(app: FastAPI, mcp_name: str) -> FastAPI:
    # Add API router
    api_router = create_api_router(
        find_files_service=FindFilesService.create(settings),
        query_service=QueryService.create(settings)
    )
    app.include_router(api_router)

    # Add frontend router
    frontend_router = create_frontend_router()
    # Comment to remove from Tools
    # app.include_router(frontend_router)

    mcp = FastMCP.from_fastapi(app=app, name=mcp_name)
    
    # Construct server URL for frontend assets (use localhost for iframe compatibility)
    server_url = settings.frontend_base_url
    add_tools(
        mcp=mcp,
        find_files_service=FindFilesService.create(settings),
        server_url=server_url,
        mcp_server_name=settings.mcp_server_name
    )
    
    mcp_app = mcp.http_app(path='/mcp')

    combined_app = FastAPI(
        title=mcp_name,
        routes=[
            *mcp_app.routes,  # MCP routes
            *app.routes,  # Original API routes
            *frontend_router.routes
        ],
        lifespan=mcp_app.lifespan,
    )
    
    # Add CORS middleware to allow iframe access to assets
    combined_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for iframe compatibility
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    combined_app.add_middleware(IframeHeadersMiddleware)

    # Mount static assets for frontend
    mount_static_assets(combined_app)

    @combined_app.get("/health")
    async def health():
        return {"status": "how-bout-now"}

    return combined_app


load_dotenv()
setup_logging()

settings = MCPServerSettings()

def init_server(*args):
    app = FastAPI(title="FetchCraft API")

    combined_app = configure_fetchcraft_mcp(app, mcp_name="Fetchcraft Files MCP Server")

    # print environment
    print("Environment variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

    return combined_app


def main():
    """Entry point for the fetchcraft-mcp CLI command."""
    import uvicorn
    uvicorn.run("fetchcraft.mcp.server:init_server", host=settings.host, port=settings.port, factory=True)


if __name__ == "__main__":
    main()

