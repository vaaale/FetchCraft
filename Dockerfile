# =============================================================================
# Stage 1: Build Admin Frontend
# =============================================================================
FROM node:22-alpine AS admin-frontend-builder

WORKDIR /app/packages/fetchcraft-admin/frontend

# Copy package files first for better caching
COPY packages/fetchcraft-admin/frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY packages/fetchcraft-admin/frontend/ ./

# Build the frontend
RUN npm run build

# =============================================================================
# Stage 2: Build MCP Server Frontend
# =============================================================================
FROM node:22-alpine AS mcp-frontend-builder

WORKDIR /app/packages/fetchcraft-mcp-server/frontend

# Copy package files first for better caching
COPY packages/fetchcraft-mcp-server/frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY packages/fetchcraft-mcp-server/frontend/ ./

# Build the frontend
RUN npm run build

# =============================================================================
# Stage 3: Python Backend
# =============================================================================
FROM python:3.12-slim AS backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the entire project
COPY . .

# Copy built frontends from builder stages
COPY --from=admin-frontend-builder /app/packages/fetchcraft-admin/frontend/dist ./packages/fetchcraft-admin/frontend/dist
COPY --from=mcp-frontend-builder /app/packages/fetchcraft-mcp-server/frontend/dist ./packages/fetchcraft-mcp-server/frontend/dist

# Sync dependencies using uv
RUN uv sync --frozen

# Create Documents directory
RUN mkdir -p /api/Documents

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["uv", "run", "fetchcraft-openapi-server"]
