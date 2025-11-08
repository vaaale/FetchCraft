# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
COPY .env.example ./.env
COPY src/ ./src/

# Install Python dependencies
# First upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the package with all dependencies
RUN pip install --no-cache-dir -e .

# Create directories for documents and data
RUN mkdir -p /app/Documents

# Expose the FastAPI port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8001

# Run the server
CMD ["python", "-m", "demo.openapi_html_server.openapi_server"]
