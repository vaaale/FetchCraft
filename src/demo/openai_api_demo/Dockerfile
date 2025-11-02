FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY src/demo/fastapi_demo/requirements.txt /app/requirements-demo.txt
RUN pip install --no-cache-dir -r requirements-demo.txt

# Copy the entire src directory
COPY src/ /app/src/

# Create Documents directory
RUN mkdir -p /app/Documents

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "demo.fastapi_demo.server"]
