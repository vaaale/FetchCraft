# Use Python 3.12 slim image
FROM python:3.12-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the entire project
COPY . .

# Sync dependencies using uv
RUN uv sync --frozen

# Create Documents directory
RUN mkdir -p /app/Documents

# Expose the default port
#EXPOSE 8001

# Set environment variables
#ENV HOST=0.0.0.0
#ENV PORT=8001
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["uv", "run", "fetchcraft-openapi-server"]
