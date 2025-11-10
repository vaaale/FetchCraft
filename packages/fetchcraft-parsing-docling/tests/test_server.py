"""
Tests for the Docling parsing server.

These tests verify the server structure and basic functionality.
"""

import pytest
from fastapi.testclient import TestClient
from fetchcraft.parsing.docling.server import app
from fetchcraft.parsing.docling.models import HealthResponse, BatchParseResponse


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["name"] == "Docling Document Parsing API"


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    # Validate response structure
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "config" in data
    
    # Validate config fields
    config = data["config"]
    assert "max_concurrent_requests" in config
    assert "max_concurrent_files" in config
    assert "max_file_size_mb" in config
    assert "page_chunks" in config
    assert "do_ocr" in config
    assert "do_table_structure" in config


def test_openapi_schema(client):
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    # Check that our endpoints are documented
    assert "/parse" in schema["paths"]
    assert "/health" in schema["paths"]


def test_parse_endpoint_no_files(client):
    """Test parse endpoint with no files returns error."""
    response = client.post("/parse")
    assert response.status_code == 422  # Validation error


def test_cors_headers(client):
    """Test that CORS headers are present."""
    response = client.options("/health")
    # CORS middleware should handle OPTIONS requests
    assert response.status_code == 200


def test_response_models():
    """Test that response models can be instantiated."""
    # Test HealthResponse
    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        config={
            "max_concurrent_requests": 10,
            "max_concurrent_files": 4,
            "max_file_size_mb": 100,
            "page_chunks": True,
            "do_ocr": True,
            "do_table_structure": True
        }
    )
    assert health.status == "healthy"
    assert health.version == "1.0.0"
    
    # Test BatchParseResponse
    batch = BatchParseResponse(
        results=[],
        total_files=0,
        successful=0,
        failed=0,
        total_nodes=0,
        total_processing_time_ms=0.0
    )
    assert batch.total_files == 0
    assert batch.successful == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
