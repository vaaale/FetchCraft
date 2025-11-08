# Fetchcraft Docling Parsing Server

A FastAPI server for parsing documents using Docling. The server provides a REST API endpoint for uploading and parsing documents into structured `DocumentNode` objects.

## Features

- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, HTML, Images (with OCR), AsciiDoc, Markdown
- **Batch Processing**: Upload and parse multiple files in a single request
- **Configurable Concurrency**: Control concurrent requests and file processing
- **Advanced Document Understanding**: Layout analysis, table extraction, OCR capabilities
- **Structured Output**: Returns DocumentNode objects with metadata and hierarchical structure

## Installation

```bash
# Install the package
pip install -e .

# Or install with uv
uv pip install -e .
```

## Configuration

Configuration is done via environment variables. Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8080` | Server port |
| `MAX_CONCURRENT_REQUESTS` | `10` | Maximum concurrent requests |
| `MAX_CONCURRENT_FILES` | `4` | Maximum files processed in parallel |
| `MAX_FILE_SIZE_MB` | `100` | Maximum file size in MB |
| `PAGE_CHUNKS` | `true` | Split documents into separate pages |
| `DO_OCR` | `true` | Enable OCR for scanned documents |
| `DO_TABLE_STRUCTURE` | `true` | Extract table structure |

## Usage

### Start the Server

```bash
# Using the installed script
fetchcraft-docling-server

# Or run directly with Python
python -m fetchcraft.parsing.docling.server

# Or use uvicorn
uvicorn fetchcraft.parsing.docling.server:app --host 0.0.0.0 --port 8080
```

### API Endpoints

#### Parse Documents

**POST** `/parse`

Upload and parse one or more documents.

**Request**:
- Content-Type: `multipart/form-data`
- Body: One or more files with field name `files`

**Example with curl**:
```bash
# Parse a single file
curl -X POST http://localhost:8080/parse \
  -F "files=@document.pdf"

# Parse multiple files
curl -X POST http://localhost:8080/parse \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@presentation.pptx"
```

**Example with Python**:
```python
import requests

# Parse single file
with open("document.pdf", "rb") as f:
    files = {"files": f}
    response = requests.post("http://localhost:8080/parse", files=files)
    result = response.json()

# Parse multiple files
files = [
    ("files", open("document1.pdf", "rb")),
    ("files", open("document2.docx", "rb")),
]
response = requests.post("http://localhost:8080/parse", files=files)
result = response.json()
```

**Response**:
```json
{
  "results": [
    {
      "filename": "document.pdf",
      "success": true,
      "nodes": [
        {
          "id": "uuid-here",
          "node_type": "DocumentNode",
          "text": "Document content...",
          "metadata": {
            "page_number": 1,
            "total_pages": 10,
            "filename": "document.pdf",
            "file_size": 12345,
            "file_type": "pdf"
          }
        }
      ],
      "error": null,
      "num_nodes": 10,
      "processing_time_ms": 1234.56
    }
  ],
  "total_files": 1,
  "successful": 1,
  "failed": 0,
  "total_nodes": 10,
  "total_processing_time_ms": 1234.56
}
```

#### Health Check

**GET** `/health`

Check server status and configuration.

**Example**:
```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "config": {
    "max_concurrent_requests": 10,
    "max_concurrent_files": 4,
    "max_file_size_mb": 100,
    "page_chunks": true,
    "do_ocr": true,
    "do_table_structure": true
  }
}
```

#### Root

**GET** `/`

Get API information and supported formats.

```bash
curl http://localhost:8080/
```

## Supported File Formats

- **PDF** (`.pdf`) - Including scanned documents with OCR
- **Microsoft Word** (`.docx`)
- **Microsoft PowerPoint** (`.pptx`)
- **Microsoft Excel** (`.xlsx`)
- **HTML** (`.html`, `.htm`)
- **Images** (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`) - With OCR
- **AsciiDoc** (`.asciidoc`, `.adoc`)
- **Markdown** (`.md`)

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

## Concurrency Control

The server implements two levels of concurrency control:

1. **Request-Level Concurrency**: Limits the number of concurrent HTTP requests being processed
   - Configured via `MAX_CONCURRENT_REQUESTS`
   - Default: 10 concurrent requests

2. **File-Level Concurrency**: Limits the number of files being parsed simultaneously
   - Configured via `MAX_CONCURRENT_FILES`
   - Default: 4 concurrent files
   - Applies to files within the same request and across different requests

This two-tier approach ensures:
- The server can handle multiple clients simultaneously
- Resource-intensive parsing operations are throttled to prevent overload
- Optimal use of CPU and memory resources

## Error Handling

The server provides detailed error information for failed parsing operations:

```json
{
  "results": [
    {
      "filename": "corrupted.pdf",
      "success": false,
      "nodes": [],
      "error": "Failed to parse document: Invalid PDF structure",
      "num_nodes": 0,
      "processing_time_ms": 123.45
    }
  ],
  "total_files": 1,
  "successful": 0,
  "failed": 1,
  "total_nodes": 0,
  "total_processing_time_ms": 123.45
}
```

## Performance Tips

1. **Adjust Concurrency**: Tune `MAX_CONCURRENT_FILES` based on your CPU cores and available memory
2. **File Size Limits**: Set `MAX_FILE_SIZE_MB` appropriate to your use case
3. **Disable OCR**: Set `DO_OCR=false` if you don't need OCR for faster processing
4. **Batch Processing**: Upload multiple files in a single request for better throughput
5. **Page Chunks**: Set `PAGE_CHUNKS=false` if you want entire documents as single nodes

## Docker Support

You can containerize the server for easy deployment:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy application
COPY src/ src/

# Expose port
EXPOSE 8080

# Run server
CMD ["fetchcraft-docling-server"]
```

Build and run:
```bash
docker build -t docling-parser .
docker run -p 8080:8080 -e MAX_CONCURRENT_FILES=8 docling-parser
```

## Development

To run the server in development mode with auto-reload:

```bash
uvicorn fetchcraft.parsing.docling.server:app --reload --host 0.0.0.0 --port 8080
```

## License

MIT
