# Quick Start Guide

Get the Docling parsing server up and running in minutes.

## 1. Installation

```bash
cd packages/fetchcraft-parsing-docling
pip install -e .
```

## 2. Configuration (Optional)

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
# Edit .env to customize settings
```

## 3. Start the Server

```bash
fetchcraft-docling-server
```

The server will start on `http://localhost:8080` by default.

## 4. Test the Server

### Check Server Health

```bash
curl http://localhost:8080/health
```

### Parse a Document

```bash
# Single file
curl -X POST http://localhost:8080/parse \
  -F "files=@path/to/document.pdf"

# Multiple files
curl -X POST http://localhost:8080/parse \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@presentation.pptx"
```

### Using the Test Client

```bash
python test_client.py document1.pdf document2.docx
```

## 5. View API Documentation

Open your browser to:
- **Interactive Docs**: http://localhost:8080/docs
- **API Schema**: http://localhost:8080/openapi.json

## Example Output

When you parse a document, you'll receive a response like:

```json
{
  "results": [
    {
      "filename": "document.pdf",
      "success": true,
      "nodes": [
        {
          "id": "abc123",
          "node_type": "DocumentNode",
          "text": "Page content here...",
          "metadata": {
            "page_number": 1,
            "total_pages": 5,
            "filename": "document.pdf",
            "file_size": 45678,
            "file_type": "pdf",
            "width": 612.0,
            "height": 792.0
          }
        }
      ],
      "error": null,
      "num_nodes": 5,
      "processing_time_ms": 2345.67
    }
  ],
  "total_files": 1,
  "successful": 1,
  "failed": 0,
  "total_nodes": 5,
  "total_processing_time_ms": 2345.67
}
```

## Configuration Options

Customize behavior via environment variables in `.env`:

```bash
# Increase concurrent file processing for better performance
MAX_CONCURRENT_FILES=8

# Allow larger files
MAX_FILE_SIZE_MB=200

# Disable OCR if not needed (faster)
DO_OCR=false

# Get entire documents as single nodes instead of pages
PAGE_CHUNKS=false
```

## Next Steps

- See [README.md](README.md) for detailed documentation
- Check out the API docs at http://localhost:8080/docs
- Integrate the API into your application

## Troubleshooting

**Server won't start:**
- Check if port 8080 is already in use
- Try a different port: `PORT=8081 fetchcraft-docling-server`

**File parsing fails:**
- Check file format is supported
- Verify file is not corrupted
- Check file size is within limits

**Slow performance:**
- Increase `MAX_CONCURRENT_FILES` if you have more CPU cores
- Disable `DO_OCR` if OCR is not needed
- Set `PAGE_CHUNKS=false` for faster processing of small documents
