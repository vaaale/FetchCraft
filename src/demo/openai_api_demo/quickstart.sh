#!/bin/bash

# FastAPI RAG Server Quick Start Script

set -e

echo "======================================================================"
echo "🚀 FastAPI RAG Server - Quick Start"
echo "======================================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check if Qdrant is running
echo "📦 Checking if Qdrant is running..."
if ! curl -s http://localhost:6333/health &> /dev/null; then
    echo "⚠️  Qdrant is not running. Starting Qdrant..."
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
    echo "⏳ Waiting for Qdrant to be ready..."
    sleep 5
    
    # Wait for Qdrant to be healthy
    for i in {1..30}; do
        if curl -s http://localhost:6333/health &> /dev/null; then
            echo "✅ Qdrant is ready!"
            break
        fi
        echo "   Still waiting... ($i/30)"
        sleep 2
    done
else
    echo "✅ Qdrant is already running"
fi

echo ""
echo "======================================================================"
echo "📂 Document Setup"
echo "======================================================================"
echo ""

# Create Documents directory if it doesn't exist
if [ ! -d "Documents" ]; then
    echo "📁 Creating Documents directory..."
    mkdir -p Documents
    
    # Create a sample document
    cat > Documents/sample.txt << EOF
# Sample Document for RAG Testing

## Hybrid Search

Hybrid search combines two approaches:
1. Dense vectors (semantic search) - understands meaning
2. Sparse vectors (keyword search) - matches exact terms

This combination provides better results than either approach alone.

## Benefits

- Better recall for technical terms and model numbers
- Improved semantic understanding
- Handles typos and synonyms better
- Works well with domain-specific vocabulary

## Use Cases

Hybrid search is particularly useful for:
- Technical documentation
- Legal documents
- Medical records
- Product catalogs with SKUs
EOF
    
    echo "✅ Created sample document in Documents/sample.txt"
else
    echo "✅ Documents directory already exists"
    
    # Count files
    file_count=$(find Documents -type f | wc -l)
    echo "   Found $file_count file(s) in Documents/"
fi

echo ""
echo "======================================================================"
echo "🔧 Installing Python Dependencies"
echo "======================================================================"
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment. It's recommended to use one."
    echo "   Create one with: python -m venv .venv && source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
if [ -f "src/demo/fastapi_demo/requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -q -r src/demo/openai_api_demo/requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  Requirements file not found. Installing minimal dependencies..."
    pip install -q fastapi uvicorn[standard] pydantic-ai qdrant-client openai fastembed
fi

echo ""
echo "======================================================================"
echo "🎬 Starting FastAPI Server"
echo "======================================================================"
echo ""

# Set default environment variables if not set
export DOCUMENTS_PATH=${DOCUMENTS_PATH:-Documents}
export ENABLE_HYBRID=${ENABLE_HYBRID:-true}
export FUSION_METHOD=${FUSION_METHOD:-rrf}

echo "Configuration:"
echo "  • Documents: $DOCUMENTS_PATH"
echo "  • Hybrid Search: $ENABLE_HYBRID"
echo "  • Fusion Method: $FUSION_METHOD"
echo ""
echo "======================================================================"
echo ""
echo "🚀 Server will start at: http://localhost:8000"
echo "   OpenAI endpoint: http://localhost:8000/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "======================================================================"
echo ""

# Start the server
python -m demo.openai_api_demo.server
