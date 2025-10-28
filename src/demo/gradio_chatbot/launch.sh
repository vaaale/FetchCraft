#!/bin/bash

# Launch script for Gradio Chatbot
# This script checks prerequisites and starts the app

set -e

echo "=================================="
echo "🤖 RAG Chatbot Launch Script"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found"
    echo "   Please run this script from the demo/gradio_chatbot directory"
    exit 1
fi

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✓ Loading configuration from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  No .env file found (using environment variables)"
    echo "   Copy .env.example to .env to customize settings"
fi

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check if Qdrant is running
echo ""
echo "Checking Qdrant connection..."
if curl -s "http://${QDRANT_HOST:-localhost}:${QDRANT_PORT:-6333}/collections" > /dev/null 2>&1; then
    echo "✓ Qdrant is running"
else
    echo "⚠️  Cannot connect to Qdrant at ${QDRANT_HOST:-localhost}:${QDRANT_PORT:-6333}"
    echo ""
    echo "Start Qdrant with:"
    echo "  docker run -p 6333:6333 qdrant/qdrant"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required packages
echo ""
echo "Checking required packages..."
missing_packages=()

if ! python3 -c "import gradio" 2>/dev/null; then
    missing_packages+=("gradio")
fi

if ! python3 -c "import fastembed" 2>/dev/null; then
    missing_packages+=("fastembed")
fi

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "⚠️  Missing packages: ${missing_packages[*]}"
    echo ""
    read -p "Install missing packages? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install "${missing_packages[@]}"
    else
        echo "Please install: pip install ${missing_packages[*]}"
        exit 1
    fi
else
    echo "✓ All required packages installed"
fi

# Check documents path
echo ""
echo "Checking documents path..."
docs_path="${DOCUMENTS_PATH:-Documents}"
if [ -d "$docs_path" ]; then
    num_files=$(find "$docs_path" -type f | wc -l)
    echo "✓ Documents directory exists ($num_files files)"
else
    echo "⚠️  Documents directory not found: $docs_path"
    echo ""
    read -p "Create directory? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$docs_path"
        echo "✓ Created $docs_path"
        echo "  Add your documents to this directory"
    fi
fi

# Start the app
echo ""
echo "=================================="
echo "🚀 Starting Gradio Chatbot..."
echo "=================================="
echo ""

cd ../..  # Go to src directory
python3 -m demo.gradio_chatbot.app

# If we get here, the app has stopped
echo ""
echo "👋 Chatbot stopped"
