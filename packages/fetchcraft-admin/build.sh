#!/bin/bash
# Build script for Fetchcraft Admin

set -e

echo "================================"
echo "Building Fetchcraft Admin"
echo "================================"
echo ""

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    echo ""
fi

# Build the frontend
echo "🏗️  Building frontend..."
npm run build
echo ""

# Navigate back to package root
cd ..

echo "✅ Build complete!"
echo ""
echo "The frontend has been built and placed in:"
echo "  src/fetchcraft/admin/frontend/dist"
echo ""
echo "You can now install and run the package:"
echo "  uv pip install -e ."
echo "  fetchcraft-admin"
echo ""
