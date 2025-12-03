# Fetchcraft Admin Setup Guide

Complete setup instructions for the Fetchcraft Admin web application.

## Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher
- npm or yarn
- Access to a running ingestion queue database

## Installation

### 1. Build the Frontend

First, build the React frontend:

```bash
cd packages/fetchcraft-admin
chmod +x build.sh
./build.sh
```

Or manually:

```bash
cd packages/fetchcraft-admin/frontend
npm install
npm run build
cd ..
```

### 2. Install the Python Package

```bash
# From the repository root
uv pip install -e packages/fetchcraft-admin

# Or if you're in the package directory
uv pip install -e .
```

### 3. Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Database configuration
DB_PATH=ingestion_queue.db
INGESTION_DB=ingestion_queue.db

# Server configuration
HOST=0.0.0.0
PORT=8080

# Document paths
DOCUMENTS_PATH=Documents

# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=fetchcraft_chatbot

# MongoDB configuration
MONGO_URI=mongodb://localhost:27017
DOCUMENT_DB=fetchcraft

# Additional ingestion configuration...
```

### 4. Start the Server

```bash
fetchcraft-admin
```

The server will start on `http://localhost:8080` (or the configured port).

## Development

### Backend Development

To develop the backend with auto-reload:

```bash
cd packages/fetchcraft-admin
uvicorn fetchcraft.admin.server:api --reload --port 8080
```

### Frontend Development

To develop the frontend with hot module replacement:

```bash
cd packages/fetchcraft-admin/frontend
npm run dev
```

The development server will run on `http://localhost:5173` and proxy API requests to the backend at `http://localhost:8080`.

## Usage

### Queue Messages Tab

- **View Messages**: Browse all messages in the ingestion queue
- **Filter by State**: Select "Done", "Pending", or "All" to filter messages
- **Adjust Rows**: Choose how many rows to display (50, 100, 200, or All)
- **Auto-Refresh**: The view automatically refreshes every 5 seconds
- **Statistics**: View queue statistics at the top of the page

### Ingestion Control Tab

- **View Status**: See the current status of the ingestion job
- **Start Job**: Click "Start Ingestion" to begin processing documents
- **Stop Job**: Click "Stop Ingestion" to halt the current job
- **Auto-Refresh**: Status updates automatically every 2 seconds

## Troubleshooting

### Database Not Found

If you see "Database not found" errors:

1. Check that `DB_PATH` in `.env` points to the correct SQLite database
2. Ensure the ingestion pipeline has created the database
3. Verify file permissions

### Frontend Not Loading

If the frontend shows "Frontend not built":

1. Run the build script: `./build.sh`
2. Check that `src/fetchcraft/admin/frontend/dist` exists
3. Rebuild if necessary: `cd frontend && npm run build`

### API Errors

If API calls fail:

1. Check that the backend server is running
2. Verify environment variables are set correctly
3. Check browser console for detailed error messages
4. Ensure the database and required services are accessible

### Port Already in Use

If port 8080 is already in use:

1. Change the `PORT` in `.env`
2. Or stop the conflicting service

## Production Deployment

For production deployment:

1. Build the frontend: `./build.sh`
2. Set production environment variables
3. Use a process manager like systemd or supervisord
4. Consider using a reverse proxy (nginx, traefik) for SSL

Example systemd service:

```ini
[Unit]
Description=Fetchcraft Admin
After=network.target

[Service]
Type=simple
User=fetchcraft
WorkingDirectory=/path/to/fetchcraft
Environment="PATH=/path/to/venv/bin"
EnvironmentFile=/path/to/fetchcraft/.env
ExecStart=/path/to/venv/bin/fetchcraft-admin
Restart=always

[Install]
WantedBy=multi-user.target
```

## API Documentation

Once the server is running, visit:

- Interactive API docs: `http://localhost:8080/docs`
- Alternative API docs: `http://localhost:8080/redoc`

## Architecture

- **Backend**: FastAPI (Python)
  - REST API for queue management and ingestion control
  - Serves static frontend files
  - Manages ingestion job processes

- **Frontend**: React + TypeScript
  - Single-page application
  - Two-tab interface (Queue Messages, Ingestion Control)
  - Real-time updates via polling
  - TailwindCSS for styling

## Support

For issues or questions:

1. Check the logs from the server
2. Review environment configuration
3. Ensure all dependencies are installed
4. Verify external services (MongoDB, Qdrant) are running
