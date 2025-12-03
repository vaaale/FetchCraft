# Fetchcraft Admin

A modern, web-based administration interface for Fetchcraft document ingestion pipelines. Built with FastAPI and React.

## ✨ Features

### Queue Management
- 📊 **Real-time Statistics**: View total messages, done, pending, and failed counts
- 🔍 **Advanced Filtering**: Filter by state (Done, Pending, Processing, Failed, All)
- 📄 **Pagination**: Choose rows per page (50, 100, 200, or All)
- 🔄 **Auto-refresh**: Queue updates every 5 seconds
- 📋 **Detailed View**: See message ID, queue, state, attempts, timestamps, and body preview

### Ingestion Control
- ▶️ **Start/Stop Jobs**: Control ingestion jobs with simple buttons
- 📈 **Status Monitoring**: Real-time job status and process ID tracking
- ⚡ **Auto-refresh**: Status updates every 2 seconds
- 💡 **Job Information**: Learn about what ingestion jobs do

### User Experience
- 🎨 **Modern UI**: Clean, professional interface with TailwindCSS
- 📱 **Responsive Design**: Works on desktop and mobile
- 🚀 **Fast**: Built with Vite and React for optimal performance
- 🔐 **Production Ready**: Includes CORS support and error handling

## 🚀 Quick Start

```bash
# 1. Build the application
cd packages/fetchcraft-admin
./build.sh

# 2. Install the package
uv pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Start the server
fetchcraft-admin
```

Open `http://localhost:8080` in your browser.

📖 For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)

## 📦 Installation

### Prerequisites

- Python 3.12+
- Node.js 18+
- npm or yarn

### Step-by-Step

1. **Build the Frontend**
   ```bash
   cd packages/fetchcraft-admin
   ./build.sh
   ```

2. **Install the Package**
   ```bash
   uv pip install -e .
   ```

3. **Configure**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run**
   ```bash
   fetchcraft-admin
   ```

📖 For complete setup details, see [SETUP.md](SETUP.md)

## ⚙️ Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `ingestion_queue.db` | Path to SQLite queue database |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8080` | Server port |
| `DOCUMENTS_PATH` | `Documents` | Path to documents for ingestion |

See `.env.example` for all available options.

## 🎯 Usage

### Queue Messages Tab

1. View real-time queue statistics
2. Filter messages by state using the dropdown
3. Adjust rows per page for comfortable viewing
4. Click refresh to manually update (or wait for auto-refresh)
5. Navigate between pages using pagination controls

### Ingestion Control Tab

1. Check current job status
2. Click "Start Ingestion" to begin processing documents
3. Click "Stop Ingestion" to halt the current job
4. View process ID and detailed status information

## 🛠️ Development

### Backend Development

The backend is built with **FastAPI**:

```bash
cd packages/fetchcraft-admin
uvicorn fetchcraft.admin.server:api --reload --port 8080
```

Visit `http://localhost:8080/docs` for interactive API documentation.

### Frontend Development

The frontend is built with **React + TypeScript + Vite**:

```bash
cd packages/fetchcraft-admin/frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` with API proxying to the backend.

### Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- Pydantic - Data validation
- SQLite - Queue database
- Uvicorn - ASGI server

**Frontend:**
- React 18 - UI framework
- TypeScript - Type safety
- Vite - Build tool
- TailwindCSS - Styling
- Lucide React - Icons

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/messages` | List queue messages (filterable) |
| `GET` | `/api/stats` | Get queue statistics |
| `POST` | `/api/ingestion/start` | Start ingestion job |
| `POST` | `/api/ingestion/stop` | Stop ingestion job |
| `GET` | `/api/ingestion/status` | Get job status |

Interactive API documentation: `http://localhost:8080/docs`

### Example: List Messages

```bash
curl "http://localhost:8080/api/messages?state=done&limit=50&offset=0"
```

### Example: Start Ingestion

```bash
curl -X POST "http://localhost:8080/api/ingestion/start"
```

## 🏗️ Architecture

```
fetchcraft-admin/
├── src/fetchcraft/admin/
│   ├── __init__.py
│   ├── server.py           # FastAPI backend
│   └── frontend/dist/      # Built React app (served by FastAPI)
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main application
│   │   ├── api.ts          # API client
│   │   └── components/
│   │       ├── QueueTab.tsx       # Queue messages UI
│   │       └── IngestionTab.tsx   # Ingestion control UI
│   ├── package.json
│   └── vite.config.ts
├── pyproject.toml
├── build.sh
├── README.md
├── SETUP.md
└── QUICKSTART.md
```

## 🐛 Troubleshooting

### Common Issues

**Database not found**
- Check `DB_PATH` in `.env`
- Ensure the ingestion pipeline has created the database
- Verify file permissions

**Frontend not loading**
- Run `./build.sh` to rebuild the frontend
- Check that `src/fetchcraft/admin/frontend/dist` exists

**Port already in use**
- Change `PORT` in `.env`
- Or stop the conflicting service

**API errors**
- Check that all required services are running (MongoDB, Qdrant)
- Verify environment variables
- Check server logs for details

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [SETUP.md](SETUP.md) - Detailed setup and configuration
- [frontend/README.md](frontend/README.md) - Frontend development guide

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. Backend code follows Python best practices
2. Frontend code passes TypeScript checks
3. UI is responsive and accessible
4. Tests pass (when available)

## 📄 License

Part of the Fetchcraft project.
