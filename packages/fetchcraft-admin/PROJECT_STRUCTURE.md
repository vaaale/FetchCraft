# Fetchcraft Admin - Project Structure

Complete overview of the project structure and files.

## Directory Tree

```
fetchcraft-admin/
├── src/fetchcraft/admin/
│   ├── __init__.py              # Package initialization
│   ├── server.py                # FastAPI backend server
│   └── frontend/dist/           # Built React app (created after build)
│
├── frontend/                    # React frontend source
│   ├── src/
│   │   ├── main.tsx            # React entry point
│   │   ├── App.tsx             # Main application component
│   │   ├── api.ts              # API client functions
│   │   ├── index.css           # Global styles (TailwindCSS)
│   │   ├── vite-env.d.ts       # Vite type definitions
│   │   └── components/
│   │       ├── QueueTab.tsx    # Queue messages UI component
│   │       └── IngestionTab.tsx # Ingestion control UI component
│   │
│   ├── public/                  # Static assets
│   ├── index.html              # HTML template
│   ├── package.json            # Frontend dependencies
│   ├── tsconfig.json           # TypeScript configuration
│   ├── tsconfig.node.json      # TypeScript config for Vite
│   ├── vite.config.ts          # Vite build configuration
│   ├── tailwind.config.js      # TailwindCSS configuration
│   ├── postcss.config.js       # PostCSS configuration
│   ├── .eslintrc.cjs           # ESLint configuration
│   ├── .gitignore              # Git ignore rules
│   └── README.md               # Frontend development guide
│
├── pyproject.toml              # Python package configuration
├── build.sh                    # Build script (executable)
├── .env.example                # Example environment variables
├── README.md                   # Main documentation
├── QUICKSTART.md               # Quick start guide
├── SETUP.md                    # Detailed setup instructions
└── PROJECT_STRUCTURE.md        # This file
```

## File Descriptions

### Backend (Python)

#### `src/fetchcraft/admin/__init__.py`
- Package initialization
- Version information

#### `src/fetchcraft/admin/server.py`
- FastAPI application
- API endpoints for queue management and ingestion control
- Static file serving for frontend
- Process management for ingestion jobs
- Database connection handling

### Frontend (React/TypeScript)

#### `frontend/src/main.tsx`
- React application entry point
- Renders root component

#### `frontend/src/App.tsx`
- Main application component
- Tab navigation
- Layout and header

#### `frontend/src/api.ts`
- API client functions
- TypeScript interfaces for API responses
- HTTP request handling

#### `frontend/src/components/QueueTab.tsx`
- Queue messages table view
- Filtering (state dropdown)
- Pagination (rows per page, page navigation)
- Statistics display
- Auto-refresh functionality

#### `frontend/src/components/IngestionTab.tsx`
- Ingestion job control interface
- Start/Stop buttons
- Status display
- Process information

#### `frontend/src/index.css`
- Global styles
- TailwindCSS imports
- CSS custom properties

### Configuration Files

#### `pyproject.toml`
- Python package metadata
- Dependencies (FastAPI, Uvicorn, etc.)
- Build system configuration
- Entry point definition

#### `frontend/package.json`
- Frontend dependencies
- NPM scripts (dev, build, preview)
- Project metadata

#### `frontend/vite.config.ts`
- Vite build configuration
- React plugin
- Path aliases
- Build output directory
- Proxy configuration for API

#### `frontend/tailwind.config.js`
- TailwindCSS theme customization
- Color palette
- Border radius values

#### `frontend/tsconfig.json`
- TypeScript compiler options
- Path aliases
- Module resolution

#### `.env.example`
- Example environment variables
- Configuration documentation

### Build & Documentation

#### `build.sh`
- Automated build script
- Installs frontend dependencies
- Builds React application
- Places output in correct location

#### `README.md`
- Main project documentation
- Feature overview
- Installation instructions
- Usage guide
- API reference

#### `QUICKSTART.md`
- Quick start guide
- 5-minute setup instructions
- Common tasks

#### `SETUP.md`
- Detailed setup instructions
- Configuration options
- Troubleshooting
- Production deployment guide

## API Endpoints

### Queue Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/messages` | GET | List messages (filterable, paginated) |
| `/api/stats` | GET | Queue statistics |

### Ingestion Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingestion/start` | POST | Start ingestion job |
| `/api/ingestion/stop` | POST | Stop ingestion job |
| `/api/ingestion/status` | GET | Get job status |

### Static Files

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve React app |
| `/assets/*` | GET | Serve static assets |
| `/{path}` | GET | SPA routing (returns index.html) |

## Data Models

### Message (Frontend/Backend)
```typescript
{
  id: string
  queue: string
  state: string  // done, ready, processing, failed
  attempts: number
  available_at: string | null
  lease_until: string | null
  body_preview: string
}
```

### Queue Stats
```typescript
{
  total_messages: number
  by_state: Record<string, number>
  by_queue: Record<string, number>
  failed_messages: number
  oldest_pending: string | null
}
```

### Ingestion Status
```typescript
{
  status: 'running' | 'stopped' | 'error'
  pid: number | null
}
```

## Technologies

### Backend
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **SQLite** - Queue database
- **Uvicorn** - ASGI server
- **Python 3.12+** - Language version

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS
- **Lucide React** - Icon library

### Development Tools
- **ESLint** - JavaScript/TypeScript linting
- **PostCSS** - CSS processing
- **Autoprefixer** - CSS vendor prefixes

## Build Process

1. **Frontend Build** (`npm run build`)
   - TypeScript compilation
   - React component bundling
   - Asset optimization
   - Output to `../src/fetchcraft/admin/frontend/dist`

2. **Python Package Install** (`uv pip install -e .`)
   - Installs Python dependencies
   - Creates entry point script
   - Makes package importable

3. **Runtime**
   - FastAPI serves API endpoints
   - FastAPI serves static React files
   - React app makes API calls to backend

## Environment Variables

See `.env.example` for all variables. Key ones:

```bash
# Server
HOST=0.0.0.0
PORT=8080

# Database
DB_PATH=ingestion_queue.db
INGESTION_DB=ingestion_queue.db

# Documents
DOCUMENTS_PATH=Documents

# External Services
QDRANT_HOST=localhost
QDRANT_PORT=6333
MONGO_URI=mongodb://localhost:27017
```

## Development Workflow

### Backend Changes
1. Edit `src/fetchcraft/admin/server.py`
2. Run with `--reload`: `uvicorn fetchcraft.admin.server:app --reload`
3. Test at `http://localhost:8080/docs`

### Frontend Changes
1. Edit files in `frontend/src/`
2. Run dev server: `npm run dev` (in `frontend/`)
3. View at `http://localhost:5173`
4. Build for production: `npm run build`

### Full Build
```bash
./build.sh
```

## Deployment

### Production Build
```bash
cd packages/fetchcraft-admin
./build.sh
uv pip install -e .
```

### Run Production
```bash
fetchcraft-admin
```

or

```bash
uvicorn fetchcraft.admin.server:app --host 0.0.0.0 --port 8080
```

### With Process Manager (systemd)
Create `/etc/systemd/system/fetchcraft-admin.service`:

```ini
[Unit]
Description=Fetchcraft Admin
After=network.target

[Service]
Type=simple
User=fetchcraft
WorkingDirectory=/path/to/fetchcraft
Environment="PATH=/path/to/venv/bin"
EnvironmentFile=/path/to/.env
ExecStart=/path/to/venv/bin/fetchcraft-admin
Restart=always

[Install]
WantedBy=multi-user.target
```

## Security Considerations

- CORS is enabled for all origins in development
- For production, configure CORS origins in `server.py`
- Environment variables should never be committed
- Use reverse proxy (nginx/traefik) for SSL in production
- Consider authentication for production deployments

## Future Enhancements

Potential features to add:

- User authentication and authorization
- WebSocket support for real-time updates
- Message retry/delete functionality
- Bulk operations on messages
- Advanced filtering and search
- Export functionality (CSV, JSON)
- Detailed logging and audit trail
- Job scheduling
- Email notifications
- Multi-user support with roles
