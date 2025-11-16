# Fetchcraft Admin Frontend

React-based web interface for Fetchcraft Admin.

## Development

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The development server will run on `http://localhost:5173` and proxy API requests to the backend.

## Building

Build the production bundle:

```bash
npm run build
```

The build output will be placed in `../src/fetchcraft/admin/frontend/dist` and will be served by the FastAPI backend.

## Technology Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **Lucide React** - Icons

## Components

- `QueueTab` - Displays and filters ingestion queue messages
- `IngestionTab` - Controls for starting and stopping ingestion jobs
