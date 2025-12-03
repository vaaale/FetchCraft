# Fetchcraft File Finder - Frontend

A modern, single-page web application for semantic file search powered by RAG (Retrieval-Augmented Generation). Built with React, TypeScript, and TailwindCSS.

## Features

- **Semantic Search**: Find files based on meaning, not just keywords
- **Pagination**: Navigate through large result sets efficiently
- **Modern UI**: Clean, responsive design with TailwindCSS
- **Real-time Results**: Fast search with instant feedback
- **Score-based Ranking**: See relevance scores for each result

## Tech Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **TailwindCSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start the development server
npm run dev
```

The application will be available at `http://localhost:3001`.

The dev server is configured to proxy API requests to `http://localhost:8765`.

### Building for Production

```bash
# Build the application
npm run build
```

The built files will be in the `dist/` directory and will be served by the backend MCP server.

### Linting

```bash
# Run ESLint
npm run lint
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── FileSearchTab.tsx    # Main search component with pagination
│   ├── App.tsx                   # Root component
│   ├── api.ts                    # API client for backend communication
│   ├── index.css                 # Global styles and Tailwind imports
│   ├── main.tsx                  # Application entry point
│   └── vite-env.d.ts            # Vite type definitions
├── index.html                    # HTML template
├── package.json                  # Dependencies and scripts
├── tsconfig.json                 # TypeScript configuration
├── vite.config.ts               # Vite configuration
└── tailwind.config.js           # TailwindCSS configuration
```

## API Integration

The frontend communicates with the backend via REST API:

### Endpoints

#### `GET /api/find-files`

Search for files using semantic search.

**Query Parameters:**
- `query` (required): Search query string
- `num_results` (optional): Number of results per page (default: 10, max: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "files": [
    {
      "filename": "config.py",
      "source": "/path/to/config.py",
      "score": 0.85,
      "text_preview": "Configuration settings for..."
    }
  ],
  "total": 42,
  "offset": 0
}
```

## Component Overview

### FileSearchTab

The main component that handles:
- Search input and form submission
- Results display with file information
- Pagination controls
- Loading and error states
- Configurable results per page

Features:
- Real-time search on form submit
- Score-based color coding
- Text preview truncation
- Responsive pagination
- Clean, accessible UI

## Customization

### Styling

The application uses TailwindCSS for styling. You can customize:

- Colors and theme in `tailwind.config.js`
- Global styles in `src/index.css`
- Component-specific styles in individual `.tsx` files

### API Configuration

API endpoint configuration is in `vite.config.ts` for development and `src/api.ts` for production.

## Deployment

The frontend is designed to be served by the Fetchcraft MCP Server. After building:

1. Run `npm run build`
2. Start the MCP server
3. Navigate to `http://localhost:8765`

The server will automatically serve the built frontend.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Contributing

When making changes:

1. Follow the existing code style
2. Use TypeScript strict mode
3. Add appropriate error handling
4. Test on different screen sizes
5. Run linting before committing

## License

Part of the Fetchcraft project.
