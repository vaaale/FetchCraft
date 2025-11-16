# Fetchcraft Admin - Quick Start

Get up and running with Fetchcraft Admin in 5 minutes.

## Quick Installation

```bash
# 1. Navigate to the package directory
cd packages/fetchcraft-admin

# 2. Build the frontend
./build.sh

# 3. Install the package
uv pip install -e .

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings

# 5. Start the server
fetchcraft-admin
```

## Access the Application

Open your browser and navigate to:

```
http://localhost:8080
```

## What You'll See

### Tab 1: Queue Messages
- **Statistics**: Total messages, done, pending, and failed counts
- **Filters**: 
  - State dropdown: All, Done, Pending, Processing, Failed
  - Rows dropdown: 50, 100, 200, All
- **Table**: View all messages with ID, queue, state, attempts, timestamp, and body preview
- **Pagination**: Navigate through multiple pages of messages
- **Auto-refresh**: Updates every 5 seconds

### Tab 2: Ingestion Control
- **Status Display**: Current job status (running, stopped, error) and process ID
- **Control Buttons**:
  - **Start Ingestion**: Begin processing documents
  - **Stop Ingestion**: Halt the current job
- **Information**: Details about what ingestion jobs do
- **Auto-refresh**: Updates every 2 seconds

## Common Tasks

### Start an Ingestion Job

1. Go to the "Ingestion Control" tab
2. Click "Start Ingestion"
3. Monitor the status in real-time

### Filter Queue Messages

1. Go to the "Queue Messages" tab
2. Select a state from the dropdown (e.g., "Done")
3. Adjust rows per page if needed
4. View filtered results

### Monitor Queue Statistics

1. Go to the "Queue Messages" tab
2. View the statistics cards at the top:
   - Total Messages
   - Done
   - Pending
   - Failed

## Troubleshooting

**Server won't start?**
- Check that port 8080 is not in use
- Verify environment variables in `.env`
- Ensure database path is correct

**Frontend not loading?**
- Run `./build.sh` to rebuild
- Check browser console for errors

**Database errors?**
- Verify `DB_PATH` points to existing database
- Ensure proper file permissions

## Next Steps

- Read [SETUP.md](SETUP.md) for detailed configuration
- Check [README.md](README.md) for API documentation
- Review `.env.example` for all configuration options

## API Access

The REST API is available at `/api/`:

- `GET /api/messages` - List messages
- `GET /api/stats` - Get queue statistics
- `POST /api/ingestion/start` - Start ingestion
- `POST /api/ingestion/stop` - Stop ingestion
- `GET /api/ingestion/status` - Get job status

Interactive API docs: `http://localhost:8080/docs`
