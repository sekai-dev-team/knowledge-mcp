#!/bin/bash
set -e

# Build initial index if none exists
if [ ! -f /data/index.db ]; then
    echo "[entrypoint] No index found. Building initial index..."
    python indexer.py --full --vault /vault --db /data/index.db
    echo "[entrypoint] Initial index complete."
fi

# Start file change watcher in background
echo "[entrypoint] Starting change watcher..."
python change_watcher.py --vault /vault --db /data/index.db &

# Start MCP server in foreground
echo "[entrypoint] Starting MCP server..."
exec python mcp_server.py --vault /vault --db /data/index.db --host 0.0.0.0 --port 8000
