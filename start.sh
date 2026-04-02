#!/bin/bash
# Metadron Capital — Full Platform Startup
# Launches both the Python Engine API and the Express frontend server.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export ENGINE_API_PORT=${ENGINE_API_PORT:-8001}
export PORT=${PORT:-5000}

echo "╔══════════════════════════════════════════════╗"
echo "║       METADRON CAPITAL — PLATFORM START      ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Engine API : http://0.0.0.0:$ENGINE_API_PORT          ║"
echo "║  Frontend   : http://0.0.0.0:$PORT              ║"
echo "║  API Docs   : http://0.0.0.0:$ENGINE_API_PORT/api/engine/docs ║"
echo "╚══════════════════════════════════════════════╝"

# Start Python Engine API in background
echo "[1/2] Starting Python Engine API on port $ENGINE_API_PORT..."
python3 -m uvicorn engine.api.server:app \
    --host 0.0.0.0 \
    --port "$ENGINE_API_PORT" \
    --log-level info &
ENGINE_PID=$!

# Wait briefly for Python server to start
sleep 2

# Start Express frontend server
echo "[2/2] Starting Express frontend on port $PORT..."
npm run dev &
EXPRESS_PID=$!

# Trap to clean up both processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$ENGINE_PID" 2>/dev/null || true
    kill "$EXPRESS_PID" 2>/dev/null || true
    wait
    echo "All services stopped."
}
trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "Both servers running. Press Ctrl+C to stop."
wait
