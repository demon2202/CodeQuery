#!/usr/bin/env bash
# Start the CodeQuery backend server.
#
# Prerequisites:
#   1. Python 3.10+ with venv created and requirements installed
#   2. Ollama running with qwen2.5-coder:7b pulled
#   3. Git installed
#
# Usage:
#   chmod +x start.sh
#   ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Ollama is not running!"
    echo "   Start it in another terminal: ollama serve"
    echo "   Then pull the model: ollama pull qwen2.5-coder:7b"
    echo ""
    echo "Starting anyway (chat will fail until Ollama is available)..."
fi

# Start the server
echo ""
echo "Starting CodeQuery backend on http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

uvicorn app.main:app --host "${CQ_HOST:-0.0.0.0}" --port "${CQ_PORT:-8000}" --reload
