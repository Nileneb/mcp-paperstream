#!/bin/bash
#
# PaperStream MCP Server Startup Script
# =====================================
# Starts the integrated server with REST API, MCP, and SSE
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           PaperStream MCP Server v1.0.0                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in Docker
if [ -d "/app/src" ]; then
    IS_DOCKER=true
    echo -e "${BLUE}Environment: Docker${NC}"
else
    IS_DOCKER=false
    # Check .venv exists
    if [ ! -d ".venv" ]; then
        echo -e "${RED}❌ Virtual environment not found!${NC}"
        echo "   Run: python -m venv .venv && .venv/bin/pip install -e ."
        exit 1
    fi
    PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | cut -d' ' -f2)
    echo -e "${BLUE}Python:${NC} $PYTHON_VERSION"
fi

# Check if SD API is reachable (optional)
SD_URL="${SD_API_URL:-http://127.0.0.1:7860}"
if curl -s --connect-timeout 2 "${SD_URL}/sdapi/v1/sd-models" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Stable Diffusion API available${NC}"
else
    echo -e "${YELLOW}⚠️  Stable Diffusion API not running (optional)${NC}"
fi

# Parse arguments
MODE="${1:-integrated}"
PORT="${PAPERSTREAM_PORT:-8089}"
HOST="${PAPERSTREAM_HOST:-0.0.0.0}"

case "$MODE" in
    integrated|full)
        echo -e "${GREEN}Mode: Integrated (REST + MCP + SSE)${NC}"
        SERVER_MODULE="src.paperstream.server_integrated:app"
        ;;
    mcp|bertscore)
        echo -e "${GREEN}Mode: MCP + BERTScore only${NC}"
        SERVER_MODULE="src.paperstream.server:app"
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  integrated  Full server with REST API, MCP, SSE (default)"
        echo "  mcp         MCP + BERTScore IoT server only"
        echo ""
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Endpoints:${NC}"
echo -e "  REST API:    http://$HOST:$PORT/api/"
echo -e "  MCP:         http://$HOST:$PORT/mcp"
echo -e "  Unity SSE:   http://$HOST:$PORT/api/stream/unity"
echo -e "  Health:      http://$HOST:$PORT/health"
echo ""
echo -e "${GREEN}Starting server...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start server with uvicorn
if [ "$IS_DOCKER" = true ]; then
    # Docker: use system python, no reload
    exec python -m uvicorn "$SERVER_MODULE" \
        --host "$HOST" \
        --port "$PORT"
else
    # Local: use venv, with reload for development
    exec .venv/bin/python -m uvicorn "$SERVER_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --reload-dir src
fi
