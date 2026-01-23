#!/bin/bash
#
# MCP-PaperStream Server Startup Script
# =====================================
# Starts the MCP server with proper environment
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (works even when called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}🧬 MCP-PaperStream Server${NC}"
echo "================================"

# Check .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if SD API is reachable
SD_URL="${SD_API_URL:-http://127.0.0.1:7860}"
echo -e "${YELLOW}🔍 Checking Stable Diffusion API at ${SD_URL}...${NC}"

if curl -s --connect-timeout 5 "${SD_URL}/sdapi/v1/sd-models" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Stable Diffusion API is running${NC}"
else
    echo -e "${YELLOW}⚠️  Stable Diffusion API not reachable at ${SD_URL}${NC}"
    echo "   Make sure AUTOMATIC1111 is running with: ./webui.sh --api --listen"
    echo "   Continuing anyway (SD features will fail)..."
fi

# Configuration
HOST="${FASTMCP_HOST:-0.0.0.0}"
PORT="${FASTMCP_PORT:-8089}"

echo ""
echo -e "${GREEN}🚀 Starting MCP Server...${NC}"
echo "   Host: ${HOST}"
echo "   Port: ${PORT}"
echo "   SSE:  http://${HOST}:${PORT}/sse-bertscore"
echo ""

# Start server using python -m to avoid shebang issues
# Use 'app' (the http_app) instead of 'mcp' directly
exec .venv/bin/python -m uvicorn src.paperstream.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level info
