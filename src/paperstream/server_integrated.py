"""
PaperStream MCP Server - Integrated API
=======================================

Combines all endpoints:
- MCP Tools (via FastMCP)
- REST API for n8n webhooks
- REST API for Android devices
- SSE for Unity clients
- BERTScore distributed computation
"""

from __future__ import annotations
import os
import json
import time
import asyncio
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import uuid

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Database Initialization
# =========================
def init_database():
    """Initialize database on startup"""
    try:
        from .db import get_db
        db = get_db()
        db.initialize()
        logger.info("Database initialized")
        return db
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return None

# =========================
# MCP Server Setup
# =========================
mcp = FastMCP(
    name="paperstream",
    instructions="""
    PaperStream MCP Server - Scientific Paper Validation Platform
    
    Tools:
    - submit_paper: Submit paper for processing (n8n webhook)
    - create_rule: Create validation rule with BioBERT embeddings
    - get_jobs: Get validation jobs for Android devices
    - submit_validation: Submit validation results
    - get_paper_status: Get paper validation status
    - get_leaderboard: Get gamification leaderboard
    """,
    version="1.0.0",
)

# =========================
# MCP Tools
# =========================

@mcp.tool()
async def submit_paper(
    paper_id: str,
    title: Optional[str] = None,
    pdf_url: Optional[str] = None,
    priority: int = 5,
    source: str = "mcp"
) -> Dict[str, Any]:
    """
    Submit a new paper for processing.
    
    Args:
        paper_id: Unique identifier (e.g., PMC12345, DOI)
        title: Paper title
        pdf_url: URL to download PDF
        priority: Processing priority (1-10)
        source: Source of submission
    
    Returns:
        Submission result
    """
    from .api import get_paper_handler
    handler = get_paper_handler()
    return await handler.submit_paper(
        paper_id=paper_id,
        title=title,
        pdf_url=pdf_url,
        priority=priority,
        source=source
    )

@mcp.tool()
async def create_rule(
    rule_id: str,
    question: str,
    positive_phrases: list,
    negative_phrases: Optional[list] = None,
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Create a validation rule with BioBERT embeddings.
    
    Args:
        rule_id: Unique rule identifier
        question: The validation question
        positive_phrases: Phrases indicating positive match
        negative_phrases: Phrases indicating negative match
        threshold: Similarity threshold
    
    Returns:
        Rule creation result
    """
    from .api import get_rule_handler
    handler = get_rule_handler()
    return handler.create_rule(
        rule_id=rule_id,
        question=question,
        positive_phrases=positive_phrases,
        negative_phrases=negative_phrases,
        threshold=threshold
    )

@mcp.tool()
async def get_paper_status(paper_id: str) -> Dict[str, Any]:
    """
    Get validation status for a paper.
    
    Args:
        paper_id: Paper identifier
    
    Returns:
        Paper status with validation results
    """
    from .pipeline import get_consensus_engine
    engine = get_consensus_engine()
    return engine.get_paper_validation_status(paper_id)

@mcp.tool()
async def get_leaderboard(limit: int = 10) -> Dict[str, Any]:
    """
    Get gamification leaderboard.
    
    Args:
        limit: Number of top players to return
    
    Returns:
        Leaderboard entries
    """
    from .db import get_db
    db = get_db()
    entries = db.get_leaderboard(limit)
    return {
        "leaderboard": [e.to_dict() for e in entries],
        "total_players": len(entries)
    }

@mcp.tool()
async def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics.
    
    Returns:
        Database and system stats
    """
    from .db import get_db
    db = get_db()
    return db.get_stats()

@mcp.tool()
async def process_paper(paper_id: str) -> Dict[str, Any]:
    """
    Process a paper (extract sections, generate embeddings).
    
    Args:
        paper_id: Paper to process
    
    Returns:
        Processing result
    """
    from .pipeline import get_paper_processor
    processor = get_paper_processor()
    return await processor.process_paper(paper_id)

@mcp.tool()
async def load_default_rules() -> Dict[str, Any]:
    """
    Load default validation rules into database.
    
    Returns:
        Loading result
    """
    from .api import get_rule_handler
    handler = get_rule_handler()
    return handler.load_default_rules()

# =========================
# REST API Handlers
# =========================

async def api_submit_paper(request: Request) -> JSONResponse:
    """POST /api/papers/submit - n8n webhook"""
    try:
        body = await request.json()
        from .api import get_paper_handler
        handler = get_paper_handler()
        result = await handler.submit_paper(**body)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in submit_paper: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_get_paper(request: Request) -> JSONResponse:
    """GET /api/papers/{paper_id}"""
    paper_id = request.path_params["paper_id"]
    from .api import get_paper_handler
    handler = get_paper_handler()
    result = handler.get_paper(paper_id)
    if result:
        return JSONResponse(result)
    return JSONResponse({"error": "Paper not found"}, status_code=404)

async def api_list_papers(request: Request) -> JSONResponse:
    """GET /api/papers"""
    status = request.query_params.get("status")
    limit = int(request.query_params.get("limit", 100))
    from .api import get_paper_handler
    handler = get_paper_handler()
    result = handler.list_papers(status=status, limit=limit)
    return JSONResponse({"papers": result})

async def api_create_rule(request: Request) -> JSONResponse:
    """POST /api/rules/create"""
    try:
        body = await request.json()
        from .api import get_rule_handler
        handler = get_rule_handler()
        result = handler.create_rule(**body)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in create_rule: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_list_rules(request: Request) -> JSONResponse:
    """GET /api/rules"""
    from .api import get_rule_handler
    handler = get_rule_handler()
    rules = handler.list_rules()
    return JSONResponse({"rules": rules})

async def api_get_jobs(request: Request) -> JSONResponse:
    """GET /api/jobs/next - Android endpoint"""
    device_id = request.query_params.get("device_id")
    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    
    limit = int(request.query_params.get("limit", 5))
    
    from .api import get_job_handler
    handler = get_job_handler()
    result = handler.get_next_jobs(device_id, limit)
    return JSONResponse(result)

async def api_submit_validation(request: Request) -> JSONResponse:
    """POST /api/validation/submit - Android endpoint"""
    try:
        body = await request.json()
        device_id = body.get("device_id")
        results = body.get("results", [])
        
        if not device_id:
            return JSONResponse({"error": "device_id required"}, status_code=400)
        
        from .api import get_job_handler
        handler = get_job_handler()
        result = handler.submit_results(device_id, results)
        
        # Emit SSE event if Unity clients connected
        from .api.sse_stream import get_unity_sse_stream
        sse = get_unity_sse_stream()
        if sse.get_client_count() > 0:
            await sse.emit_leaderboard_update([], [device_id])
        
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in submit_validation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_register_device(request: Request) -> JSONResponse:
    """POST /api/devices/register"""
    try:
        body = await request.json()
        from .api import get_device_handler
        handler = get_device_handler()
        result = handler.register_device(**body)
        
        # Emit SSE event
        from .api.sse_stream import get_unity_sse_stream
        sse = get_unity_sse_stream()
        if sse.get_client_count() > 0:
            await sse.emit_device_joined(
                body.get("device_id"),
                body.get("device_name", "Unknown"),
                len(handler.list_devices())
            )
        
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in register_device: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_get_device(request: Request) -> JSONResponse:
    """GET /api/devices/{device_id}"""
    device_id = request.path_params["device_id"]
    from .api import get_device_handler
    handler = get_device_handler()
    result = handler.get_device(device_id)
    if result:
        return JSONResponse(result)
    return JSONResponse({"error": "Device not found"}, status_code=404)

async def api_get_consensus(request: Request) -> JSONResponse:
    """GET /api/consensus/{paper_id}"""
    paper_id = request.path_params["paper_id"]
    from .pipeline import get_consensus_engine
    engine = get_consensus_engine()
    result = engine.get_paper_validation_status(paper_id)
    return JSONResponse(result)

async def api_get_stats(request: Request) -> JSONResponse:
    """GET /api/stats"""
    from .db import get_db
    db = get_db()
    stats = db.get_stats()
    
    from .api.sse_stream import get_unity_sse_stream
    sse = get_unity_sse_stream()
    stats["unity_clients"] = sse.get_client_count()
    
    return JSONResponse(stats)

# =========================
# SSE Endpoint for Unity
# =========================

async def unity_sse_handler(request: Request) -> StreamingResponse:
    """GET /api/stream/unity - SSE for Unity clients"""
    client_id = request.query_params.get("client_id", f"unity_{uuid.uuid4().hex[:8]}")
    
    from .api.sse_stream import get_unity_sse_stream
    sse = get_unity_sse_stream()
    
    return StreamingResponse(
        sse.subscribe(client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# =========================
# Health Check
# =========================

async def health_check(request: Request) -> JSONResponse:
    """GET /health"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })

# =========================
# Route Definitions
# =========================

api_routes = [
    # Papers
    Route("/api/papers/submit", api_submit_paper, methods=["POST"]),
    Route("/api/papers", api_list_papers, methods=["GET"]),
    Route("/api/papers/{paper_id}", api_get_paper, methods=["GET"]),
    
    # Rules
    Route("/api/rules/create", api_create_rule, methods=["POST"]),
    Route("/api/rules", api_list_rules, methods=["GET"]),
    
    # Jobs (Android)
    Route("/api/jobs/next", api_get_jobs, methods=["GET"]),
    Route("/api/validation/submit", api_submit_validation, methods=["POST"]),
    
    # Devices
    Route("/api/devices/register", api_register_device, methods=["POST"]),
    Route("/api/devices/{device_id}", api_get_device, methods=["GET"]),
    
    # Consensus
    Route("/api/consensus/{paper_id}", api_get_consensus, methods=["GET"]),
    
    # Stats
    Route("/api/stats", api_get_stats, methods=["GET"]),
    
    # Unity SSE
    Route("/api/stream/unity", unity_sse_handler, methods=["GET"]),
    
    # Health
    Route("/health", health_check, methods=["GET"]),
]

# =========================
# App Factory
# =========================

def create_app() -> Starlette:
    """Create the combined Starlette app with MCP and REST routes"""
    
    # Initialize database
    init_database()
    
    # Create MCP ASGI app with SSE transport for n8n integration
    # SSE transport allows tool invocation via HTTP stream (n8n MCP node)
    # The SSE app exposes: /sse (connect), /messages (send)
    mcp_sse_app = mcp.http_app(transport="sse")
    
    # Also create streamable-http endpoint for MCP Inspector compatibility
    mcp_http_app = mcp.http_app(transport="streamable-http")
    
    # Combine routes
    # Mount at root "" so endpoints become /sse and /messages (for n8n)
    # Mount /mcp for MCP Inspector / Claude Desktop
    routes = api_routes + [
        # SSE endpoints at root for n8n MCP integration
        # -> /sse (SSE stream), /messages (POST messages)
        Mount("", app=mcp_sse_app),
        # Streamable HTTP for MCP Inspector / Claude Desktop
        Mount("/mcp", app=mcp_http_app),
    ]
    
    # Create Starlette app with CORS
    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
        on_startup=[lambda: logger.info("PaperStream server starting...")],
    )
    
    return app

# =========================
# ASGI App
# =========================
app = create_app()

# =========================
# Main
# =========================
if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv("PAPERSTREAM_HOST", "0.0.0.0")
    PORT = int(os.getenv("PAPERSTREAM_PORT", "8089"))
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           PaperStream MCP Server v1.0.0                  ║
╠══════════════════════════════════════════════════════════╣
║  Endpoints:                                              ║
║  - MCP SSE (n8n):  http://{HOST}:{PORT}/sse                  ║
║  - MCP Messages:   http://{HOST}:{PORT}/messages             ║
║  - MCP HTTP:       http://{HOST}:{PORT}/mcp                  ║
║  - REST API:       http://{HOST}:{PORT}/api/                 ║
║  - Unity SSE:      http://{HOST}:{PORT}/api/stream/unity     ║
║  - Health:         http://{HOST}:{PORT}/health               ║
╠══════════════════════════════════════════════════════════╣
║  n8n MCP Integration:                                    ║
║  URL: http://{HOST}:{PORT}/sse                               ║
║  Transport: SSE (Server-Sent Events)                     ║
║  Tools: submit_paper, create_rule, process_paper, ...    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=HOST, port=PORT)
