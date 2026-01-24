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
    
    WORKFLOW OPTIONS:
    
    === Option A: Direct PDF URL ===
    1. load_default_rules() - Load validation rules (once per session)
    2. submit_paper(paper_id, pdf_url, title) - Submit paper with PDF URL
    3. download_paper(paper_id) - Download the PDF file
    4. process_paper(paper_id) - Extract sections, generate embeddings, create jobs
    
    === Option B: Use paper-search-mcp downloads (RECOMMENDED) ===
    1. load_default_rules() - Load validation rules (once per session)
    2. [paper-search-mcp] search_arxiv/biorxiv/pubmed(...) - Find papers
    3. [paper-search-mcp] download_arxiv/biorxiv(...) - Download to shared volume
    4. submit_paper(paper_id, title) - Submit paper (no URL needed)
    5. link_paper_pdf(paper_id) - Link downloaded PDF from shared volume
    6. process_paper(paper_id) - Extract sections, generate embeddings, create jobs
    
    === Option C: Batch processing ===
    1. load_default_rules()
    2. submit_paper(...) for each paper
    3. process_all_pending() - Downloads + processes ALL pending papers
    
    Available Tools:
    - submit_paper: Submit paper for processing
    - download_paper: Download PDF from submitted url
    - link_paper_pdf: Link PDF from shared volume (from paper-search-mcp)
    - list_shared_papers: List PDFs available in shared volume
    - process_paper: Extract sections, embeddings, create validation jobs
    - process_all_pending: Batch process all pending papers
    - create_rule: Create custom validation rule
    - load_default_rules: Load 17 predefined validation rules
    - create_jobs: Manually create jobs (usually automatic via process_paper)
    - get_job_stats: Get job statistics (pending/assigned/completed)
    - get_paper_status: Get paper validation status
    - get_system_stats: Get system statistics
    - get_leaderboard: Get gamification leaderboard
    
    SHARED VOLUME: PDFs downloaded via paper-search-mcp are stored in /shared/papers
    and can be linked using link_paper_pdf().
    """,
    version="1.0.0",
)

# =========================
# MCP Tools
# =========================

@mcp.tool()
async def submit_paper(
    paper_id: str,
    title: str = "",
    pdf_url: str = "",
    priority: str = "5",
    source: str = "mcp"
) -> Dict[str, Any]:
    """
    Submit a new paper for processing and validation.
    
    Args:
        paper_id: Unique identifier (e.g. "PMC12345", "10.1234/example.doi")
        title: Paper title. Leave empty if unknown.
        pdf_url: URL to download PDF. Leave empty if not available.
        priority: Processing priority 1-10 as string (higher = more urgent, default "5")
        source: Source of submission (e.g. "n8n", "manual", default "mcp")
    
    Returns:
        Submission result with status and paper_id
    
    Example:
        submit_paper(
            paper_id="PMC12345",
            title="A Randomized Trial of Treatment X",
            pdf_url="https://example.com/paper.pdf",
            priority="8",
            source="n8n"
        )
    """
    from .api import get_paper_handler
    handler = get_paper_handler()
    
    # Parse priority from string
    try:
        priority_int = int(priority)
    except (ValueError, TypeError):
        priority_int = 5
    
    return await handler.submit_paper(
        paper_id=paper_id,
        title=title if title else None,
        pdf_url=pdf_url if pdf_url else None,
        priority=priority_int,
        source=source
    )

@mcp.tool()
async def create_rule(
    rule_id: str,
    question: str,
    positive_phrases: str,
    negative_phrases: str = "",
    threshold: str = "0.75"
) -> Dict[str, Any]:
    """
    Create a validation rule with BioBERT embeddings.
    
    Args:
        rule_id: Unique rule identifier (e.g. "is_rct", "has_control_group")
        question: The validation question (e.g. "Is this a randomized controlled trial?")
        positive_phrases: Comma-separated phrases indicating positive match (e.g. "randomized controlled trial, RCT, clinical trial")
        negative_phrases: Comma-separated phrases indicating negative match (e.g. "review, meta-analysis"). Optional.
        threshold: Similarity threshold 0.0-1.0 as string (default "0.75")
    
    Returns:
        Rule creation result with embedding dimensions
    
    Example:
        create_rule(
            rule_id="is_rct",
            question="Is this a randomized controlled trial?",
            positive_phrases="randomized controlled trial, RCT, clinical trial, phase I, phase II",
            negative_phrases="review article, meta-analysis, editorial",
            threshold="0.7"
        )
    """
    from .api import get_rule_handler
    handler = get_rule_handler()
    
    # Parse comma-separated strings into lists
    pos_list = [p.strip() for p in positive_phrases.split(",") if p.strip()]
    neg_list = [n.strip() for n in negative_phrases.split(",") if n.strip()] if negative_phrases else None
    
    # Parse threshold from string
    try:
        threshold_float = float(threshold)
    except (ValueError, TypeError):
        threshold_float = 0.75
    
    return handler.create_rule(
        rule_id=rule_id,
        question=question,
        positive_phrases=pos_list,
        negative_phrases=neg_list,
        threshold=threshold_float
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
async def get_leaderboard(limit: str = "10") -> Dict[str, Any]:
    """
    Get gamification leaderboard.
    
    Args:
        limit: Number of top players to return as string (default "10")
    
    Returns:
        Leaderboard entries
    """
    from .db import get_db
    db = get_db()
    
    try:
        limit_int = int(limit)
    except (ValueError, TypeError):
        limit_int = 10
    
    entries = db.get_leaderboard(limit_int)
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
async def download_paper(paper_id: str) -> Dict[str, Any]:
    """
    Download PDF for a paper that was previously submitted.
    
    MUST be called after submit_paper if a pdf_url was provided.
    The PDF must be downloaded before process_paper can work.
    
    Args:
        paper_id: Paper ID to download PDF for
    
    Returns:
        Download result with status
    
    Example:
        # 1. Submit paper with URL
        submit_paper(paper_id="PMC123", pdf_url="https://...")
        # 2. Download the PDF
        download_paper(paper_id="PMC123")
        # 3. Now process it
        process_paper(paper_id="PMC123")
    """
    from .api import get_paper_handler
    from .db import get_db
    
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        return {"error": f"Paper not found: {paper_id}", "status": "failed"}
    
    if not paper.pdf_url:
        return {"error": f"No PDF URL for paper: {paper_id}", "status": "failed"}
    
    if paper.pdf_local_path:
        from pathlib import Path
        if Path(paper.pdf_local_path).exists():
            return {
                "status": "already_downloaded",
                "paper_id": paper_id,
                "pdf_path": paper.pdf_local_path
            }
    
    handler = get_paper_handler()
    success = await handler.download_pdf(paper_id)
    
    if success:
        # Refresh paper data
        paper = db.get_paper(paper_id)
        return {
            "status": "downloaded",
            "paper_id": paper_id,
            "pdf_path": paper.pdf_local_path if paper else None
        }
    else:
        return {"error": f"Failed to download PDF for {paper_id}", "status": "failed"}

@mcp.tool()
async def link_paper_pdf(paper_id: str, filename: str = "") -> Dict[str, Any]:
    """
    Link an already downloaded PDF to a paper for processing.
    
    Use this when the PDF was downloaded via paper-search-mcp tools
    (download_arxiv, download_biorxiv, etc.) and is in the shared volume.
    
    Args:
        paper_id: Paper ID to link (must be submitted first via submit_paper)
        filename: PDF filename (default: tries {paper_id}.pdf)
    
    Returns:
        Link result with status and pdf_path
    
    Example workflow:
        # 1. Search and download via paper-search-mcp
        # download_arxiv("2106.12345") -> saves to /shared/papers/2106.12345.pdf
        
        # 2. Submit paper to paperstream
        submit_paper(paper_id="2106.12345", title="My Paper")
        
        # 3. Link the already downloaded PDF
        link_paper_pdf(paper_id="2106.12345")
        # OR with explicit filename:
        link_paper_pdf(paper_id="2106.12345", filename="2106.12345.pdf")
        
        # 4. Process the paper
        process_paper(paper_id="2106.12345")
    """
    from .api import get_paper_handler
    from .db import get_db
    
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        return {"error": f"Paper not found: {paper_id}. Submit it first with submit_paper()", "status": "failed"}
    
    handler = get_paper_handler()
    
    # If no filename specified, try to find it
    if not filename:
        found_path = handler.find_paper_in_shared(paper_id)
        if found_path:
            filename = found_path.name
        else:
            filename = f"{paper_id}.pdf"
    
    return handler.link_downloaded_paper(paper_id, filename)

@mcp.tool()
async def list_shared_papers() -> Dict[str, Any]:
    """
    List all PDFs available in the shared papers directory.
    
    These are papers downloaded via paper-search-mcp that can be linked
    to paperstream for processing.
    
    Returns:
        Dict with list of available PDF files
    """
    import os
    from pathlib import Path
    
    shared_dir = Path(os.getenv("SHARED_PAPERS_DIR", "/shared/papers"))
    
    if not shared_dir.exists():
        return {
            "status": "warning",
            "message": f"Shared papers directory does not exist: {shared_dir}",
            "files": []
        }
    
    pdf_files = list(shared_dir.glob("*.pdf"))
    
    return {
        "status": "ok",
        "directory": str(shared_dir),
        "count": len(pdf_files),
        "files": [f.name for f in pdf_files]
    }

@mcp.tool()
async def process_paper(paper_id: str) -> Dict[str, Any]:
    """
    Process a paper: extract sections, generate BioBERT embeddings, create voxels.
    
    IMPORTANT: The PDF must be downloaded first! Call download_paper() or link_paper_pdf() before this.
    
    This tool:
    1. Extracts text sections from the PDF (abstract, methods, results, etc.)
    2. Generates BioBERT embeddings for each section
    3. Creates voxel grids for Unity visualization
    4. Automatically creates validation jobs if rules exist
    
    Args:
        paper_id: Paper ID to process (must have PDF downloaded)
    
    Returns:
        Processing result with sections created and jobs generated
    
    Example:
        # Complete workflow:
        submit_paper(paper_id="PMC123", pdf_url="https://...")
        download_paper(paper_id="PMC123")  # Download PDF first!
        process_paper(paper_id="PMC123")   # Now process
    """
    from .pipeline import get_paper_processor
    from .db import get_db
    from .api import get_paper_handler
    
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        return {"error": f"Paper not found: {paper_id}"}
    
    # Auto-download if URL exists but PDF not downloaded
    if paper.pdf_url and not paper.pdf_local_path:
        logger.info(f"Auto-downloading PDF for {paper_id}")
        handler = get_paper_handler()
        success = await handler.download_pdf(paper_id)
        if not success:
            return {"error": f"Failed to download PDF for {paper_id}. Check pdf_url."}
        # Refresh paper data
        paper = db.get_paper(paper_id)
    
    if not paper.pdf_local_path:
        return {"error": f"No PDF file for paper: {paper_id}. Submit with pdf_url or download manually."}
    
    processor = get_paper_processor()
    return await processor.process_paper(paper_id)

@mcp.tool()
async def process_all_pending() -> Dict[str, Any]:
    """
    Process all papers that have PDFs but haven't been processed yet.
    
    This is a batch operation that:
    1. Finds all papers with status 'pending' or 'downloading' that have a pdf_url
    2. Downloads PDFs if needed
    3. Processes each paper (extract sections, embeddings)
    4. Creates validation jobs
    
    Returns:
        Batch processing result with counts
    """
    from .db import get_db
    from .api import get_paper_handler
    from .pipeline import get_paper_processor
    
    db = get_db()
    handler = get_paper_handler()
    processor = get_paper_processor()
    
    results = {
        "downloaded": 0,
        "processed": 0,
        "jobs_created": 0,
        "failed": 0,
        "errors": []
    }
    
    # Get pending papers with URLs
    pending = db.get_papers_by_status("pending", limit=100)
    downloading = db.get_papers_by_status("downloading", limit=100)
    all_papers = pending + downloading
    
    for paper in all_papers:
        try:
            # Download if needed
            if paper.pdf_url and not paper.pdf_local_path:
                success = await handler.download_pdf(paper.paper_id)
                if success:
                    results["downloaded"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Download failed: {paper.paper_id}")
                    continue
            
            # Refresh and process
            paper = db.get_paper(paper.paper_id)
            if paper and paper.pdf_local_path:
                result = await processor.process_paper(paper.paper_id)
                if "error" not in result:
                    results["processed"] += 1
                    results["jobs_created"] += result.get("jobs_created", 0)
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{paper.paper_id}: {result['error']}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{paper.paper_id}: {str(e)}")
    
    return results

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

@mcp.tool()
async def create_jobs(paper_id: str = "") -> Dict[str, Any]:
    """
    Create validation jobs for a paper or all ready papers.
    
    Jobs are created for each combination of paper sections × active rules.
    These jobs are then distributed to Android/Unity devices for validation.
    
    Args:
        paper_id: Paper ID to create jobs for. Leave empty to create jobs for ALL ready papers.
    
    Returns:
        Job creation result with counts
    
    Example:
        create_jobs(paper_id="PMC12345")  # Jobs for specific paper
        create_jobs()  # Jobs for all ready papers without jobs
    """
    from .db import get_db
    from .pipeline import get_paper_processor
    
    db = get_db()
    processor = get_paper_processor()
    
    if paper_id:
        # Create jobs for specific paper
        paper = db.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper not found: {paper_id}"}
        if paper.status != "ready":
            return {"error": f"Paper not ready (status: {paper.status}). Process it first."}
        
        result = processor._create_validation_jobs(paper_id)
        return result
    else:
        # Create jobs for all ready papers that don't have jobs yet
        papers = db.get_papers_by_status("ready", limit=1000)
        
        total_jobs = 0
        papers_processed = 0
        errors = []
        
        for paper in papers:
            # Check if paper already has jobs
            with db.get_connection() as conn:
                existing = conn.execute(
                    "SELECT COUNT(*) FROM validation_jobs WHERE paper_id = ?",
                    (paper.paper_id,)
                ).fetchone()[0]
            
            if existing == 0:
                result = processor._create_validation_jobs(paper.paper_id)
                if "error" not in result:
                    total_jobs += result.get("jobs_created", 0)
                    papers_processed += 1
                else:
                    errors.append({"paper_id": paper.paper_id, "error": result["error"]})
        
        return {
            "papers_processed": papers_processed,
            "total_jobs_created": total_jobs,
            "errors": errors if errors else None
        }

@mcp.tool()
async def get_job_stats() -> Dict[str, Any]:
    """
    Get validation job statistics.
    
    Returns:
        Job counts by status, papers with/without jobs
    """
    from .db import get_db
    db = get_db()
    
    with db.get_connection() as conn:
        # Jobs by status
        job_stats = {}
        for status in ["pending", "assigned", "completed"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM validation_jobs WHERE status = ?",
                (status,)
            ).fetchone()[0]
            job_stats[status] = count
        
        # Total jobs
        job_stats["total"] = sum(job_stats.values())
        
        # Papers with jobs
        papers_with_jobs = conn.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM validation_jobs"
        ).fetchone()[0]
        
        # Ready papers without jobs
        ready_papers = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE status = 'ready'"
        ).fetchone()[0]
        
        papers_without_jobs = conn.execute(
            """
            SELECT COUNT(*) FROM papers p 
            WHERE p.status = 'ready' 
            AND NOT EXISTS (SELECT 1 FROM validation_jobs j WHERE j.paper_id = p.paper_id)
            """
        ).fetchone()[0]
    
    return {
        "jobs": job_stats,
        "papers_with_jobs": papers_with_jobs,
        "ready_papers": ready_papers,
        "papers_without_jobs": papers_without_jobs
    }

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

async def api_get_active_rules(request: Request) -> JSONResponse:
    """GET /api/rules/active - Unity endpoint for active rules"""
    from .db import get_db
    db = get_db()
    rules = db.get_active_rules()
    
    # Format for Unity client
    rules_data = []
    for rule in rules:
        rules_data.append({
            "rule_id": rule.rule_id,
            "question": rule.question,
            "threshold": rule.threshold,
            "is_active": rule.is_active
        })
    
    return JSONResponse({"rules": rules_data, "count": len(rules_data)})

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
    Route("/api/rules/active", api_get_active_rules, methods=["GET"]),  # Unity endpoint
    Route("/api/rules", api_list_rules, methods=["GET"]),
    
    # Jobs (Android/Unity)
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
