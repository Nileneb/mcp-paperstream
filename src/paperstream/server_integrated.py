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
async def download_paper(paper_id: str, auto_process: bool = True) -> Dict[str, Any]:
    """
    Download PDF for a paper and optionally process it automatically.
    
    Args:
        paper_id: Paper ID to download PDF for
        auto_process: If True (default), automatically process after download
                     (extract sections, generate BioBERT embeddings)
    
    Returns:
        Download result with status and processing result if auto_process=True
    
    Example (simplified - one step does everything!):
        # 1. Submit paper with URL
        submit_paper(paper_id="PMC123", pdf_url="https://...")
        # 2. Download AND process in one step
        download_paper(paper_id="PMC123")
        # Paper is now ready with embeddings!
    """
    from .api import get_paper_handler
    from .db import get_db
    from .pipeline import get_paper_processor
    
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        return {"error": f"Paper not found: {paper_id}", "status": "failed"}
    
    if not paper.pdf_url:
        return {"error": f"No PDF URL for paper: {paper_id}", "status": "failed"}
    
    if paper.pdf_local_path:
        from pathlib import Path
        if Path(paper.pdf_local_path).exists():
            result = {
                "status": "already_downloaded",
                "paper_id": paper_id,
                "pdf_path": paper.pdf_local_path
            }
            # Still process if requested and not yet processed
            if auto_process and paper.status != "ready":
                processor = get_paper_processor()
                process_result = await processor.process_paper(paper_id)
                result["auto_processed"] = True
                result["processing_result"] = process_result
            return result
    
    handler = get_paper_handler()
    success = await handler.download_pdf(paper_id)
    
    if success:
        # Refresh paper data
        paper = db.get_paper(paper_id)
        result = {
            "status": "downloaded",
            "paper_id": paper_id,
            "pdf_path": paper.pdf_local_path if paper else None
        }
        
        # Auto-process if requested
        if auto_process:
            logger.info(f"Auto-processing paper {paper_id} after download")
            processor = get_paper_processor()
            process_result = await processor.process_paper(paper_id)
            result["auto_processed"] = True
            result["processing_result"] = process_result
        
        return result
    else:
        return {"error": f"Failed to download PDF for {paper_id}", "status": "failed"}

@mcp.tool()
async def link_paper_pdf(paper_id: str, filename: str = "", auto_process: bool = True) -> Dict[str, Any]:
    """
    Link an already downloaded PDF to a paper and optionally process it.
    
    Use this when the PDF was downloaded via paper-search-mcp tools
    (download_arxiv, download_biorxiv, etc.) and is in the shared volume.
    
    NEW: With auto_process=True (default), this will automatically:
    1. Link the PDF
    2. Process the paper (extract sections, generate BioBERT embeddings)
    3. Make the paper ready for validation jobs
    
    Args:
        paper_id: Paper ID to link (must be submitted first via submit_paper)
        filename: PDF filename (default: tries {paper_id}.pdf)
        auto_process: If True (default), automatically process the paper after linking
    
    Returns:
        Link result with status, pdf_path, and processing result if auto_process=True
    
    Example workflow (simplified):
        # 1. Search and download via paper-search-mcp
        # download_arxiv("2106.12345") -> saves to /shared/papers/2106.12345.pdf
        
        # 2. Submit paper to paperstream
        submit_paper(paper_id="2106.12345", title="My Paper")
        
        # 3. Link AND process in one step!
        link_paper_pdf(paper_id="2106.12345")
        # Paper is now ready with embeddings!
    """
    from .api import get_paper_handler
    from .db import get_db
    from .pipeline import get_paper_processor
    
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
    
    link_result = handler.link_downloaded_paper(paper_id, filename)
    
    # Auto-process if requested and link was successful
    if auto_process and link_result.get("status") == "linked":
        logger.info(f"Auto-processing paper {paper_id} after linking PDF")
        processor = get_paper_processor()
        process_result = await processor.process_paper(paper_id)
        
        return {
            **link_result,
            "auto_processed": True,
            "processing_result": process_result
        }
    
    return link_result

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

async def api_process_pending(request: Request) -> JSONResponse:
    """POST /api/papers/process-pending - Process all pending papers with PDFs"""
    try:
        from .api import get_paper_handler
        handler = get_paper_handler()
        result = await handler.process_pending_papers()
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in process_pending: {e}")
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
    """GET /api/rules/active - Unity endpoint for ALL rules WITH EMBEDDINGS
    
    Returns ALL rules (regardless of is_active flag) because Unity needs to
    check every paper against EVERY rule to find which ones apply.
    """
    from .db import get_db
    import base64
    
    db = get_db()
    rules = db.get_active_rules()  # Returns ALL rules now!
    
    # Format for Unity client - INCLUDE EMBEDDINGS for matching!
    rules_data = []
    for rule in rules:
        rule_dict = {
            "rule_id": rule.rule_id,
            "question": rule.question,
            "threshold": rule.threshold,
            "is_active": True  # Always return as active - Unity needs all!
        }
        
        # CRITICAL: Include embeddings for Unity matching!
        if rule.pos_embedding:
            rule_dict["pos_embedding_b64"] = base64.b64encode(rule.pos_embedding).decode("ascii")
        
        if rule.neg_embedding:
            rule_dict["neg_embedding_b64"] = base64.b64encode(rule.neg_embedding).decode("ascii")
        
        rules_data.append(rule_dict)
    
    return JSONResponse({"rules": rules_data, "count": len(rules_data)})


async def api_get_active_rule(request: Request) -> JSONResponse:
    """
    GET /api/rule/active - DATAMODEL.md contract endpoint
    
    Returns the currently active validation rule for the game.
    Unity shows this as the reference shape at the top of the UI.
    
    Returns:
    {
        "rule": { /* Rule Molecule with chunks */ },
        "threshold": 0.7,
        "question": "Is this a Randomized Controlled Trial?"
    }
    """
    from .api.rule_handler import get_rule_handler
    from .db import get_db
    
    handler = get_rule_handler()
    db = get_db()
    
    # Get first active rule (in production, this would be configurable)
    rules = db.get_active_rules()
    
    if not rules:
        return JSONResponse({
            "error": "No active rules configured",
            "message": "Call load_default_rules() to initialize rules"
        }, status_code=404)
    
    # Get the first rule as the "active" game rule
    active_rule = rules[0]
    
    # Get full molecule representation with chunks
    rule_chunks = handler.get_rule_chunks(active_rule.rule_id)
    
    if not rule_chunks:
        return JSONResponse({"error": "Rule has no chunks"}, status_code=500)
    
    return JSONResponse({
        "rule": rule_chunks,
        "threshold": active_rule.threshold,
        "question": active_rule.question
    })


async def api_get_rule_chunks(request: Request) -> JSONResponse:
    """
    GET /api/rules/{rule_id}/chunks - Get rule as molecule chunks
    GET /api/rule-chunks?rule_id=... - Alternative with query param
    
    Returns rule with 2 chunks (positive/negative) for Unity "Rule-Molecule" visualization:
    {
        "rule_id": "...",
        "question": "...",
        "chunks": [
            {
                "chunk_id": 0,
                "chunk_type": "positive",
                "embedding_b64": "...",
                "color": {"r": 0.2, "g": 0.9, "b": 0.3},  // Green
                "connects_to": [1]
            },
            {
                "chunk_id": 1,
                "chunk_type": "negative", 
                "embedding_b64": "...",
                "color": {"r": 0.9, "g": 0.2, "b": 0.2},  // Red
                "connects_to": []
            }
        ],
        "molecule_config": {"layout": "dipole", ...}
    }
    """
    # Support both path param and query param
    rule_id = request.path_params.get("rule_id")
    if not rule_id:
        rule_id = request.query_params.get("rule_id")
    
    if not rule_id:
        return JSONResponse({"error": "rule_id required"}, status_code=400)
    
    from .api.rule_handler import get_rule_handler
    handler = get_rule_handler()
    
    result = handler.get_rule_chunks(rule_id)
    if not result:
        return JSONResponse({"error": "Rule not found"}, status_code=404)
    
    return JSONResponse(result)


async def api_get_all_rule_chunks(request: Request) -> JSONResponse:
    """
    GET /api/rules/chunks - Get ALL rules as molecule chunks
    
    Returns all active rules with their chunk representations.
    Unity can render all rule-molecules at once.
    """
    from .api.rule_handler import get_rule_handler
    handler = get_rule_handler()
    
    rules_with_chunks = handler.list_rules_with_chunks()
    
    return JSONResponse({
        "rules": rules_with_chunks,
        "count": len(rules_with_chunks),
        "molecule_config": {
            "embedding_dim": 768,
            "layout": "dipole",
            "scale": 1.0
        }
    })


# =========================
# Jobs API - 1 Job = 1 Paper
# =========================

async def api_get_job_stats(request: Request) -> JSONResponse:
    """
    GET /api/jobs/stats - Get job queue statistics
    
    Shows validation progress and round-robin status.
    """
    from .db import get_db
    db = get_db()
    
    with db.get_connection() as conn:
        # Paper status counts
        status_counts = {}
        for row in conn.execute("SELECT status, COUNT(*) as cnt FROM papers GROUP BY status"):
            status_counts[row['status']] = row['cnt']
        
        # Papers with embeddings (ready for validation)
        ready_with_embeddings = conn.execute("""
            SELECT COUNT(DISTINCT p.paper_id) 
            FROM papers p
            JOIN paper_sections ps ON p.paper_id = ps.paper_id
            WHERE p.status = 'ready' AND ps.embedding IS NOT NULL
        """).fetchone()[0]
        
        # Total validations
        total_validations = conn.execute("SELECT COUNT(*) FROM paper_validations").fetchone()[0]
        
        # Validations per paper (round-robin distribution)
        validation_dist = {}
        for row in conn.execute("""
            SELECT p.paper_id, p.title, COUNT(pv.paper_id) as validations
            FROM papers p
            LEFT JOIN paper_validations pv ON p.paper_id = pv.paper_id
            WHERE p.status = 'ready'
            GROUP BY p.paper_id
            ORDER BY validations DESC
            LIMIT 20
        """):
            validation_dist[row['paper_id']] = {
                'title': row['title'][:50] if row['title'] else '?',
                'validations': row['validations']
            }
        
        # Active assignments
        active_assignments = conn.execute("""
            SELECT COUNT(*) FROM paper_assignments 
            WHERE expires_at > CURRENT_TIMESTAMP
        """).fetchone()[0]
    
    return JSONResponse({
        "paper_status": status_counts,
        "ready_with_embeddings": ready_with_embeddings,
        "total_validations": total_validations,
        "active_assignments": active_assignments,
        "validation_distribution": validation_dist,
        "round_robin_enabled": True
    })


async def api_get_next_job(request: Request) -> JSONResponse:
    """
    GET /api/jobs/next - Get next paper to validate
    
    Android calls this to get ONE paper with PDF thumbnail.
    Android already has all rules from /api/rules/active with embeddings.
    Android validates paper locally against all rules.
    
    Returns:
    {
        "status": "assigned",
        "job": {
            "paper_id": "...",
            "title": "...",
            "pdf_thumbnail_b64": "base64...",
            "expires_in_seconds": 600
        }
    }
    """
    device_id = request.query_params.get("device_id")
    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)
    
    from .api.job_handler import get_job_handler
    handler = get_job_handler()
    result = handler.get_next_job(device_id)
    return JSONResponse(result)


async def api_get_paper_chunks(request: Request) -> JSONResponse:
    """
    GET /api/papers/{paper_id}/chunks - Get paper as molecule chunks
    GET /api/chunks?paper_id=... - Alternative with query param (for DOIs with /)
    
    Returns all section embeddings for Unity "Paper-Molecule" visualization:
    {
        "paper_id": "...",
        "title": "...",
        "chunks": [
            {
                "chunk_id": 0,
                "section_name": "abstract",
                "text_preview": "First 500 chars...",
                "embedding_b64": "...",  // 768-dim BioBERT
                "color": {"r": 0.2, "g": 0.6, "b": 0.9},
                "position": {"x": 0, "y": 0, "z": 0},
                "connects_to": [1]  // Links to next chunk
            },
            ...
        ],
        "molecule_config": {
            "chunk_size": 768,
            "layout": "chain",  // or "spiral", "cluster"
            "scale": 1.0
        }
    }
    """
    # Support both path param and query param (for DOIs with /)
    paper_id = request.path_params.get("paper_id")
    if not paper_id:
        paper_id = request.query_params.get("paper_id")
    
    if not paper_id:
        return JSONResponse({"error": "paper_id required"}, status_code=400)
    
    from .api.job_handler import get_job_handler
    handler = get_job_handler()
    
    # Get paper info
    from .db import get_db
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        return JSONResponse({"error": "Paper not found"}, status_code=404)
    
    # Get chunks
    chunks = handler._get_paper_chunks(paper_id)
    
    if not chunks:
        return JSONResponse({
            "error": "Paper not processed yet (no chunks)",
            "paper_id": paper_id
        }, status_code=404)
    
    return JSONResponse({
        "paper_id": paper_id,
        "title": paper.title or "Unknown",
        "chunks": chunks,
        "chunks_count": len(chunks),
        "molecule_config": {
            "embedding_dim": 768,
            "layout": "chain",
            "scale": 1.0,
            "connection_type": "sequential"
        }
    })


async def api_submit_job(request: Request) -> JSONResponse:
    """
    POST /api/jobs/submit - Submit paper validation results
    
    Android sends:
    {
        "device_id": "...",
        "paper_id": "...",
        "results": [
            {"rule_id": "is_rct", "matched": true, "confidence": 0.85, "regions": [[x1,y1,x2,y2]]},
            {"rule_id": "has_placebo", "matched": false, "confidence": 0.2},
            ...
        ]
    }
    
    Returns:
    {
        "accepted": true,
        "paper_id": "...",
        "rules_matched": 5,
        "rules_checked": 18,
        "points_earned": 86
    }
    """
    try:
        body = await request.json()
        device_id = body.get("device_id")
        paper_id = body.get("paper_id")
        results = body.get("results", [])
        
        if not device_id or not paper_id:
            return JSONResponse({"error": "device_id and paper_id required"}, status_code=400)
        
        from .api.job_handler import get_job_handler
        handler = get_job_handler()
        result = handler.submit_results(device_id, paper_id, results)
        
        # Emit SSE event for Unity dashboard
        from .api.sse_stream import get_unity_sse_stream
        sse = get_unity_sse_stream()
        if sse.get_client_count() > 0:
            await sse.emit_validation_result(paper_id, results)
        
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in api_submit_job: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_job_response(request: Request) -> JSONResponse:
    """
    POST /api/jobs/{job_id}/response - DATAMODEL.md contract endpoint
    
    Submit player response for a validation job.
    
    Request body:
    {
        "device_id": "unity_client_xyz",
        "action": "collect",        // or "skip"
        "response_time_ms": 2340
    }
    
    Returns:
    {
        "accepted": true,
        "job_id": "...",
        "action": "collect",
        "points_earned": 10
    }
    """
    try:
        job_id = request.path_params.get("job_id")
        if not job_id:
            return JSONResponse({"error": "job_id required"}, status_code=400)
        
        body = await request.json()
        device_id = body.get("device_id")
        action = body.get("action")  # "collect" or "skip"
        response_time_ms = body.get("response_time_ms", 0)
        
        if not device_id or not action:
            return JSONResponse({"error": "device_id and action required"}, status_code=400)
        
        if action not in ["collect", "skip"]:
            return JSONResponse({"error": "action must be 'collect' or 'skip'"}, status_code=400)
        
        from .api.job_handler import get_job_handler
        from .db import get_db
        
        handler = get_job_handler()
        db = get_db()
        
        # For now, job_id is treated as paper_id (1 job = 1 paper)
        # In future, this could be a separate job tracking system
        paper_id = job_id.replace("job_", "") if job_id.startswith("job_") else job_id
        
        # Convert action to result format
        # "collect" = matched/interested, "skip" = not matched/not interested
        matched = (action == "collect")
        
        # Store result
        result = handler.submit_results(
            device_id=device_id,
            paper_id=paper_id,
            results=[{
                "rule_id": "visual_match",  # Generic rule for visual matching
                "matched": matched,
                "confidence": 1.0 if matched else 0.0,
                "response_time_ms": response_time_ms,
            }]
        )
        
        # Calculate points (faster response = more points)
        base_points = 10 if matched else 5
        speed_bonus = max(0, (10000 - response_time_ms) // 1000)  # Up to 10 bonus points
        points_earned = base_points + speed_bonus
        
        # Emit SSE event
        from .api.sse_stream import get_unity_sse_stream
        sse = get_unity_sse_stream()
        if sse.get_client_count() > 0:
            await sse.emit_validation_result(paper_id, [{
                "device_id": device_id,
                "action": action,
                "response_time_ms": response_time_ms
            }])
        
        return JSONResponse({
            "accepted": True,
            "job_id": job_id,
            "paper_id": paper_id,
            "action": action,
            "points_earned": points_earned,
            "response_time_ms": response_time_ms
        })
        
    except Exception as e:
        logger.error(f"Error in api_job_response: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def jobs_sse_handler(request: Request) -> StreamingResponse:
    """
    SSE /api/jobs/stream - DATAMODEL.md contract endpoint
    
    Streams validation jobs to Unity clients.
    
    Events:
    {
        "event": "job",
        "data": {
            "job_id": "job_abc123",
            "paper_id": "paper_12345",
            "chunk": { /* Single Chunk to validate */ },
            "timeout_ms": 10000
        }
    }
    """
    client_id = request.query_params.get("client_id", f"jobs_{uuid.uuid4().hex[:8]}")
    device_id = request.query_params.get("device_id", client_id)
    
    from .api.sse_stream import get_unity_sse_stream
    from .api.job_handler import get_job_handler
    
    sse = get_unity_sse_stream()
    handler = get_job_handler()
    
    async def job_event_stream():
        # Send connection event
        yield f"event: connected\ndata: {json.dumps({'client_id': client_id, 'device_id': device_id})}\n\n"
        
        job_interval = 5  # seconds between job checks
        last_job_time = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Check for new job periodically
                if current_time - last_job_time >= job_interval:
                    job_result = handler.get_next_job(device_id)
                    
                    if job_result.get("status") == "assigned":
                        job_data = job_result.get("job", {})
                        
                        # Format as DATAMODEL.md contract
                        event_data = {
                            "job_id": f"job_{job_data.get('paper_id', 'unknown')}",
                            "paper_id": job_data.get("paper_id"),
                            "title": job_data.get("title"),
                            "chunk": job_data.get("chunks", [{}])[0] if job_data.get("chunks") else None,
                            "timeout_ms": job_data.get("expires_in_seconds", 600) * 1000,
                        }
                        
                        yield f"event: job\ndata: {json.dumps(event_data)}\n\n"
                    
                    last_job_time = current_time
                
                # Send heartbeat every 30 seconds
                await asyncio.sleep(1)
                if int(current_time) % 30 == 0:
                    yield f"event: heartbeat\ndata: {json.dumps({'ts': int(current_time * 1000)})}\n\n"
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in jobs SSE stream: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        job_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# =========================
# Device API
# =========================

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
    # Check Qdrant connection
    qdrant_status = "unknown"
    try:
        from .db.vector_store import health_check as qdrant_health
        qdrant_info = qdrant_health()
        qdrant_status = qdrant_info.get("status", "unknown")
    except Exception as e:
        qdrant_status = f"error: {e}"
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "qdrant": qdrant_status
    })


# =========================
# Qdrant Vector Store API
# =========================

async def api_qdrant_sync(request: Request) -> JSONResponse:
    """POST /api/qdrant/sync - Sync SQLite data to Qdrant"""
    try:
        from .db.vector_store import (
            init_collections, 
            sync_papers_from_sqlite, 
            sync_rules_from_sqlite,
            health_check as qdrant_health
        )
        
        # Initialize collections if needed
        init_collections()
        
        # Sync papers
        papers_synced = sync_papers_from_sqlite("data/paperstream.db")
        
        # Sync rules
        rules_synced = sync_rules_from_sqlite("data/paperstream.db")
        
        return JSONResponse({
            "status": "synced",
            "papers_synced": papers_synced,
            "rules_synced": rules_synced,
            "qdrant_health": qdrant_health()
        })
    except Exception as e:
        logger.error(f"Qdrant sync failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_qdrant_status(request: Request) -> JSONResponse:
    """GET /api/qdrant/status - Get Qdrant vector store status"""
    try:
        from .db.vector_store import health_check as qdrant_health
        return JSONResponse(qdrant_health())
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "unavailable"}, status_code=503)


async def api_qdrant_search_papers(request: Request) -> JSONResponse:
    """POST /api/qdrant/search/papers - Search similar papers"""
    try:
        body = await request.json()
        embedding = body.get("embedding")  # 768-dim list
        limit = body.get("limit", 10)
        
        if not embedding:
            return JSONResponse({"error": "embedding required"}, status_code=400)
        
        from .db.vector_store import search_similar_papers
        results = search_similar_papers(embedding, limit=limit)
        
        return JSONResponse({
            "results": [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in results
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_qdrant_match_rules(request: Request) -> JSONResponse:
    """POST /api/qdrant/match/rules - Match paper embedding against all rules"""
    try:
        body = await request.json()
        paper_embedding = body.get("paper_embedding")  # 768-dim list
        limit = body.get("limit", 20)
        
        if not paper_embedding:
            return JSONResponse({"error": "paper_embedding required"}, status_code=400)
        
        from .db.vector_store import match_paper_to_rules
        results = match_paper_to_rules(paper_embedding, limit=limit)
        
        return JSONResponse({"matches": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =========================
# Route Definitions
# =========================

api_routes = [
    # Papers (more specific routes first!)
    Route("/api/papers/submit", api_submit_paper, methods=["POST"]),
    Route("/api/papers/process-pending", api_process_pending, methods=["POST"]),  # Process pending papers
    Route("/api/papers/{paper_id}/chunks", api_get_paper_chunks, methods=["GET"]),  # Molecule chunks - BEFORE {paper_id}!
    Route("/api/papers/{paper_id}", api_get_paper, methods=["GET"]),
    Route("/api/papers", api_list_papers, methods=["GET"]),
    Route("/api/chunks", api_get_paper_chunks, methods=["GET"]),  # Alternative with ?paper_id= for DOIs
    
    # Rules - Android/Unity gets ALL rules with embeddings (more specific first!)
    Route("/api/rules/create", api_create_rule, methods=["POST"]),
    Route("/api/rules/chunks", api_get_all_rule_chunks, methods=["GET"]),  # ALL rules as chunks
    Route("/api/rules/active", api_get_active_rules, methods=["GET"]),  # ALL rules with embeddings!
    Route("/api/rules/{rule_id}/chunks", api_get_rule_chunks, methods=["GET"]),  # Single rule chunks
    Route("/api/rules/{rule_id}", api_get_rule_chunks, methods=["GET"]),  # Alias
    Route("/api/rules", api_list_rules, methods=["GET"]),
    Route("/api/rule-chunks", api_get_rule_chunks, methods=["GET"]),  # Alternative with ?rule_id=
    
    # DATAMODEL.md contract: GET /api/rule/active (singular - active game rule)
    Route("/api/rule/active", api_get_active_rule, methods=["GET"]),
    
    # Jobs API - 1 Job = 1 Paper (ROUND ROBIN)
    Route("/api/jobs/stats", api_get_job_stats, methods=["GET"]),  # Queue stats & round-robin info
    Route("/api/jobs/next", api_get_next_job, methods=["GET"]),
    Route("/api/jobs/submit", api_submit_job, methods=["POST"]),
    Route("/api/jobs/stream", jobs_sse_handler, methods=["GET"]),  # DATAMODEL.md: SSE job stream
    Route("/api/jobs/{job_id}/response", api_job_response, methods=["POST"]),  # DATAMODEL.md: job response
    Route("/api/validation/submit", api_submit_job, methods=["POST"]),  # Alias for Unity compatibility
    
    # Devices
    Route("/api/devices/register", api_register_device, methods=["POST"]),
    Route("/api/devices/{device_id}", api_get_device, methods=["GET"]),
    
    # Consensus
    Route("/api/consensus/{paper_id}", api_get_consensus, methods=["GET"]),
    
    # Stats
    Route("/api/stats", api_get_stats, methods=["GET"]),
    
    # Unity SSE
    Route("/api/stream/unity", unity_sse_handler, methods=["GET"]),
    
    # Qdrant Vector Store
    Route("/api/qdrant/status", api_qdrant_status, methods=["GET"]),
    Route("/api/qdrant/sync", api_qdrant_sync, methods=["POST"]),
    Route("/api/qdrant/search/papers", api_qdrant_search_papers, methods=["POST"]),
    Route("/api/qdrant/match/rules", api_qdrant_match_rules, methods=["POST"]),
    
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
