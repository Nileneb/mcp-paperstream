"""
Paper Handler - API endpoints for paper management

Handles:
- POST /api/papers/submit - Submit new paper (n8n webhook)
- GET /api/papers/{paper_id} - Get paper details
- GET /api/papers - List papers
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import aiohttp

from ..db import get_db, Paper

logger = logging.getLogger(__name__)

# Shared papers directory (from paper-search-mcp downloads)
SHARED_PAPERS_DIR = Path(os.getenv("SHARED_PAPERS_DIR", "/shared/papers"))


class PaperHandler:
    """Handler for paper-related operations"""
    
    def __init__(self):
        self.db = get_db()
        self.download_dir = Path(__file__).parent.parent.parent.parent / "data" / "papers"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._download_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
    
    async def submit_paper(
        self,
        paper_id: str,
        title: Optional[str] = None,
        authors: Optional[str] = None,
        journal: Optional[str] = None,
        publication_date: Optional[str] = None,
        pdf_url: Optional[str] = None,
        priority: int = 5,
        source: str = "n8n"
    ) -> Dict[str, Any]:
        """
        Submit a new paper for processing.
        
        Called by n8n webhook: POST /api/papers/submit
        
        Args:
            paper_id: Unique identifier (e.g., PMC12345, DOI)
            title: Paper title
            authors: Author names (comma-separated)
            journal: Journal name
            publication_date: Publication date
            pdf_url: URL to download PDF
            priority: Processing priority (1-10, higher = more urgent)
            source: Source of submission (n8n, manual, etc.)
        
        Returns:
            {
                "status": "accepted" | "exists" | "error",
                "paper_id": str,
                "message": str
            }
        """
        # Check if paper already exists
        existing = self.db.get_paper(paper_id)
        if existing:
            return {
                "status": "exists",
                "paper_id": paper_id,
                "message": f"Paper already exists with status: {existing.status}",
                "current_status": existing.status
            }
        
        # Create paper record
        paper = Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            journal=journal,
            publication_date=publication_date,
            pdf_url=pdf_url,
            priority=priority,
            source=source,
            status="pending"
        )
        
        try:
            paper = self.db.create_paper(paper)
            logger.info(f"Paper submitted: {paper_id} (priority={priority})")
            
            # Queue for download if URL provided
            if pdf_url:
                await self._download_queue.put(paper_id)
                self._start_download_worker()
            
            return {
                "status": "accepted",
                "paper_id": paper_id,
                "message": "Paper queued for processing",
                "priority": priority
            }
            
        except Exception as e:
            logger.error(f"Failed to submit paper {paper_id}: {e}")
            return {
                "status": "error",
                "paper_id": paper_id,
                "message": str(e)
            }
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get paper details"""
        paper = self.db.get_paper(paper_id)
        if paper:
            return paper.to_dict()
        return None
    
    def link_downloaded_paper(self, paper_id: str, filename: str) -> Dict[str, Any]:
        """
        Link an already downloaded PDF (from paper-search-mcp) to a paper.
        
        This allows using PDFs downloaded via paper-search-mcp tools
        (download_arxiv, download_biorxiv, etc.) with paperstream processing.
        
        Args:
            paper_id: Paper ID to link
            filename: PDF filename in the shared papers directory
        
        Returns:
            Status dict with result
        """
        # Check shared directory first
        shared_path = SHARED_PAPERS_DIR / filename
        if shared_path.exists():
            pdf_path = shared_path
        else:
            # Try local download directory
            local_path = self.download_dir / filename
            if local_path.exists():
                pdf_path = local_path
            else:
                return {
                    "status": "error",
                    "paper_id": paper_id,
                    "message": f"PDF not found: {filename}. Searched in {SHARED_PAPERS_DIR} and {self.download_dir}"
                }
        
        # Update paper record
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE papers 
                    SET pdf_local_path = ?, status = 'processing', 
                        downloaded_at = CURRENT_TIMESTAMP
                    WHERE paper_id = ?
                    """,
                    (str(pdf_path), paper_id)
                )
            
            logger.info(f"Linked PDF for {paper_id}: {pdf_path}")
            return {
                "status": "linked",
                "paper_id": paper_id,
                "pdf_path": str(pdf_path)
            }
        except Exception as e:
            logger.error(f"Failed to link PDF for {paper_id}: {e}")
            return {
                "status": "error",
                "paper_id": paper_id,
                "message": str(e)
            }
    
    def find_paper_in_shared(self, paper_id: str) -> Optional[Path]:
        """
        Find a PDF for a paper_id in the shared directory.
        
        Tries common filename patterns:
        - {paper_id}.pdf
        - {paper_id} with / replaced by _
        """
        patterns = [
            f"{paper_id}.pdf",
            f"{paper_id.replace('/', '_')}.pdf",
            f"{paper_id.replace(':', '_')}.pdf",
        ]
        
        for pattern in patterns:
            path = SHARED_PAPERS_DIR / pattern
            if path.exists():
                return path
        
        return None
    
    def list_papers(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List papers, optionally filtered by status"""
        if status:
            papers = self.db.get_papers_by_status(status, limit)
        else:
            # Get all statuses
            papers = []
            for s in ["pending", "downloading", "processing", "ready", "failed"]:
                papers.extend(self.db.get_papers_by_status(s, limit))
        
        return [p.to_dict() for p in papers[:limit]]
    
    def get_sections(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get all sections for a paper"""
        sections = self.db.get_sections_for_paper(paper_id)
        return [s.to_dict() for s in sections]
    
    async def download_pdf(self, paper_id: str) -> bool:
        """
        Download PDF for a paper.
        
        Returns:
            True if successful
        """
        paper = self.db.get_paper(paper_id)
        if not paper or not paper.pdf_url:
            logger.warning(f"Cannot download {paper_id}: no URL")
            return False
        
        self.db.update_paper_status(paper_id, "downloading")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url, timeout=60) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    # Save PDF
                    pdf_path = self.download_dir / f"{paper_id}.pdf"
                    content = await response.read()
                    
                    with open(pdf_path, "wb") as f:
                        f.write(content)
            
            # Update paper record
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE papers 
                    SET pdf_local_path = ?, status = 'processing', 
                        downloaded_at = CURRENT_TIMESTAMP
                    WHERE paper_id = ?
                    """,
                    (str(pdf_path), paper_id)
                )
            
            logger.info(f"Downloaded PDF for {paper_id}: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {paper_id}: {e}")
            self.db.update_paper_status(paper_id, "failed")
            return False
    
    def _start_download_worker(self):
        """Start background download worker if not running"""
        if not self._processing:
            asyncio.create_task(self._download_worker())
    
    async def _download_worker(self):
        """Background worker to process download queue"""
        self._processing = True
        try:
            while True:
                try:
                    paper_id = await asyncio.wait_for(
                        self._download_queue.get(), 
                        timeout=5.0
                    )
                    await self.download_pdf(paper_id)
                except asyncio.TimeoutError:
                    # No more items, stop worker
                    break
        finally:
            self._processing = False


# Singleton instance
_paper_handler: Optional[PaperHandler] = None


def get_paper_handler() -> PaperHandler:
    """Get singleton paper handler"""
    global _paper_handler
    if _paper_handler is None:
        _paper_handler = PaperHandler()
    return _paper_handler
