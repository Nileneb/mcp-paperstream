"""
Paper Handler - API endpoints for paper management

Handles:
- POST /api/papers/submit - Submit new paper (n8n webhook)
- GET /api/papers/{paper_id} - Get paper details
- GET /api/papers - List papers
- GET /api/papers/{paper_id}/thumbnail - Get paper thumbnail
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import aiohttp

from ..db import get_db, Paper
from ..processing import generate_thumbnail, ThumbnailGenerator

logger = logging.getLogger(__name__)

# Shared papers directory (from paper-search-mcp downloads)
SHARED_PAPERS_DIR = Path(os.getenv("SHARED_PAPERS_DIR", "/shared/papers"))


class PaperHandler:
    """Handler for paper-related operations"""
    
    def __init__(self):
        self.db = get_db()
        self.download_dir = Path(__file__).parent.parent.parent.parent / "data" / "papers"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir = Path(__file__).parent.parent.parent.parent / "data" / "images"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self._download_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._thumbnail_gen = ThumbnailGenerator(
            width=200,
            height=280,
            cache_dir=self.thumbnail_dir
        )
    
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
    
    def get_thumbnail(
        self,
        paper_id: str,
        width: int = 200,
        height: int = 280,
        as_base64: bool = True,
        regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Get thumbnail for a paper's PDF.
        
        Generates from first page if not cached.
        For Unity: returns base64 encoded PNG.
        
        Args:
            paper_id: Paper identifier
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels
            as_base64: Return as base64 string (default for Unity)
            regenerate: Force regeneration even if cached
        
        Returns:
            {
                "status": "success" | "error" | "no_pdf",
                "paper_id": str,
                "thumbnail_base64": str (if as_base64),
                "thumbnail_path": str (if saved to disk),
                "width": int,
                "height": int
            }
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {
                "status": "error",
                "paper_id": paper_id,
                "message": "Paper not found"
            }
        
        # Check if PDF exists
        pdf_path = None
        if paper.pdf_local_path and Path(paper.pdf_local_path).exists():
            pdf_path = Path(paper.pdf_local_path)
        else:
            # Try to find in shared directory
            pdf_path = self.find_paper_in_shared(paper_id)
        
        if not pdf_path:
            return {
                "status": "no_pdf",
                "paper_id": paper_id,
                "message": "No PDF available for this paper"
            }
        
        # Generate safe filename from paper_id
        safe_id = paper_id.replace("/", "_").replace(":", "_").replace("\\", "_")
        thumb_filename = f"{safe_id}_thumb.png"
        thumb_path = self.thumbnail_dir / thumb_filename
        
        # Check cache unless regenerate requested
        if not regenerate and thumb_path.exists():
            logger.debug(f"Using cached thumbnail for {paper_id}")
            if as_base64:
                import base64
                thumb_base64 = base64.b64encode(thumb_path.read_bytes()).decode("utf-8")
                return {
                    "status": "success",
                    "paper_id": paper_id,
                    "thumbnail_base64": thumb_base64,
                    "thumbnail_path": str(thumb_path),
                    "width": width,
                    "height": height,
                    "cached": True
                }
            else:
                return {
                    "status": "success",
                    "paper_id": paper_id,
                    "thumbnail_path": str(thumb_path),
                    "width": width,
                    "height": height,
                    "cached": True
                }
        
        # Generate new thumbnail
        try:
            if width != 200 or height != 280:
                # Use custom size
                gen = ThumbnailGenerator(width=width, height=height, cache_dir=self.thumbnail_dir)
            else:
                gen = self._thumbnail_gen
            
            saved_path = gen.generate_and_save(
                pdf_path=pdf_path,
                output_path=thumb_path,
                page_num=0,
                output_format="png"
            )
            
            if saved_path:
                # Update database with thumbnail path
                try:
                    with self.db.get_connection() as conn:
                        conn.execute(
                            """
                            UPDATE papers 
                            SET thumbnail_path = ?, thumbnail_generated_at = CURRENT_TIMESTAMP
                            WHERE paper_id = ?
                            """,
                            (str(saved_path), paper_id)
                        )
                except Exception as e:
                    # Table might not have column yet - non-critical
                    logger.debug(f"Could not update thumbnail_path in DB: {e}")
                
                if as_base64:
                    import base64
                    thumb_base64 = base64.b64encode(saved_path.read_bytes()).decode("utf-8")
                    return {
                        "status": "success",
                        "paper_id": paper_id,
                        "thumbnail_base64": thumb_base64,
                        "thumbnail_path": str(saved_path),
                        "width": width,
                        "height": height,
                        "cached": False
                    }
                else:
                    return {
                        "status": "success",
                        "paper_id": paper_id,
                        "thumbnail_path": str(saved_path),
                        "width": width,
                        "height": height,
                        "cached": False
                    }
            else:
                return {
                    "status": "error",
                    "paper_id": paper_id,
                    "message": "Thumbnail generation failed"
                }
                
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {paper_id}: {e}")
            return {
                "status": "error",
                "paper_id": paper_id,
                "message": str(e)
            }
    
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
                    success = await self.download_pdf(paper_id)
                    
                    # Auto-process after successful download
                    if success:
                        try:
                            from ..pipeline.paper_processor import get_paper_processor
                            processor = get_paper_processor()
                            result = await processor.process_paper(paper_id)
                            if "error" in result:
                                logger.error(f"Auto-process failed for {paper_id}: {result['error']}")
                            else:
                                logger.info(f"Auto-processed {paper_id}: {result.get('sections_created', 0)} sections")
                        except Exception as e:
                            logger.error(f"Auto-process error for {paper_id}: {e}")
                            
                except asyncio.TimeoutError:
                    # No more items, stop worker
                    break
        finally:
            self._processing = False
    
    def _find_pdf_for_paper(self, paper_id: str) -> Optional[Path]:
        """
        Try to find PDF for a paper in various locations.
        
        Searches:
        1. Shared papers directory
        2. Local download directory
        
        Tries various filename patterns to match paper_id.
        """
        # Clean paper_id for filename matching
        clean_id = paper_id.replace("/", "_").replace(":", "_").replace("\\", "_")
        
        # Patterns to try
        patterns = [
            f"{paper_id}.pdf",
            f"{clean_id}.pdf",
            f"semantic_{paper_id}.pdf",
            f"semantic_{clean_id}.pdf",
        ]
        
        # Search directories
        search_dirs = [SHARED_PAPERS_DIR, self.download_dir]
        
        for directory in search_dirs:
            if not directory.exists():
                continue
            for pattern in patterns:
                path = directory / pattern
                if path.exists():
                    return path
            
            # Also try partial match (paper_id contained in filename)
            for file in directory.glob("*.pdf"):
                if clean_id in file.name or paper_id in file.name:
                    return file
        
        return None
    
    async def process_pending_papers(self) -> Dict[str, Any]:
        """
        Process all pending papers - finds PDFs and generates embeddings.
        
        1. Find pending papers without sections
        2. Try to locate PDFs in shared directory
        3. Link PDFs and process papers
        
        Returns:
            Summary of processing results
        """
        from ..pipeline.paper_processor import get_paper_processor
        processor = get_paper_processor()
        
        # Find ALL pending papers (not just those with pdf_local_path)
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT p.paper_id, p.pdf_local_path
                FROM papers p
                LEFT JOIN paper_sections s ON p.paper_id = s.paper_id
                WHERE p.status IN ('pending', 'processing')
                GROUP BY p.paper_id
                HAVING COUNT(s.id) = 0
            """)
            papers_to_process = cursor.fetchall()
        
        results = {
            "total": len(papers_to_process),
            "processed": 0,
            "failed": 0,
            "linked": 0,
            "details": []
        }
        
        for paper_id, pdf_path in papers_to_process:
            # Try to find PDF if not already linked
            if not pdf_path or not Path(pdf_path).exists():
                found_pdf = self._find_pdf_for_paper(paper_id)
                if found_pdf:
                    # Link the found PDF
                    with self.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                            (str(found_pdf), paper_id)
                        )
                    pdf_path = str(found_pdf)
                    results["linked"] += 1
                    logger.info(f"Linked PDF for {paper_id}: {found_pdf}")
                else:
                    results["failed"] += 1
                    results["details"].append({"paper_id": paper_id, "status": "no_pdf_found"})
                    continue
            
            # Process the paper
            try:
                result = await processor.process_paper(paper_id)
                if "error" in result:
                    results["failed"] += 1
                    results["details"].append({"paper_id": paper_id, "status": "error", "message": result["error"]})
                else:
                    results["processed"] += 1
                    results["details"].append({
                        "paper_id": paper_id, 
                        "status": "success",
                        "sections": result.get("sections_created", 0)
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"paper_id": paper_id, "status": "error", "message": str(e)})
        
        return results


# Singleton instance
_paper_handler: Optional[PaperHandler] = None


def get_paper_handler() -> PaperHandler:
    """Get singleton paper handler"""
    global _paper_handler
    if _paper_handler is None:
        _paper_handler = PaperHandler()
    return _paper_handler
