"""
Paper Processing Pipeline

Handles the complete paper processing workflow:
1. PDF Download
2. PDF → PNG Rendering (PyMuPDF)
3. Text Extraction & Section Detection
4. BioBERT Embedding Generation
5. Voxel Grid Conversion for Unity

"""

import asyncio
import logging
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

# Optional: PyMuPDF for PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

from ..db import get_db, Paper, PaperSection
from ..handlers import get_biobert_handler

logger = logging.getLogger(__name__)


# Section patterns for scientific papers
SECTION_PATTERNS = {
    "abstract": r"(?i)^abstract\s*$|^summary\s*$",
    "introduction": r"(?i)^introduction\s*$|^background\s*$",
    "methods": r"(?i)^methods?\s*$|^materials?\s+and\s+methods?\s*$|^methodology\s*$",
    "results": r"(?i)^results?\s*$|^findings\s*$",
    "discussion": r"(?i)^discussion\s*$",
    "conclusion": r"(?i)^conclusions?\s*$|^summary\s+and\s+conclusions?\s*$",
    "references": r"(?i)^references\s*$|^bibliography\s*$",
}

# Colors for sections (RGB normalized 0-1)
SECTION_COLORS = {
    "abstract": (0.2, 0.6, 0.9),      # Blue
    "introduction": (0.3, 0.8, 0.3),  # Green
    "methods": (0.9, 0.7, 0.2),       # Orange
    "results": (0.8, 0.3, 0.3),       # Red
    "discussion": (0.6, 0.3, 0.8),    # Purple
    "conclusion": (0.9, 0.5, 0.7),    # Pink
    "references": (0.5, 0.5, 0.5),    # Gray
    "other": (0.7, 0.7, 0.7),         # Light Gray
}


@dataclass
class ExtractedSection:
    """A section extracted from a PDF"""
    name: str
    text: str
    page_number: int
    start_y: float = 0.0
    end_y: float = 0.0


def sanitize_paper_id(paper_id: str) -> str:
    """
    Sanitize paper ID for use in file paths.
    Replaces characters that are problematic in file systems.
    """
    # Replace common problematic characters
    sanitized = paper_id.replace('/', '_').replace(':', '_').replace('\\', '_')
    sanitized = sanitized.replace('<', '_').replace('>', '_').replace('"', '_')
    sanitized = sanitized.replace('|', '_').replace('?', '_').replace('*', '_')
    return sanitized


class PaperProcessor:
    """
    Processes scientific papers from PDF to Unity-ready voxels.
    """
    
    def __init__(self):
        self.db = get_db()
        self._biobert = None
        
        # Directories
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.papers_dir = self.base_dir / "data" / "papers"
        self.images_dir = self.base_dir / "data" / "images"
        
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def biobert(self):
        """Lazy load BioBERT"""
        if self._biobert is None:
            self._biobert = get_biobert_handler()
        return self._biobert
    
    async def process_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Full processing pipeline for a paper.
        
        Steps:
        1. Load PDF
        2. Extract text and detect sections
        3. Render pages to images
        4. Generate embeddings
        5. Convert to voxel grids
        6. Create validation jobs
        
        Args:
            paper_id: Paper ID to process
        
        Returns:
            Processing result with stats
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper not found: {paper_id}"}
        
        if not paper.pdf_local_path:
            return {"error": f"No PDF file for paper: {paper_id}"}
        
        pdf_path = Path(paper.pdf_local_path)
        if not pdf_path.exists():
            return {"error": f"PDF file not found: {pdf_path}"}
        
        if not HAS_PYMUPDF:
            return {"error": "PyMuPDF not installed. Run: pip install PyMuPDF"}
        
        self.db.update_paper_status(paper_id, "processing")
        
        try:
            # 1. Extract sections
            sections = self._extract_sections(pdf_path)
            logger.info(f"Extracted {len(sections)} sections from {paper_id}")
            
            # 2. Render pages to images
            image_paths = self._render_pages(pdf_path, paper_id)
            logger.info(f"Rendered {len(image_paths)} pages for {paper_id}")
            
            # 3. Generate embeddings and create section records
            section_records = []
            for section in sections:
                # Generate embedding
                embedding = self.biobert.embed(section.text[:512])  # First 512 chars
                
                # Convert to voxel grid
                voxel_grid = self._embedding_to_voxels(embedding)
                
                # Get color
                color = SECTION_COLORS.get(section.name, SECTION_COLORS["other"])
                
                # Create section record
                section_record = PaperSection(
                    paper_id=paper_id,
                    section_name=section.name,
                    section_text=section.text,
                    page_number=section.page_number,
                    image_path=str(image_paths[section.page_number - 1]) if section.page_number <= len(image_paths) else None,
                    voxel_data=json.dumps(voxel_grid),
                    color_r=color[0],
                    color_g=color[1],
                    color_b=color[2],
                )
                section_record.set_embedding_array(np.array(embedding))
                
                section_record = self.db.create_section(section_record)
                section_records.append(section_record)
            
            # 4. Update paper status
            self.db.update_paper_status(paper_id, "ready")
            with self.db.get_connection() as conn:
                conn.execute(
                    "UPDATE papers SET processed_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
                    (paper_id,)
                )
            
            # 5. Create validation jobs for this paper
            jobs_result = self._create_validation_jobs(paper_id)
            jobs_created = jobs_result.get("jobs_created", 0)
            
            logger.info(f"Paper {paper_id} processed successfully, {jobs_created} jobs created")
            
            return {
                "paper_id": paper_id,
                "status": "ready",
                "sections_created": len(section_records),
                "pages_rendered": len(image_paths),
                "jobs_created": jobs_created,
                "sections": [s.to_dict() for s in section_records]
            }
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper_id}: {e}")
            self.db.update_paper_status(paper_id, "failed")
            return {"error": str(e), "paper_id": paper_id}
    
    def _extract_sections(self, pdf_path: Path) -> List[ExtractedSection]:
        """
        Extract text and detect sections from PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of extracted sections
        """
        doc = fitz.open(str(pdf_path))
        sections = []
        current_section = None
        current_text = []
        current_page = 1
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] != 0:  # Skip non-text blocks
                    continue
                
                for line in block.get("lines", []):
                    text = " ".join(span["text"] for span in line.get("spans", []))
                    text = text.strip()
                    
                    if not text:
                        continue
                    
                    # Check if this is a section header
                    detected_section = self._detect_section_header(text)
                    
                    if detected_section:
                        # Save previous section
                        if current_section and current_text:
                            sections.append(ExtractedSection(
                                name=current_section,
                                text=" ".join(current_text),
                                page_number=current_page
                            ))
                        
                        # Start new section
                        current_section = detected_section
                        current_text = []
                        current_page = page_num
                    else:
                        # Add to current section
                        current_text.append(text)
        
        # Save last section
        if current_section and current_text:
            sections.append(ExtractedSection(
                name=current_section,
                text=" ".join(current_text),
                page_number=current_page
            ))
        
        doc.close()
        
        # If no sections detected, create one "full_text" section
        if not sections:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            sections.append(ExtractedSection(
                name="full_text",
                text=full_text[:10000],  # Limit to 10k chars
                page_number=1
            ))
        
        return sections
    
    def _detect_section_header(self, text: str) -> Optional[str]:
        """Detect if text is a section header"""
        text_clean = text.strip()
        
        # Skip if too long (headers are usually short)
        if len(text_clean) > 50:
            return None
        
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, text_clean):
                return section_name
        
        return None
    
    def _render_pages(
        self,
        pdf_path: Path,
        paper_id: str,
        resolution: int = 512
    ) -> List[Path]:
        """
        Render PDF pages to PNG images.
        
        Args:
            pdf_path: Path to PDF
            paper_id: Paper ID for naming
            resolution: Target resolution (width/height)
        
        Returns:
            List of image paths
        """
        doc = fitz.open(str(pdf_path))
        image_paths = []
        
        # Create paper-specific image directory (sanitize paper_id for filesystem)
        safe_paper_id = sanitize_paper_id(paper_id)
        paper_img_dir = self.images_dir / safe_paper_id
        paper_img_dir.mkdir(exist_ok=True)
        
        for page_num, page in enumerate(doc, 1):
            # Calculate zoom for target resolution
            rect = page.rect
            zoom = resolution / max(rect.width, rect.height)
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            img_path = paper_img_dir / f"page_{page_num:03d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
        
        doc.close()
        return image_paths
    
    def _embedding_to_voxels(
        self,
        embedding: List[float],
        grid_size: Tuple[int, int, int] = (8, 8, 12)
    ) -> List[List[List[float]]]:
        """
        Convert 768-dim embedding to 8x8x12 voxel grid.
        
        Each voxel gets a value representing embedding activation.
        
        Args:
            embedding: 768-dim embedding vector
            grid_size: Target voxel grid dimensions
        
        Returns:
            3D list representing voxel grid
        """
        x, y, z = grid_size
        total_voxels = x * y * z  # 768
        
        # Ensure embedding has right size
        emb_array = np.array(embedding, dtype=np.float32)
        if len(emb_array) != total_voxels:
            # Interpolate or truncate
            indices = np.linspace(0, len(emb_array) - 1, total_voxels).astype(int)
            emb_array = emb_array[indices]
        
        # Normalize to 0-1
        emb_min = emb_array.min()
        emb_max = emb_array.max()
        if emb_max > emb_min:
            emb_array = (emb_array - emb_min) / (emb_max - emb_min)
        else:
            emb_array = np.zeros_like(emb_array)
        
        # Reshape to 3D grid
        voxel_grid = emb_array.reshape((x, y, z))
        
        # Convert to nested lists for JSON
        return voxel_grid.tolist()
    
    async def process_pending_papers(self, limit: int = 10) -> Dict[str, Any]:
        """
        Process all pending papers.
        
        Args:
            limit: Maximum papers to process
        
        Returns:
            Processing summary
        """
        papers = self.db.get_papers_by_status("pending", limit)
        
        results = {
            "processed": 0,
            "failed": 0,
            "details": []
        }
        
        for paper in papers:
            if paper.pdf_local_path:
                result = await self.process_paper(paper.paper_id)
                results["details"].append(result)
                
                if "error" in result:
                    results["failed"] += 1
                else:
                    results["processed"] += 1
        
        return results
    
    def _create_validation_jobs(self, paper_id: str) -> Dict[str, Any]:
        """
        Create validation jobs for all sections × rules of a paper.
        
        Args:
            paper_id: Paper ID to create jobs for
        
        Returns:
            Dict with jobs_created count
        """
        import uuid
        from ..db import ValidationJob
        
        # Get all sections for this paper
        sections = self.db.get_sections_for_paper(paper_id)
        if not sections:
            logger.warning(f"No sections found for paper {paper_id}")
            return {"jobs_created": 0, "error": "No sections found"}
        
        # Get all active rules
        rules = self.db.get_active_rules()
        if not rules:
            logger.warning(f"No active rules found for creating jobs")
            return {"jobs_created": 0, "error": "No active rules"}
        
        jobs_created = 0
        
        for section in sections:
            for rule in rules:
                job = ValidationJob(
                    job_id=f"job_{uuid.uuid4().hex[:12]}",
                    paper_id=paper_id,
                    section_id=section.id,
                    rule_id=rule.rule_id,
                    status="pending"
                )
                
                try:
                    self.db.create_job(job)
                    jobs_created += 1
                except Exception as e:
                    logger.warning(f"Failed to create job: {e}")
        
        logger.info(f"Created {jobs_created} validation jobs for paper {paper_id} ({len(sections)} sections × {len(rules)} rules)")
        
        return {
            "jobs_created": jobs_created,
            "paper_id": paper_id,
            "sections": len(sections),
            "rules": len(rules)
        }


# Singleton instance
_processor: Optional[PaperProcessor] = None


def get_paper_processor() -> PaperProcessor:
    """Get singleton paper processor"""
    global _processor
    if _processor is None:
        _processor = PaperProcessor()
    return _processor
