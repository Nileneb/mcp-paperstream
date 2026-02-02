"""
Job Handler - 1 JOB = 1 PAPER

CORRECT Design:
- BioBERT runs ONLY on server (paperstream MCP)
- Server creates paper embeddings from PDF text
- Android requests job → gets ONE paper with its EMBEDDING (768-dim, base64)
- Android already has ALL rule embeddings (from /api/rules/active)
- Android compares embeddings locally (cosine similarity - lightweight!)
- Android returns: which rules matched with what confidence

DATA MODEL (see core/data_model.py):
- Text → Embedding (768-dim BioBERT)
- Embedding → Chunk (container with voxels)
- Chunk → Voxels (8x8x12 grid, visible in Unity)

Endpoints:
- GET /api/jobs/next?device_id=xxx - Get next paper with embedding
- POST /api/jobs/submit - Submit results (which rules matched)
"""

import base64
import json
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from ..db import get_db
from ..core import VoxelGrid, SECTION_COLORS

logger = logging.getLogger(__name__)


class JobHandler:
    """
    Job Handler: 1 Job = 1 Paper
    
    Flow:
    1. Android calls GET /api/jobs/next?device_id=xxx
    2. Server returns ONE paper with its BioBERT embedding (768-dim, base64)
    3. Android already has all Rule embeddings (pos + neg)
    4. Android computes cosine similarity locally (fast!)
    5. Android calls POST /api/jobs/submit with results
    
    Why embeddings instead of raw text/PDF?
    - BioBERT model is ~250MB, too large for mobile
    - Embedding comparison (cosine similarity) is very fast
    - Semantic meaning is preserved in the 768-dim vector
    """
    
    def __init__(self):
        self.db = get_db()
        self.job_ttl_seconds = 600  # 10 minutes per paper
    
    def get_next_job(self, device_id: str) -> Dict[str, Any]:
        """
        Get next paper for validation.
        
        Returns:
            {
                "status": "assigned" | "no_jobs",
                "job": {
                    "paper_id": str,
                    "title": str,
                    "paper_embedding_b64": str (768-dim float32 as base64),
                    "paper_text": str (abstract/intro for keyword matching),
                    "expires_in_seconds": int
                }
            }
        
        NOTE: paper_text is included for HYBRID matching (embedding + keywords).
        Pure embedding matching doesn't work well with BioBERT - see MATCHING_ALGORITHM.py
        """
        # Update device last_seen
        self.db.update_device_seen(device_id)
        
        # Get next paper that needs validation
        paper = self._get_next_paper(device_id)
        
        if not paper:
            return {
                "status": "no_jobs",
                "message": "No papers available for validation",
                "device_id": device_id
            }
        
        # Mark as assigned
        self._assign_paper(paper["paper_id"], device_id)
        
        # Get paper embedding and text (created during paper processing)
        embedding_b64, paper_text = self._get_paper_embedding_and_text(paper["paper_id"])
        
        if not embedding_b64:
            logger.warning(f"Paper {paper['paper_id']} has no embedding - skipping")
            return {
                "status": "no_jobs",
                "message": "Paper not yet processed (no embedding)",
                "device_id": device_id
            }
        
        # Get all section chunks for molecule visualization
        chunks = self._get_paper_chunks(paper["paper_id"])
        
        return {
            "status": "assigned",
            "job": {
                "paper_id": paper["paper_id"],
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors"),
                "journal": paper.get("journal"),
                "paper_embedding_b64": embedding_b64,  # 768-dim BioBERT embedding (primary/abstract)
                "paper_text": paper_text,  # For keyword matching (hybrid approach)
                # NEW: Chunk embeddings for Unity molecule visualization
                "chunks": chunks,  # Array of section chunks with embeddings + positions
                "chunks_count": len(chunks),
                "expires_in_seconds": self.job_ttl_seconds
            },
            "device_id": device_id
        }
    
    def submit_results(
        self,
        device_id: str,
        paper_id: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Submit validation results for a paper.
        
        Args:
            device_id: Device that validated
            paper_id: Paper that was validated  
            results: List of rule matches:
                [
                    {
                        "rule_id": "is_rct",
                        "matched": true,
                        "confidence": 0.85,
                        "regions": [[x1,y1,x2,y2], ...]  # where found on PDF
                    },
                    ...
                ]
        
        Returns:
            {
                "accepted": bool,
                "paper_id": str,
                "rules_matched": int,
                "points_earned": int
            }
        """
        try:
            # Verify assignment
            if not self._verify_assignment(paper_id, device_id):
                return {
                    "accepted": False,
                    "error": "Paper not assigned to this device"
                }
            
            # Count matches
            rules_matched = sum(1 for r in results if r.get("matched", False))
            
            # Store each result
            for result in results:
                self._store_result(
                    paper_id=paper_id,
                    rule_id=result["rule_id"],
                    device_id=device_id,
                    matched=result.get("matched", False),
                    confidence=result.get("confidence", 0.0),
                    regions=json.dumps(result.get("regions", []))
                )
            
            # Mark paper as completed
            self._complete_paper(paper_id, device_id)
            
            # Award points
            points = rules_matched * 10 + len(results) * 2
            self._award_points(device_id, points)
            
            logger.info(f"Device {device_id} validated {paper_id}: {rules_matched}/{len(results)} rules matched")
            
            return {
                "accepted": True,
                "paper_id": paper_id,
                "rules_matched": rules_matched,
                "rules_checked": len(results),
                "points_earned": points
            }
            
        except Exception as e:
            logger.error(f"Error submitting results: {e}")
            return {
                "accepted": False,
                "error": str(e)
            }
    
    def _get_next_paper(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next paper for validation - ROUND ROBIN PRINCIPLE.
        
        Multiple validations of the same paper by different devices ARE ALLOWED
        and actually WANTED for consensus building!
        
        Priority:
        1. Papers this device hasn't validated yet (fairness)
        2. Papers with fewest total validations (round robin)
        3. Papers by priority/age
        
        No blocking: A paper can be assigned to multiple devices simultaneously.
        All validations are documented.
        """
        with self.db.get_connection() as conn:
            # First try: Papers this device hasn't seen yet
            # Prefer papers with fewer total validations (round-robin)
            row = conn.execute(
                """
                SELECT p.*, 
                       COALESCE(v.validation_count, 0) as validations
                FROM papers p
                LEFT JOIN (
                    SELECT paper_id, COUNT(*) as validation_count 
                    FROM paper_validations 
                    GROUP BY paper_id
                ) v ON p.paper_id = v.paper_id
                WHERE p.status = 'ready'
                  AND p.paper_id NOT IN (
                      SELECT paper_id FROM paper_validations 
                      WHERE device_id = ?
                  )
                ORDER BY 
                    COALESCE(v.validation_count, 0) ASC,  -- Least validated first (round-robin)
                    p.priority DESC,
                    p.created_at ASC
                LIMIT 1
                """,
                (device_id,)
            ).fetchone()
            
            if row:
                return dict(row)
            
            # Fallback: If device has seen ALL papers, cycle back to least-validated
            # This allows re-validation for consensus (multiple opinions wanted!)
            row = conn.execute(
                """
                SELECT p.*, 
                       COALESCE(v.validation_count, 0) as validations,
                       COALESCE(dv.device_validations, 0) as my_validations
                FROM papers p
                LEFT JOIN (
                    SELECT paper_id, COUNT(*) as validation_count 
                    FROM paper_validations 
                    GROUP BY paper_id
                ) v ON p.paper_id = v.paper_id
                LEFT JOIN (
                    SELECT paper_id, COUNT(*) as device_validations
                    FROM paper_validations
                    WHERE device_id = ?
                    GROUP BY paper_id
                ) dv ON p.paper_id = dv.paper_id
                WHERE p.status = 'ready'
                ORDER BY 
                    COALESCE(dv.device_validations, 0) ASC,  -- Papers I've validated least
                    COALESCE(v.validation_count, 0) ASC,      -- Overall least validated
                    p.priority DESC
                LIMIT 1
                """,
                (device_id,)
            ).fetchone()
            
            if row:
                logger.info(f"Round-robin cycle: {row['paper_id']} (validations: {row['validations']})")
                return dict(row)
                
        return None
    
    def _assign_paper(self, paper_id: str, device_id: str):
        """
        Mark paper as assigned to device.
        
        NOTE: This is now just for tracking, NOT for blocking!
        Multiple devices can work on the same paper simultaneously.
        """
        expires_at = datetime.now() + timedelta(seconds=self.job_ttl_seconds)
        with self.db.get_connection() as conn:
            # Use INSERT OR IGNORE - don't block if already assigned
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_assignments 
                (paper_id, device_id, assigned_at, expires_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (paper_id, device_id, expires_at)
            )
    
    def _verify_assignment(self, paper_id: str, device_id: str) -> bool:
        """
        Verify device can submit results for this paper.
        
        RELAXED: We now accept results even if assignment expired,
        as long as the paper exists and is 'ready'.
        Multiple validations are WANTED for consensus!
        """
        with self.db.get_connection() as conn:
            # Check if paper exists and is ready
            row = conn.execute(
                """
                SELECT 1 FROM papers
                WHERE paper_id = ? AND status = 'ready'
                """,
                (paper_id,)
            ).fetchone()
            return row is not None
    
    def _complete_paper(self, paper_id: str, device_id: str):
        """Mark paper validation as complete - records the validation"""
        with self.db.get_connection() as conn:
            # Remove assignment for this device (if exists)
            conn.execute(
                "DELETE FROM paper_assignments WHERE paper_id = ? AND device_id = ?",
                (paper_id, device_id)
            )
            # Record validation (allows multiple per device for re-validation!)
            conn.execute(
                """
                INSERT INTO paper_validations (paper_id, device_id, validated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (paper_id, device_id)
            )
    
    def _store_result(
        self,
        paper_id: str,
        rule_id: str,
        device_id: str,
        matched: bool,
        confidence: float,
        regions: str
    ):
        """Store rule match result"""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_rule_results 
                (paper_id, rule_id, device_id, matched, confidence, regions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (paper_id, rule_id, device_id, matched, confidence, regions)
            )
    
    def _award_points(self, device_id: str, points: int):
        """Award points to device"""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE device_registry 
                SET total_points = COALESCE(total_points, 0) + ?,
                    jobs_completed = COALESCE(jobs_completed, 0) + 1
                WHERE device_id = ?
                """,
                (points, device_id)
            )
    
    def _get_paper_embedding_and_text(self, paper_id: str) -> tuple:
        """
        Get paper's BioBERT embedding as base64 AND the section text.
        
        Returns both because Android needs:
        1. Embedding for semantic similarity (cosine)
        2. Text for keyword matching (hybrid approach)
        
        Returns:
            (embedding_b64, text) or (None, None) if not found
        """
        try:
            with self.db.get_connection() as conn:
                # Get embedding AND text from paper_sections (prefer 'abstract')
                row = conn.execute(
                    """
                    SELECT embedding, section_text FROM paper_sections 
                    WHERE paper_id = ? AND embedding IS NOT NULL
                    ORDER BY 
                        CASE section_name 
                            WHEN 'abstract' THEN 1 
                            WHEN 'introduction' THEN 2 
                            ELSE 3 
                        END
                    LIMIT 1
                    """,
                    (paper_id,)
                ).fetchone()
                
                if row and row["embedding"]:
                    # embedding is stored as bytes, convert to base64
                    embedding_bytes = row["embedding"]
                    if isinstance(embedding_bytes, bytes):
                        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                    else:
                        embedding_b64 = embedding_bytes
                    
                    # Limit text to first 2000 chars to keep API response size reasonable
                    section_text = row["section_text"] or ""
                    section_text = section_text[:2000]
                    
                    return embedding_b64, section_text
                    
        except Exception as e:
            logger.error(f"Failed to get embedding for {paper_id}: {e}")
        
        return None, None
    
    def _get_paper_embedding(self, paper_id: str) -> Optional[str]:
        """
        Get paper's BioBERT embedding as base64.
        (Legacy method - use _get_paper_embedding_and_text instead)
        """
        embedding_b64, _ = self._get_paper_embedding_and_text(paper_id)
        return embedding_b64
    
    def _get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get ALL section embeddings as chunks for Unity molecule visualization.
        
        UNIFIED STRUCTURE (same as Rules):
        - chunk_id: Sequential index
        - chunk_type: Section name (abstract, methods, etc.)
        - text_preview: First 500 chars
        - embedding_b64: 768-dim BioBERT embedding
        - voxels: 8x8x12 grid for Unity rendering (VISIBLE cubes)
        - color: RGB for visualization
        - position: 3D position (Chunk = INVISIBLE container at this position)
        - connects_to: Links to next chunk (wires/lanes)
        
        Unity renders:
        1. Chunk = invisible cube at position (0,0,0 origin for voxels)
        2. Voxels = visible cubes INSIDE chunk
        3. Wires = connections between chunks
        """
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT section_name, section_text, embedding, 
                           color_r, color_g, color_b, page_number, voxel_data
                    FROM paper_sections 
                    WHERE paper_id = ? AND embedding IS NOT NULL
                    ORDER BY page_number, 
                        CASE section_name 
                            WHEN 'abstract' THEN 1 
                            WHEN 'introduction' THEN 2 
                            WHEN 'methods' THEN 3
                            WHEN 'results' THEN 4
                            WHEN 'discussion' THEN 5
                            WHEN 'conclusion' THEN 6
                            WHEN 'references' THEN 7
                            ELSE 8 
                        END
                    """,
                    (paper_id,)
                ).fetchall()
                
                if not rows:
                    return []
                
                chunks = []
                for idx, row in enumerate(rows):
                    section_name = row["section_name"]
                    
                    # Convert embedding to base64 AND create voxels
                    embedding_bytes = row["embedding"]
                    if isinstance(embedding_bytes, bytes):
                        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                        # Create voxel grid from embedding
                        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                        voxel_grid = VoxelGrid.from_embedding(embedding_array, threshold=0.1)
                    else:
                        embedding_b64 = embedding_bytes
                        voxel_grid = VoxelGrid(voxels=[])
                    
                    # Get color from DB or default
                    color_r = row["color_r"]
                    color_g = row["color_g"] 
                    color_b = row["color_b"]
                    if color_r is None:
                        default_color = SECTION_COLORS.get(section_name, SECTION_COLORS["other"])
                        color_r, color_g, color_b = default_color
                    
                    # Build chunk with unified structure
                    chunk = {
                        "chunk_id": idx,
                        "chunk_type": section_name,  # Unified: was "section_name"
                        "text_preview": (row["section_text"] or "")[:500],
                        "embedding_b64": embedding_b64,
                        # NEW: Include voxels for Unity visualization
                        "voxels": voxel_grid.to_dict(),
                        "color": {
                            "r": color_r or 0.5,
                            "g": color_g or 0.5,
                            "b": color_b or 0.5
                        },
                        # Position: where to place the INVISIBLE chunk container
                        "position": {
                            "x": idx * 2.0,  # Spread along X axis
                            "y": 0.0,
                            "z": 0.0
                        },
                        # Connections: wires to next chunk
                        "connects_to": [idx + 1] if idx < len(rows) - 1 else []
                    }
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Failed to get chunks for {paper_id}: {e}")
            return []


# Singleton
_handler: Optional[JobHandler] = None


def get_job_handler() -> JobHandler:
    """Get singleton job handler"""
    global _handler
    if _handler is None:
        _handler = JobHandler()
    return _handler
