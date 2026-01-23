"""
Job Handler - API endpoints for validation job distribution

Handles:
- GET /api/jobs/next - Get next jobs for Android device
- POST /api/validation/submit - Submit validation results
- GET /api/jobs/{job_id} - Get job status
"""

import json
import logging
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid

from ..db import get_db, ValidationJob, ValidationResult, DeviceRegistry

logger = logging.getLogger(__name__)


class JobHandler:
    """Handler for validation job distribution"""
    
    def __init__(self):
        self.db = get_db()
        self.job_ttl_seconds = 300  # 5 minutes to complete a job
    
    def get_next_jobs(
        self,
        device_id: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get next available jobs for an Android device.
        
        Called by Android app: GET /api/jobs/next?device_id=xxx&limit=5
        
        Args:
            device_id: Unique device identifier
            limit: Maximum jobs to return (max 10)
        
        Returns:
            {
                "jobs": [...],
                "assigned_count": int,
                "device_id": str
            }
        """
        limit = min(limit, 10)  # Cap at 10
        
        # Update device last_seen
        self.db.update_device_seen(device_id)
        
        # First, clean up expired jobs
        reassigned = self.db.reassign_expired_jobs()
        if reassigned > 0:
            logger.info(f"Reassigned {reassigned} expired jobs")
        
        # Get available jobs with full data
        jobs_data = self.db.get_jobs_for_device(device_id, limit)
        
        # Assign jobs to this device
        assigned_jobs = []
        for job_data in jobs_data:
            if self.db.assign_job(job_data["job_id"], device_id, self.job_ttl_seconds):
                # Convert embeddings to base64 for transport
                job_response = {
                    "job_id": job_data["job_id"],
                    "paper_id": job_data["paper_id"],
                    "section_id": job_data["section_id"],
                    "rule_id": job_data["rule_id"],
                    "question": job_data["question"],
                    "threshold": job_data["threshold"],
                    "section_text": job_data["section_text"][:500] if job_data["section_text"] else None,
                }
                
                # Add voxel data if available
                if job_data.get("voxel_data"):
                    job_response["voxel_data"] = job_data["voxel_data"]
                
                # Encode embeddings as base64
                if job_data.get("section_embedding"):
                    job_response["section_embedding_b64"] = base64.b64encode(
                        job_data["section_embedding"]
                    ).decode("ascii")
                
                if job_data.get("pos_embedding"):
                    job_response["pos_embedding_b64"] = base64.b64encode(
                        job_data["pos_embedding"]
                    ).decode("ascii")
                
                if job_data.get("neg_embedding"):
                    job_response["neg_embedding_b64"] = base64.b64encode(
                        job_data["neg_embedding"]
                    ).decode("ascii")
                
                assigned_jobs.append(job_response)
        
        logger.info(f"Assigned {len(assigned_jobs)} jobs to {device_id}")
        
        return {
            "jobs": assigned_jobs,
            "assigned_count": len(assigned_jobs),
            "device_id": device_id,
            "expires_in_seconds": self.job_ttl_seconds
        }
    
    def submit_results(
        self,
        device_id: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Submit validation results from Android device.
        
        Called by Android app: POST /api/validation/submit
        
        Args:
            device_id: Device identifier
            results: List of results [
                {
                    "job_id": str,
                    "is_match": bool,
                    "similarity": float,
                    "confidence": float,
                    "time_taken_ms": int (optional)
                }
            ]
        
        Returns:
            {
                "accepted": int,
                "rejected": int,
                "total_points": int,
                "errors": [...]
            }
        """
        accepted = 0
        rejected = 0
        total_points = 0
        errors = []
        
        for result in results:
            try:
                job_id = result["job_id"]
                is_match = result["is_match"]
                similarity = result["similarity"]
                confidence = result.get("confidence", similarity)
                time_taken_ms = result.get("time_taken_ms")
                
                # Verify job exists and is assigned to this device
                with self.db.get_connection() as conn:
                    job_row = conn.execute(
                        """
                        SELECT * FROM validation_jobs 
                        WHERE job_id = ? AND assigned_to = ? AND status = 'assigned'
                        """,
                        (job_id, device_id)
                    ).fetchone()
                    
                    if not job_row:
                        errors.append({
                            "job_id": job_id,
                            "error": "Job not found or not assigned to this device"
                        })
                        rejected += 1
                        continue
                
                # Calculate points (based on confidence and match)
                points = self._calculate_points(similarity, confidence, is_match)
                
                # Create result record
                validation_result = ValidationResult(
                    job_id=job_id,
                    device_id=device_id,
                    paper_id=job_row["paper_id"],
                    rule_id=job_row["rule_id"],
                    section_id=job_row["section_id"],
                    is_match=is_match,
                    similarity=similarity,
                    confidence=confidence,
                    points_earned=points,
                    time_taken_ms=time_taken_ms
                )
                
                self.db.submit_result(validation_result)
                
                # Mark job as completed
                self.db.complete_job(job_id, is_match, similarity, confidence)
                
                # Update consensus
                self.db.update_consensus(job_row["paper_id"], job_row["rule_id"])
                
                # Update leaderboard
                self.db.update_leaderboard(device_id, points, is_match)
                
                accepted += 1
                total_points += points
                
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                errors.append({
                    "job_id": result.get("job_id", "unknown"),
                    "error": str(e)
                })
                rejected += 1
        
        logger.info(f"Device {device_id}: {accepted} results accepted, {total_points} points")
        
        return {
            "accepted": accepted,
            "rejected": rejected,
            "total_points": total_points,
            "errors": errors if errors else None
        }
    
    def _calculate_points(
        self,
        similarity: float,
        confidence: float,
        is_match: bool
    ) -> int:
        """
        Calculate points for a validation result.
        
        Scoring:
        - Base: 10 points
        - Similarity bonus: up to 40 points
        - Confidence bonus: up to 30 points
        - Match bonus: 20 points if match found
        """
        base = 10
        similarity_bonus = int(similarity * 40)
        confidence_bonus = int(confidence * 30)
        match_bonus = 20 if is_match else 0
        
        return base + similarity_bonus + confidence_bonus + match_bonus
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM validation_jobs WHERE job_id = ?",
                (job_id,)
            ).fetchone()
            
            if row:
                return {
                    "job_id": row["job_id"],
                    "paper_id": row["paper_id"],
                    "rule_id": row["rule_id"],
                    "status": row["status"],
                    "assigned_to": row["assigned_to"],
                    "is_match": row["is_match"],
                    "similarity": row["similarity"]
                }
        return None
    
    def create_jobs_for_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Create validation jobs for all sections × rules of a paper.
        
        Called after paper processing is complete.
        
        Returns:
            {
                "jobs_created": int,
                "paper_id": str
            }
        """
        # Get all sections for paper
        sections = self.db.get_sections_for_paper(paper_id)
        if not sections:
            return {
                "jobs_created": 0,
                "paper_id": paper_id,
                "error": "No sections found"
            }
        
        # Get all active rules
        rules = self.db.get_active_rules()
        if not rules:
            return {
                "jobs_created": 0,
                "paper_id": paper_id,
                "error": "No active rules"
            }
        
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
        
        logger.info(f"Created {jobs_created} jobs for paper {paper_id}")
        
        return {
            "jobs_created": jobs_created,
            "paper_id": paper_id,
            "sections": len(sections),
            "rules": len(rules)
        }


# Singleton instance
_job_handler: Optional[JobHandler] = None


def get_job_handler() -> JobHandler:
    """Get singleton job handler"""
    global _job_handler
    if _job_handler is None:
        _job_handler = JobHandler()
    return _job_handler
