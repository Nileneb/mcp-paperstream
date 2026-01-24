"""
Consensus Engine

Aggregates validation results from multiple Android devices
and determines consensus for paper-rule combinations.

Features:
- Majority voting
- Agreement ratio calculation
- Minimum vote threshold (default: 3)
- Weighted voting based on device accuracy
- SSE emission on paper validation complete
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from ..db import get_db, PaperConsensus, ValidationResult
from ..api.sse_stream import get_unity_sse_stream
from ..processing import pdf_to_voxel_grid

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """
    Calculates consensus from distributed validation results.
    """
    
    def __init__(self):
        self.db = get_db()
        self.min_votes_required = 3
        self.high_agreement_threshold = 0.8
    
    def calculate_consensus(
        self,
        paper_id: str,
        rule_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate consensus for a paper-rule combination.
        
        Args:
            paper_id: Paper identifier
            rule_id: Rule identifier
        
        Returns:
            Consensus result or None if insufficient votes
        """
        results = self.db.get_results_for_paper_rule(paper_id, rule_id)
        
        if not results:
            return None
        
        vote_count = len(results)
        
        # Count matches and non-matches
        match_votes = [r for r in results if r.is_match]
        non_match_votes = [r for r in results if not r.is_match]
        
        match_count = len(match_votes)
        non_match_count = len(non_match_votes)
        
        # Calculate averages
        avg_similarity = sum(r.similarity for r in results) / vote_count
        avg_confidence = sum(r.confidence for r in results) / vote_count
        
        # Agreement ratio (how much devices agree with each other)
        majority_count = max(match_count, non_match_count)
        agreement_ratio = majority_count / vote_count
        
        # Determine consensus (majority vote)
        is_match = match_count > non_match_count
        
        # Collect sections where matches were found
        matched_sections = list(set(r.section_id for r in match_votes))
        
        # Check if validated (enough votes)
        is_validated = vote_count >= self.min_votes_required
        
        # Create consensus record
        consensus = self.db.update_consensus(paper_id, rule_id)
        
        result = {
            "paper_id": paper_id,
            "rule_id": rule_id,
            "is_match": is_match,
            "vote_count": vote_count,
            "match_votes": match_count,
            "non_match_votes": non_match_count,
            "avg_similarity": round(avg_similarity, 4),
            "avg_confidence": round(avg_confidence, 4),
            "agreement_ratio": round(agreement_ratio, 4),
            "is_validated": is_validated,
            "matched_sections": matched_sections,
            "consensus_strength": self._calculate_strength(agreement_ratio, avg_confidence)
        }
        
        logger.info(
            f"Consensus for {paper_id}/{rule_id}: "
            f"match={is_match}, votes={vote_count}, agreement={agreement_ratio:.2f}"
        )
        
        # Auto-check if paper is fully validated and emit SSE
        if is_validated:
            self.check_and_finalize(paper_id)
        
        return result
    
    def _calculate_strength(
        self,
        agreement_ratio: float,
        avg_confidence: float
    ) -> str:
        """
        Calculate consensus strength label.
        
        Returns:
            "strong", "moderate", or "weak"
        """
        score = (agreement_ratio + avg_confidence) / 2
        
        if score >= 0.8:
            return "strong"
        elif score >= 0.6:
            return "moderate"
        else:
            return "weak"
    
    def get_paper_validation_status(self, paper_id: str) -> Dict[str, Any]:
        """
        Get complete validation status for a paper across all rules.
        
        Args:
            paper_id: Paper identifier
        
        Returns:
            {
                "paper_id": str,
                "rules_checked": int,
                "rules_validated": int,
                "overall_status": "validated" | "pending" | "incomplete",
                "rules": [...]
            }
        """
        # Get all active rules
        rules = self.db.get_active_rules()
        
        rule_statuses = []
        rules_validated = 0
        
        for rule in rules:
            consensus = self.db.get_consensus(paper_id, rule.rule_id)
            
            if consensus:
                status = {
                    "rule_id": rule.rule_id,
                    "question": rule.question,
                    "is_match": consensus.is_match,
                    "is_validated": consensus.is_validated,
                    "vote_count": consensus.vote_count,
                    "agreement_ratio": consensus.agreement_ratio
                }
                
                if consensus.is_validated:
                    rules_validated += 1
            else:
                status = {
                    "rule_id": rule.rule_id,
                    "question": rule.question,
                    "is_match": None,
                    "is_validated": False,
                    "vote_count": 0,
                    "agreement_ratio": None
                }
            
            rule_statuses.append(status)
        
        # Determine overall status
        if rules_validated == len(rules):
            overall_status = "validated"
        elif rules_validated > 0:
            overall_status = "incomplete"
        else:
            overall_status = "pending"
        
        return {
            "paper_id": paper_id,
            "rules_checked": len(rules),
            "rules_validated": rules_validated,
            "overall_status": overall_status,
            "rules": rule_statuses
        }
    
    def get_pending_validations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get paper-rule combinations that need more votes.
        
        Returns:
            List of pending validations
        """
        with self.db.get_connection() as conn:
            # Find paper-rule combinations with insufficient votes
            rows = conn.execute(
                """
                SELECT 
                    p.paper_id,
                    p.title,
                    r.rule_id,
                    r.question,
                    COALESCE(c.vote_count, 0) as vote_count,
                    c.is_validated
                FROM papers p
                CROSS JOIN rules r
                LEFT JOIN paper_consensus c ON p.paper_id = c.paper_id AND r.rule_id = c.rule_id
                WHERE p.status = 'ready'
                  AND r.is_active = 1
                  AND (c.is_validated IS NULL OR c.is_validated = 0)
                ORDER BY COALESCE(c.vote_count, 0) DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            
            return [
                {
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                    "rule_id": r["rule_id"],
                    "question": r["question"],
                    "vote_count": r["vote_count"],
                    "votes_needed": self.min_votes_required - r["vote_count"]
                }
                for r in rows
            ]
    
    def recalculate_all_consensus(self) -> Dict[str, Any]:
        """
        Recalculate consensus for all paper-rule combinations.
        
        Returns:
            Summary of recalculation
        """
        with self.db.get_connection() as conn:
            # Get all unique paper-rule combinations with results
            rows = conn.execute(
                """
                SELECT DISTINCT paper_id, rule_id 
                FROM validation_results
                """
            ).fetchall()
        
        recalculated = 0
        for row in rows:
            self.calculate_consensus(row["paper_id"], row["rule_id"])
            recalculated += 1
        
        return {
            "recalculated": recalculated,
            "message": f"Recalculated consensus for {recalculated} paper-rule combinations"
        }
    
    async def finalize_paper_consensus(self, paper_id: str) -> Dict[str, Any]:
        """
        Finalize consensus for a paper and emit SSE event to Unity.
        
        Called when all rules have been validated for a paper.
        Generates voxel data and sends paper_validated event.
        
        Args:
            paper_id: Paper identifier
        
        Returns:
            Finalization result with voxel data
        """
        # Get paper details
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"status": "error", "message": "Paper not found"}
        
        # Get validation status
        status = self.get_paper_validation_status(paper_id)
        
        # Build rules results dict
        rules_results = {}
        for rule_status in status.get("rules", []):
            rules_results[rule_status["rule_id"]] = rule_status.get("is_match", False)
        
        # Generate voxel data if PDF available
        voxel_data = None
        thumbnail_base64 = None
        
        if paper.pdf_local_path:
            from pathlib import Path
            pdf_path = Path(paper.pdf_local_path)
            
            if pdf_path.exists():
                # Generate voxels
                voxel_data = pdf_to_voxel_grid(pdf_path)
                logger.info(f"Generated voxel data for {paper_id}: {voxel_data['stats']['total']} voxels")
                
                # Generate thumbnail
                try:
                    from ..api.paper_handler import get_paper_handler
                    handler = get_paper_handler()
                    thumb_result = handler.get_thumbnail(paper_id, as_base64=True)
                    if thumb_result.get("status") == "success":
                        thumbnail_base64 = thumb_result.get("thumbnail_base64")
                except Exception as e:
                    logger.warning(f"Could not generate thumbnail: {e}")
        
        # Emit SSE event to Unity
        sse = get_unity_sse_stream()
        await sse.emit_paper_validated(
            paper_id=paper_id,
            title=paper.title or paper_id,
            rules_results=rules_results,
            voxel_data=voxel_data,
            thumbnail_base64=thumbnail_base64
        )
        
        logger.info(f"Emitted paper_validated SSE for {paper_id}")
        
        return {
            "status": "finalized",
            "paper_id": paper_id,
            "title": paper.title,
            "rules_validated": status.get("rules_validated", 0),
            "overall_status": status.get("overall_status"),
            "has_voxel_data": voxel_data is not None,
            "has_thumbnail": thumbnail_base64 is not None,
            "voxel_stats": voxel_data.get("stats") if voxel_data else None
        }
    
    def check_and_finalize(self, paper_id: str) -> bool:
        """
        Check if paper is fully validated and trigger finalization.
        
        Returns:
            True if finalization was triggered
        """
        status = self.get_paper_validation_status(paper_id)
        
        if status.get("overall_status") == "validated":
            # All rules validated - trigger async finalization
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.finalize_paper_consensus(paper_id))
                else:
                    loop.run_until_complete(self.finalize_paper_consensus(paper_id))
                return True
            except Exception as e:
                logger.error(f"Failed to finalize consensus for {paper_id}: {e}")
        
        return False


# Singleton instance
_consensus_engine: Optional[ConsensusEngine] = None


def get_consensus_engine() -> ConsensusEngine:
    """Get singleton consensus engine"""
    global _consensus_engine
    if _consensus_engine is None:
        _consensus_engine = ConsensusEngine()
    return _consensus_engine
