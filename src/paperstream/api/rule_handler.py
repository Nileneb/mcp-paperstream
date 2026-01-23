"""
Rule Handler - API endpoints for validation rules

Handles:
- POST /api/rules/create - Create new rule with BioBERT embeddings
- GET /api/rules - List all active rules
- GET /api/rules/{rule_id} - Get rule details
"""

import json
import logging
from typing import Optional, Dict, Any, List
import numpy as np

from ..db import get_db, Rule
from ..handlers import get_biobert_handler

logger = logging.getLogger(__name__)


# Default rules for paper validation
DEFAULT_RULES = [
    {
        "rule_id": "is_rct",
        "question": "Is this a Randomized Controlled Trial (RCT)?",
        "positive_phrases": [
            "randomized controlled trial",
            "RCT",
            "randomized clinical trial",
            "randomized study",
            "randomly assigned",
            "random allocation"
        ],
        "negative_phrases": [
            "observational study",
            "retrospective",
            "case report",
            "review article",
            "meta-analysis"
        ],
        "threshold": 0.75
    },
    {
        "rule_id": "has_placebo",
        "question": "Does the study use a placebo control?",
        "positive_phrases": [
            "placebo-controlled",
            "placebo group",
            "placebo arm",
            "sham treatment",
            "inactive control"
        ],
        "negative_phrases": [
            "no placebo",
            "active comparator only",
            "open-label"
        ],
        "threshold": 0.70
    },
    {
        "rule_id": "is_blinded",
        "question": "Is the study blinded?",
        "positive_phrases": [
            "double-blind",
            "single-blind",
            "triple-blind",
            "blinded assessment",
            "masked study"
        ],
        "negative_phrases": [
            "open-label",
            "unblinded",
            "no blinding"
        ],
        "threshold": 0.70
    },
    {
        "rule_id": "reports_primary_outcome",
        "question": "Does the study report a primary outcome?",
        "positive_phrases": [
            "primary outcome",
            "primary endpoint",
            "primary efficacy",
            "main outcome measure"
        ],
        "negative_phrases": [
            "no primary outcome defined",
            "exploratory analysis only"
        ],
        "threshold": 0.65
    },
    {
        "rule_id": "sample_size_adequate",
        "question": "Does the study have adequate sample size (N > 50)?",
        "positive_phrases": [
            "patients enrolled",
            "participants included",
            "sample size calculation",
            "power analysis"
        ],
        "negative_phrases": [
            "pilot study",
            "small sample",
            "preliminary results"
        ],
        "threshold": 0.60
    },
    {
        "rule_id": "has_statistical_analysis",
        "question": "Does the study include proper statistical analysis?",
        "positive_phrases": [
            "statistical analysis",
            "p-value",
            "confidence interval",
            "intention-to-treat",
            "per-protocol analysis"
        ],
        "negative_phrases": [
            "descriptive only",
            "no statistical tests"
        ],
        "threshold": 0.65
    }
]


class RuleHandler:
    """Handler for validation rule operations"""
    
    def __init__(self):
        self.db = get_db()
        self._biobert = None
    
    @property
    def biobert(self):
        """Lazy load BioBERT handler"""
        if self._biobert is None:
            self._biobert = get_biobert_handler()
        return self._biobert
    
    def create_rule(
        self,
        rule_id: str,
        question: str,
        positive_phrases: List[str],
        negative_phrases: Optional[List[str]] = None,
        threshold: float = 0.75,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Create a new validation rule with BioBERT embeddings.
        
        Called by n8n webhook: POST /api/rules/create
        
        Args:
            rule_id: Unique identifier for the rule
            question: The validation question
            positive_phrases: Phrases indicating positive match
            negative_phrases: Phrases indicating negative match
            threshold: Similarity threshold for match (0-1)
            created_by: Creator identifier
        
        Returns:
            {
                "status": "created" | "exists" | "error",
                "rule_id": str,
                "message": str
            }
        """
        # Check if rule exists
        existing = self.db.get_rule(rule_id)
        if existing:
            return {
                "status": "exists",
                "rule_id": rule_id,
                "message": "Rule already exists"
            }
        
        try:
            # Generate embeddings for positive phrases
            pos_embedding = self._compute_phrase_embedding(positive_phrases)
            
            # Generate embeddings for negative phrases
            neg_embedding = None
            if negative_phrases:
                neg_embedding = self._compute_phrase_embedding(negative_phrases)
            
            # Create rule
            rule = Rule(
                rule_id=rule_id,
                question=question,
                positive_phrases=json.dumps(positive_phrases),
                negative_phrases=json.dumps(negative_phrases) if negative_phrases else None,
                pos_embedding=pos_embedding,
                neg_embedding=neg_embedding,
                threshold=threshold,
                is_active=True,
                created_by=created_by
            )
            
            rule = self.db.create_rule(rule)
            logger.info(f"Rule created: {rule_id}")
            
            return {
                "status": "created",
                "rule_id": rule_id,
                "message": f"Rule created with {len(positive_phrases)} positive phrases",
                "embedding_dim": 768
            }
            
        except Exception as e:
            logger.error(f"Failed to create rule {rule_id}: {e}")
            return {
                "status": "error",
                "rule_id": rule_id,
                "message": str(e)
            }
    
    def _compute_phrase_embedding(self, phrases: List[str]) -> bytes:
        """
        Compute averaged embedding for a list of phrases.
        
        Args:
            phrases: List of phrases to embed
        
        Returns:
            bytes: Averaged embedding as float32 bytes
        """
        embeddings = []
        for phrase in phrases:
            emb = self.biobert.embed(phrase)
            embeddings.append(emb)
        
        # Average all phrase embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return np.array(avg_embedding, dtype=np.float32).tobytes()
    
    def get_rule(self, rule_id: str, include_embeddings: bool = False) -> Optional[Dict[str, Any]]:
        """Get rule details"""
        rule = self.db.get_rule(rule_id)
        if rule:
            return rule.to_dict(include_embeddings=include_embeddings)
        return None
    
    def list_rules(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """List all active rules"""
        rules = self.db.get_active_rules()
        return [r.to_dict(include_embeddings=include_embeddings) for r in rules]
    
    def update_rule(
        self,
        rule_id: str,
        question: Optional[str] = None,
        positive_phrases: Optional[List[str]] = None,
        negative_phrases: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        is_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Update an existing rule"""
        rule = self.db.get_rule(rule_id)
        if not rule:
            return {
                "status": "not_found",
                "rule_id": rule_id,
                "message": "Rule not found"
            }
        
        try:
            if question is not None:
                rule.question = question
            
            if positive_phrases is not None:
                rule.positive_phrases = json.dumps(positive_phrases)
                rule.pos_embedding = self._compute_phrase_embedding(positive_phrases)
            
            if negative_phrases is not None:
                rule.negative_phrases = json.dumps(negative_phrases)
                rule.neg_embedding = self._compute_phrase_embedding(negative_phrases)
            
            if threshold is not None:
                rule.threshold = threshold
            
            if is_active is not None:
                rule.is_active = is_active
            
            self.db.update_rule(rule)
            
            return {
                "status": "updated",
                "rule_id": rule_id,
                "message": "Rule updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update rule {rule_id}: {e}")
            return {
                "status": "error",
                "rule_id": rule_id,
                "message": str(e)
            }
    
    def load_default_rules(self) -> Dict[str, Any]:
        """
        Load default validation rules into database.
        
        Called on server startup to ensure base rules exist.
        """
        created = 0
        skipped = 0
        errors = []
        
        for rule_def in DEFAULT_RULES:
            result = self.create_rule(
                rule_id=rule_def["rule_id"],
                question=rule_def["question"],
                positive_phrases=rule_def["positive_phrases"],
                negative_phrases=rule_def.get("negative_phrases"),
                threshold=rule_def.get("threshold", 0.75),
                created_by="system_default"
            )
            
            if result["status"] == "created":
                created += 1
            elif result["status"] == "exists":
                skipped += 1
            else:
                errors.append(result)
        
        logger.info(f"Default rules: {created} created, {skipped} skipped")
        
        return {
            "created": created,
            "skipped": skipped,
            "total": len(DEFAULT_RULES),
            "errors": errors
        }
    
    def match_section_against_rule(
        self,
        section_embedding: np.ndarray,
        rule: Rule
    ) -> Dict[str, Any]:
        """
        Check if a section embedding matches a rule.
        
        Args:
            section_embedding: 768-dim embedding of section text
            rule: Rule to match against
        
        Returns:
            {
                "is_match": bool,
                "similarity": float,
                "confidence": float
            }
        """
        pos_emb = rule.get_pos_embedding_array()
        neg_emb = rule.get_neg_embedding_array()
        
        if pos_emb is None:
            return {"is_match": False, "similarity": 0.0, "confidence": 0.0}
        
        # Normalize section embedding
        section_norm = section_embedding / (np.linalg.norm(section_embedding) + 1e-9)
        
        # Cosine similarity with positive embedding
        pos_similarity = float(np.dot(section_norm, pos_emb))
        
        # If negative embedding exists, also compute that
        neg_similarity = 0.0
        if neg_emb is not None:
            neg_similarity = float(np.dot(section_norm, neg_emb))
        
        # Score: positive similarity minus negative similarity
        score = pos_similarity - (neg_similarity * 0.5)  # Weight negative less
        
        # Match if score exceeds threshold
        is_match = score >= rule.threshold
        
        # Confidence based on how far above/below threshold
        confidence = min(1.0, max(0.0, (score - rule.threshold + 0.5)))
        
        return {
            "is_match": is_match,
            "similarity": round(pos_similarity, 4),
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "threshold": rule.threshold
        }


# Singleton instance
_rule_handler: Optional[RuleHandler] = None


def get_rule_handler() -> RuleHandler:
    """Get singleton rule handler"""
    global _rule_handler
    if _rule_handler is None:
        _rule_handler = RuleHandler()
    return _rule_handler
