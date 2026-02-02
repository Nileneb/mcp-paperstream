"""
Rule Handler - API endpoints for validation rules

DATA MODEL (same as Papers - see core/data_model.py):
- Text (phrases) → Embedding (768-dim BioBERT)
- Embedding → Chunk (container with voxels)
- Chunk → Voxels (8x8x12 grid, visible in Unity)

Rules have 2 chunks:
- Chunk 0: POSITIVE embedding (green) - what we're looking for
- Chunk 1: NEGATIVE embedding (red) - what indicates absence

Handles:
- POST /api/rules/create - Create new rule with BioBERT embeddings
- GET /api/rules - List all active rules
- GET /api/rules/{rule_id} - Get rule details
- GET /api/rules/{rule_id}/chunks - Get rule as molecule chunks with voxels
"""

import base64
import json
import logging
from typing import Optional, Dict, Any, List
import numpy as np

from ..db import get_db, Rule
from ..handlers import get_biobert_handler
from ..rules import ENHANCED_RULES
from ..core import VoxelGrid, SECTION_COLORS

logger = logging.getLogger(__name__)


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
            
            # Store in Qdrant for fast vector search
            try:
                from ..db.vector_store import upsert_rule_embeddings
                pos_list = list(np.frombuffer(pos_embedding, dtype=np.float32))
                neg_list = list(np.frombuffer(neg_embedding, dtype=np.float32)) if neg_embedding else [0.0] * 768
                upsert_rule_embeddings(
                    rule_id=rule_id,
                    pos_embedding=pos_list,
                    neg_embedding=neg_list,
                    payload={"question": question, "threshold": threshold}
                )
                logger.info(f"Rule {rule_id} stored in Qdrant")
            except Exception as e:
                logger.warning(f"Failed to store rule in Qdrant: {e}")
            
            return {
                "status": "created",
                "rule_id": rule_id,
                "message": f"Rule created with {len(positive_phrases)} positive phrases",
                "embedding_dim": 768,
                "qdrant_synced": True
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
        Load enhanced validation rules into database.
        
        Uses ENHANCED_RULES with semantically rich phrases for BioBERT matching.
        Called on server startup to ensure base rules exist.
        """
        created = 0
        skipped = 0
        errors = []
        
        for rule_id, rule_def in ENHANCED_RULES.items():
            result = self.create_rule(
                rule_id=rule_id,
                question=rule_def["question"],
                positive_phrases=rule_def["positive_phrases"],
                negative_phrases=rule_def.get("negative_phrases"),
                threshold=rule_def.get("threshold", 0.65),
                created_by="system_enhanced"
            )
            
            if result["status"] == "created":
                created += 1
            elif result["status"] == "exists":
                skipped += 1
            else:
                errors.append(result)
        
        logger.info(f"Enhanced rules: {created} created, {skipped} skipped, total={len(ENHANCED_RULES)}")
        
        return {
            "created": created,
            "skipped": skipped,
            "total": len(ENHANCED_RULES),
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
    
    def get_rule_chunks(self, rule_id: str, voxel_threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        Get rule as chunks for Unity "Rule-Molecule" visualization.
        
        UNIFIED STRUCTURE (same as Papers):
        - chunk_id: 0 (positive) or 1 (negative)
        - chunk_type: "positive" or "negative"
        - text_preview: Phrases preview
        - embedding_b64: 768-dim BioBERT embedding
        - voxels: 8x8x12 grid for Unity rendering (VISIBLE cubes)
        - color: RGB (green=positive, red=negative)
        - position: 3D position (Chunk = INVISIBLE container)
        - connects_to: Link between positive and negative
        
        Args:
            rule_id: Rule identifier
            voxel_threshold: Threshold for voxel activation (default 0.3, DATAMODEL.md contract)
        
        Unity renders:
        1. Chunk = invisible cube at position (0,0,0 origin for voxels)
        2. Voxels = visible cubes INSIDE chunk
        3. Wires = connection showing polar relationship
        """
        rule = self.db.get_rule(rule_id)
        if not rule:
            return None
        
        chunks = []
        
        # Chunk 0: Positive embedding
        pos_emb = rule.get_pos_embedding_array()
        if pos_emb is not None:
            pos_b64 = base64.b64encode(pos_emb.tobytes()).decode('utf-8')
            positive_phrases = json.loads(rule.positive_phrases) if rule.positive_phrases else []
            
            # Create voxel grid from embedding (use configurable threshold, default 0.3)
            pos_voxels = VoxelGrid.from_embedding(pos_emb, threshold=voxel_threshold)
            
            chunks.append({
                "chunk_id": 0,
                "chunk_type": "positive",
                "text_preview": ", ".join(positive_phrases[:5]) + ("..." if len(positive_phrases) > 5 else ""),
                "embedding_b64": pos_b64,
                # NEW: Include voxels for Unity visualization
                "voxels": pos_voxels.to_dict(),
                "color": {"r": 0.2, "g": 0.9, "b": 0.3},  # Green for positive
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "connects_to": [1] if rule.neg_embedding else []
            })
        
        # Chunk 1: Negative embedding (if exists)
        neg_emb = rule.get_neg_embedding_array()
        if neg_emb is not None:
            neg_b64 = base64.b64encode(neg_emb.tobytes()).decode('utf-8')
            negative_phrases = json.loads(rule.negative_phrases) if rule.negative_phrases else []
            
            # Create voxel grid from embedding (use configurable threshold, default 0.3)
            neg_voxels = VoxelGrid.from_embedding(neg_emb, threshold=voxel_threshold)
            
            chunks.append({
                "chunk_id": 1,
                "chunk_type": "negative",
                "text_preview": ", ".join(negative_phrases[:5]) + ("..." if len(negative_phrases) > 5 else ""),
                "embedding_b64": neg_b64,
                # NEW: Include voxels for Unity visualization
                "voxels": neg_voxels.to_dict(),
                "color": {"r": 0.9, "g": 0.2, "b": 0.2},  # Red for negative
                "position": {"x": 2.0, "y": 0.0, "z": 0.0},
                "connects_to": []
            })
        
        return {
            "rule_id": rule_id,
            "question": rule.question,
            "threshold": rule.threshold,
            "chunks": chunks,
            "chunks_count": len(chunks),
            "molecule_config": {
                "embedding_dim": 768,
                "voxel_grid": [8, 8, 12],  # 8x8x12 = 768
                "layout": "dipole",  # Two opposing poles
                "scale": 1.0,
                "connection_type": "polar"
            }
        }
    
    def list_rules_with_chunks(self) -> List[Dict[str, Any]]:
        """
        List all active rules with their chunk representations.
        
        For Unity to render all rules as molecules at once.
        """
        rules = self.db.get_active_rules()
        result = []
        
        for rule in rules:
            rule_chunks = self.get_rule_chunks(rule.rule_id)
            if rule_chunks:
                result.append(rule_chunks)
        
        return result


# Singleton instance
_rule_handler: Optional[RuleHandler] = None


def get_rule_handler() -> RuleHandler:
    """Get singleton rule handler"""
    global _rule_handler
    if _rule_handler is None:
        _rule_handler = RuleHandler()
    return _rule_handler
