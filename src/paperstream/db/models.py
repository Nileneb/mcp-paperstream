"""
Database Models for MCP-PaperStream

Dataclasses representing all database entities.
Uses Python 3.12 dataclasses with optional fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import json
import numpy as np


@dataclass
class Paper:
    """Scientific paper entity"""
    paper_id: str
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_local_path: Optional[str] = None
    status: str = "pending"  # pending, downloading, processing, ready, failed
    downloaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    source: str = "n8n"
    priority: int = 5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> dict:
        # Handle created_at that might be string or datetime
        created_at_str = None
        if self.created_at:
            if isinstance(self.created_at, str):
                created_at_str = self.created_at
            else:
                created_at_str = self.created_at.isoformat()
        
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "pdf_url": self.pdf_url,
            "pdf_local_path": self.pdf_local_path,
            "status": self.status,
            "source": self.source,
            "priority": self.priority,
            "created_at": created_at_str,
        }


@dataclass
class PaperSection:
    """Section of a paper with embeddings"""
    paper_id: str
    section_name: str
    section_text: Optional[str] = None
    page_number: Optional[int] = None
    image_path: Optional[str] = None
    embedding: Optional[bytes] = None  # numpy float32 as bytes
    voxel_data: Optional[str] = None   # JSON string
    color_r: Optional[float] = None
    color_g: Optional[float] = None
    color_b: Optional[float] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Convert stored bytes back to numpy array"""
        if self.embedding is None:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    def set_embedding_array(self, arr: np.ndarray):
        """Store numpy array as bytes"""
        self.embedding = np.array(arr, dtype=np.float32).tobytes()
    
    def get_voxel_grid(self) -> Optional[List]:
        """Parse voxel_data JSON"""
        if self.voxel_data is None:
            return None
        return json.loads(self.voxel_data)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "section_name": self.section_name,
            "section_text": self.section_text[:200] + "..." if self.section_text and len(self.section_text) > 200 else self.section_text,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "has_embedding": self.embedding is not None,
            "has_voxel_data": self.voxel_data is not None,
            "color": [self.color_r, self.color_g, self.color_b] if self.color_r else None,
        }


@dataclass
class Rule:
    """Validation rule with BioBERT embeddings"""
    rule_id: str
    question: str
    positive_phrases: Optional[str] = None   # JSON array
    negative_phrases: Optional[str] = None   # JSON array
    pos_embedding: Optional[bytes] = None    # numpy float32
    neg_embedding: Optional[bytes] = None    # numpy float32
    threshold: float = 0.75
    is_active: bool = True
    created_by: str = "system"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def get_positive_phrases_list(self) -> List[str]:
        if self.positive_phrases is None:
            return []
        return json.loads(self.positive_phrases)
    
    def get_negative_phrases_list(self) -> List[str]:
        if self.negative_phrases is None:
            return []
        return json.loads(self.negative_phrases)
    
    def get_pos_embedding_array(self) -> Optional[np.ndarray]:
        if self.pos_embedding is None:
            return None
        return np.frombuffer(self.pos_embedding, dtype=np.float32)
    
    def get_neg_embedding_array(self) -> Optional[np.ndarray]:
        if self.neg_embedding is None:
            return None
        return np.frombuffer(self.neg_embedding, dtype=np.float32)
    
    def to_dict(self, include_embeddings: bool = False) -> dict:
        result = {
            "id": self.id,
            "rule_id": self.rule_id,
            "question": self.question,
            "positive_phrases": self.get_positive_phrases_list(),
            "negative_phrases": self.get_negative_phrases_list(),
            "threshold": self.threshold,
            "is_active": self.is_active,
            "created_by": self.created_by,
        }
        if include_embeddings:
            pos_emb = self.get_pos_embedding_array()
            neg_emb = self.get_neg_embedding_array()
            result["pos_embedding"] = pos_emb.tolist() if pos_emb is not None else None
            result["neg_embedding"] = neg_emb.tolist() if neg_emb is not None else None
        return result


@dataclass
class ValidationJob:
    """Job to be distributed to Android devices"""
    job_id: str
    paper_id: str
    section_id: int
    rule_id: str
    status: str = "pending"  # pending, assigned, completed, failed, expired
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_match: Optional[bool] = None
    similarity: Optional[float] = None
    confidence: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "paper_id": self.paper_id,
            "section_id": self.section_id,
            "rule_id": self.rule_id,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "is_match": self.is_match,
            "similarity": self.similarity,
            "confidence": self.confidence,
        }


@dataclass
class DeviceRegistry:
    """Registered Android device"""
    device_id: str
    device_name: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    jobs_completed: int = 0
    avg_processing_time: Optional[float] = None
    accuracy: Optional[float] = None
    is_active: bool = True
    last_seen: Optional[datetime] = None
    last_assigned: Optional[datetime] = None
    registered_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_model": self.device_model,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "jobs_completed": self.jobs_completed,
            "accuracy": self.accuracy,
            "is_active": self.is_active,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class ValidationResult:
    """Result submitted by Android device"""
    job_id: str
    device_id: str
    paper_id: str
    rule_id: str
    section_id: int
    is_match: bool
    similarity: float
    confidence: float
    points_earned: Optional[int] = None
    time_taken_ms: Optional[int] = None
    submitted_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "device_id": self.device_id,
            "paper_id": self.paper_id,
            "rule_id": self.rule_id,
            "section_id": self.section_id,
            "is_match": self.is_match,
            "similarity": self.similarity,
            "confidence": self.confidence,
            "points_earned": self.points_earned,
            "time_taken_ms": self.time_taken_ms,
        }


@dataclass
class PaperConsensus:
    """Aggregated validation results for paper + rule"""
    paper_id: str
    rule_id: str
    is_match: Optional[bool] = None
    avg_similarity: Optional[float] = None
    avg_confidence: Optional[float] = None
    vote_count: int = 0
    agreement_ratio: Optional[float] = None
    found_in_sections: Optional[str] = None  # JSON array
    is_validated: bool = False
    min_votes_required: int = 3
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def get_sections_list(self) -> List[int]:
        if self.found_in_sections is None:
            return []
        return json.loads(self.found_in_sections)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "rule_id": self.rule_id,
            "is_match": self.is_match,
            "avg_similarity": self.avg_similarity,
            "avg_confidence": self.avg_confidence,
            "vote_count": self.vote_count,
            "agreement_ratio": self.agreement_ratio,
            "found_in_sections": self.get_sections_list(),
            "is_validated": self.is_validated,
        }


@dataclass
class Leaderboard:
    """Gamification leaderboard entry"""
    device_id: str
    player_name: Optional[str] = None
    total_points: int = 0
    papers_validated: int = 0
    matches_found: int = 0
    accuracy: Optional[float] = None
    streak: int = 0
    max_streak: int = 0
    achievements: Optional[str] = None  # JSON array
    rank: Optional[int] = None
    last_updated: Optional[datetime] = None
    id: Optional[int] = None
    
    def get_achievements_list(self) -> List[str]:
        if self.achievements is None:
            return []
        return json.loads(self.achievements)
    
    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "device_id": self.device_id,
            "player_name": self.player_name,
            "total_points": self.total_points,
            "papers_validated": self.papers_validated,
            "matches_found": self.matches_found,
            "accuracy": self.accuracy,
            "streak": self.streak,
            "max_streak": self.max_streak,
            "achievements": self.get_achievements_list(),
        }


@dataclass
class ProcessingQueueItem:
    """Item in the processing queue"""
    paper_id: str
    task_type: str  # download, extract, embed, voxelize
    status: str = "pending"
    priority: int = 5
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None
