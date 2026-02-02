"""
Database Manager for MCP-PaperStream

SQLite database management with async support and connection pooling.
"""

import sqlite3
import asyncio
from pathlib import Path
from typing import Optional, List, Any, Dict
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import logging

from .models import (
    Paper, PaperSection, Rule, ValidationJob,
    DeviceRegistry, ValidationResult, PaperConsensus, Leaderboard
)

logger = logging.getLogger(__name__)

# Singleton instance
_db_instance: Optional["DatabaseManager"] = None


def get_db() -> "DatabaseManager":
    """Get or create singleton DatabaseManager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance


class DatabaseManager:
    """
    SQLite Database Manager for PaperStream.
    
    Handles all CRUD operations for papers, rules, jobs, devices, and results.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/paperstream.db
        """
        if db_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "paperstream.db")
        
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def initialize(self) -> bool:
        """
        Initialize database schema from migrations.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        migrations_dir = Path(__file__).parent / "migrations"
        
        # Run all migrations in order
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        with self.get_connection() as conn:
            for migration_file in migration_files:
                logger.info(f"Running migration: {migration_file.name}")
                try:
                    with open(migration_file, "r") as f:
                        # Execute line by line to handle ALTER TABLE errors gracefully
                        for statement in f.read().split(';'):
                            statement = statement.strip()
                            if statement and not statement.startswith('--'):
                                try:
                                    conn.execute(statement)
                                except sqlite3.OperationalError as e:
                                    # Ignore "duplicate column" errors for ALTER TABLE
                                    if "duplicate column" not in str(e).lower():
                                        logger.warning(f"Migration statement failed: {e}")
                except Exception as e:
                    logger.error(f"Migration {migration_file.name} failed: {e}")
        
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")
        return True
    
    # ==================== PAPERS ====================
    
    def create_paper(self, paper: Paper) -> Paper:
        """Insert a new paper"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO papers (paper_id, title, authors, journal, publication_date,
                                   pdf_url, pdf_local_path, status, source, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (paper.paper_id, paper.title, paper.authors, paper.journal,
                 paper.publication_date, paper.pdf_url, paper.pdf_local_path,
                 paper.status, paper.source, paper.priority)
            )
            paper.id = cursor.lastrowid
            paper.created_at = datetime.now()
        return paper
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by paper_id"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
        return None
    
    def get_papers_by_status(self, status: str, limit: int = 100) -> List[Paper]:
        """Get papers by status"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE status = ? ORDER BY priority DESC, created_at ASC LIMIT ?",
                (status, limit)
            ).fetchall()
            return [self._row_to_paper(r) for r in rows]
    
    def update_paper_status(self, paper_id: str, status: str) -> bool:
        """Update paper status"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE papers SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
                (status, paper_id)
            )
            return conn.total_changes > 0
    
    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["id"],
            paper_id=row["paper_id"],
            title=row["title"],
            authors=row["authors"],
            journal=row["journal"],
            publication_date=row["publication_date"],
            pdf_url=row["pdf_url"],
            pdf_local_path=row["pdf_local_path"],
            status=row["status"],
            source=row["source"],
            priority=row["priority"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    
    # ==================== PAPER SECTIONS ====================
    
    def create_section(self, section: PaperSection) -> PaperSection:
        """Insert a new paper section"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_sections (paper_id, section_name, section_text,
                                           page_number, image_path, embedding, voxel_data,
                                           color_r, color_g, color_b)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (section.paper_id, section.section_name, section.section_text,
                 section.page_number, section.image_path, section.embedding,
                 section.voxel_data, section.color_r, section.color_g, section.color_b)
            )
            section.id = cursor.lastrowid
        return section
    
    def get_sections_for_paper(self, paper_id: str) -> List[PaperSection]:
        """Get all sections for a paper"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_sections WHERE paper_id = ? ORDER BY page_number, id",
                (paper_id,)
            ).fetchall()
            return [self._row_to_section(r) for r in rows]
    
    def get_section(self, section_id: int) -> Optional[PaperSection]:
        """Get section by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM paper_sections WHERE id = ?", (section_id,)
            ).fetchone()
            if row:
                return self._row_to_section(row)
        return None
    
    def _row_to_section(self, row: sqlite3.Row) -> PaperSection:
        return PaperSection(
            id=row["id"],
            paper_id=row["paper_id"],
            section_name=row["section_name"],
            section_text=row["section_text"],
            page_number=row["page_number"],
            image_path=row["image_path"],
            embedding=row["embedding"],
            voxel_data=row["voxel_data"],
            color_r=row["color_r"],
            color_g=row["color_g"],
            color_b=row["color_b"],
        )
    
    # ==================== RULES ====================
    
    def create_rule(self, rule: Rule) -> Rule:
        """Insert a new rule"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO rules (rule_id, question, positive_phrases, negative_phrases,
                                  pos_embedding, neg_embedding, threshold, is_active, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rule.rule_id, rule.question, rule.positive_phrases, rule.negative_phrases,
                 rule.pos_embedding, rule.neg_embedding, rule.threshold, rule.is_active,
                 rule.created_by)
            )
            rule.id = cursor.lastrowid
        return rule
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get rule by rule_id"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM rules WHERE rule_id = ?", (rule_id,)
            ).fetchone()
            if row:
                return self._row_to_rule(row)
        return None
    
    def get_active_rules(self) -> List[Rule]:
        """Get ALL rules - Unity needs all rules to validate papers!
        
        Note: Returns ALL rules regardless of is_active flag.
        We want to check every paper against EVERY rule.
        """
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM rules ORDER BY created_at"
            ).fetchall()
            return [self._row_to_rule(r) for r in rows]
    
    def get_all_rules(self) -> List[Rule]:
        """Alias for get_active_rules - returns ALL rules"""
        return self.get_active_rules()
    
    def update_rule(self, rule: Rule) -> bool:
        """Update an existing rule"""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE rules SET question = ?, positive_phrases = ?, negative_phrases = ?,
                                pos_embedding = ?, neg_embedding = ?, threshold = ?,
                                is_active = ?, updated_at = CURRENT_TIMESTAMP
                WHERE rule_id = ?
                """,
                (rule.question, rule.positive_phrases, rule.negative_phrases,
                 rule.pos_embedding, rule.neg_embedding, rule.threshold,
                 rule.is_active, rule.rule_id)
            )
            return conn.total_changes > 0
    
    def _row_to_rule(self, row: sqlite3.Row) -> Rule:
        return Rule(
            id=row["id"],
            rule_id=row["rule_id"],
            question=row["question"],
            positive_phrases=row["positive_phrases"],
            negative_phrases=row["negative_phrases"],
            pos_embedding=row["pos_embedding"],
            neg_embedding=row["neg_embedding"],
            threshold=row["threshold"],
            is_active=bool(row["is_active"]),
            created_by=row["created_by"],
        )
    
    # ==================== VALIDATION JOBS ====================
    
    def create_job(self, job: ValidationJob) -> ValidationJob:
        """Insert a new validation job"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO validation_jobs (job_id, paper_id, section_id, rule_id,
                                            status, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job.job_id, job.paper_id, job.section_id, job.rule_id,
                 job.status, job.expires_at)
            )
            job.id = cursor.lastrowid
        return job
    
    def get_pending_jobs(self, limit: int = 10) -> List[ValidationJob]:
        """Get pending jobs ordered by creation time"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM validation_jobs 
                WHERE status = 'pending' 
                ORDER BY created_at ASC 
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            return [self._row_to_job(r) for r in rows]
    
    def assign_job(self, job_id: str, device_id: str, ttl_seconds: int = 300) -> bool:
        """Assign job to a device with expiration"""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE validation_jobs 
                SET status = 'assigned', assigned_to = ?, assigned_at = CURRENT_TIMESTAMP,
                    expires_at = ?
                WHERE job_id = ? AND status = 'pending'
                """,
                (device_id, expires_at, job_id)
            )
            return conn.total_changes > 0
    
    def complete_job(self, job_id: str, is_match: bool, similarity: float, confidence: float) -> bool:
        """Mark job as completed with results"""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE validation_jobs
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
                    is_match = ?, similarity = ?, confidence = ?
                WHERE job_id = ?
                """,
                (is_match, similarity, confidence, job_id)
            )
            return conn.total_changes > 0
    
    def get_expired_jobs(self) -> List[ValidationJob]:
        """Get jobs that have expired without completion"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM validation_jobs
                WHERE status = 'assigned' AND expires_at < CURRENT_TIMESTAMP
                """
            ).fetchall()
            return [self._row_to_job(r) for r in rows]
    
    def reassign_expired_jobs(self) -> int:
        """Reset expired jobs back to pending"""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE validation_jobs
                SET status = 'pending', assigned_to = NULL, assigned_at = NULL,
                    retry_count = retry_count + 1
                WHERE status = 'assigned' 
                  AND expires_at < CURRENT_TIMESTAMP
                  AND retry_count < max_retries
                """
            )
            count = conn.total_changes
            
            # Mark jobs that exceeded retries as failed
            conn.execute(
                """
                UPDATE validation_jobs
                SET status = 'failed'
                WHERE status = 'assigned'
                  AND expires_at < CURRENT_TIMESTAMP
                  AND retry_count >= max_retries
                """
            )
            return count
    
    def get_jobs_for_device(self, device_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get jobs assigned to or available for a device, with full data"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT j.*, 
                       s.section_text, s.embedding as section_embedding, s.voxel_data,
                       r.question, r.pos_embedding, r.neg_embedding, r.threshold
                FROM validation_jobs j
                JOIN paper_sections s ON j.section_id = s.id
                JOIN rules r ON j.rule_id = r.rule_id
                WHERE j.status = 'pending'
                ORDER BY j.created_at ASC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            
            jobs = []
            for row in rows:
                jobs.append({
                    "job_id": row["job_id"],
                    "paper_id": row["paper_id"],
                    "section_id": row["section_id"],
                    "rule_id": row["rule_id"],
                    "section_text": row["section_text"],
                    "voxel_data": row["voxel_data"],
                    "question": row["question"],
                    "threshold": row["threshold"],
                    # Embeddings als base64 für Transport
                    "section_embedding": row["section_embedding"],
                    "pos_embedding": row["pos_embedding"],
                    "neg_embedding": row["neg_embedding"],
                })
            return jobs
    
    def _row_to_job(self, row: sqlite3.Row) -> ValidationJob:
        return ValidationJob(
            id=row["id"],
            job_id=row["job_id"],
            paper_id=row["paper_id"],
            section_id=row["section_id"],
            rule_id=row["rule_id"],
            status=row["status"],
            assigned_to=row["assigned_to"],
            is_match=row["is_match"],
            similarity=row["similarity"],
            confidence=row["confidence"],
            retry_count=row["retry_count"],
        )
    
    # ==================== DEVICES ====================
    
    def register_device(self, device: DeviceRegistry) -> DeviceRegistry:
        """Register a new device or update existing"""
        with self.get_connection() as conn:
            # Try update first
            conn.execute(
                """
                INSERT INTO device_registry (device_id, device_name, device_model, 
                                            os_version, app_version, is_active, last_seen)
                VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(device_id) DO UPDATE SET
                    device_name = excluded.device_name,
                    device_model = excluded.device_model,
                    os_version = excluded.os_version,
                    app_version = excluded.app_version,
                    is_active = 1,
                    last_seen = CURRENT_TIMESTAMP
                """,
                (device.device_id, device.device_name, device.device_model,
                 device.os_version, device.app_version)
            )
            
            # Get the device back
            row = conn.execute(
                "SELECT * FROM device_registry WHERE device_id = ?", 
                (device.device_id,)
            ).fetchone()
            return self._row_to_device(row)
    
    def update_device_seen(self, device_id: str) -> bool:
        """Update device last_seen timestamp"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE device_registry SET last_seen = CURRENT_TIMESTAMP WHERE device_id = ?",
                (device_id,)
            )
            return conn.total_changes > 0
    
    def get_active_devices(self) -> List[DeviceRegistry]:
        """Get all active devices"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM device_registry WHERE is_active = 1 ORDER BY last_seen DESC"
            ).fetchall()
            return [self._row_to_device(r) for r in rows]
    
    def get_least_recently_assigned_device(self) -> Optional[DeviceRegistry]:
        """Get device that was least recently assigned a job (for fair distribution)"""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM device_registry 
                WHERE is_active = 1 
                ORDER BY last_assigned ASC NULLS FIRST
                LIMIT 1
                """
            ).fetchone()
            if row:
                return self._row_to_device(row)
        return None
    
    def _row_to_device(self, row: sqlite3.Row) -> DeviceRegistry:
        return DeviceRegistry(
            id=row["id"],
            device_id=row["device_id"],
            device_name=row["device_name"],
            device_model=row["device_model"],
            os_version=row["os_version"],
            app_version=row["app_version"],
            jobs_completed=row["jobs_completed"],
            avg_processing_time=row["avg_processing_time"],
            accuracy=row["accuracy"],
            is_active=bool(row["is_active"]),
        )
    
    # ==================== VALIDATION RESULTS ====================
    
    def submit_result(self, result: ValidationResult) -> ValidationResult:
        """Submit a validation result"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO validation_results (job_id, device_id, paper_id, rule_id,
                                               section_id, is_match, similarity, confidence,
                                               points_earned, time_taken_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (result.job_id, result.device_id, result.paper_id, result.rule_id,
                 result.section_id, result.is_match, result.similarity, result.confidence,
                 result.points_earned, result.time_taken_ms)
            )
            result.id = cursor.lastrowid
            
            # Update device stats
            conn.execute(
                """
                UPDATE device_registry 
                SET jobs_completed = jobs_completed + 1,
                    last_seen = CURRENT_TIMESTAMP
                WHERE device_id = ?
                """,
                (result.device_id,)
            )
        return result
    
    def get_results_for_paper_rule(self, paper_id: str, rule_id: str) -> List[ValidationResult]:
        """Get all results for a paper-rule combination"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM validation_results 
                WHERE paper_id = ? AND rule_id = ?
                ORDER BY submitted_at
                """,
                (paper_id, rule_id)
            ).fetchall()
            return [self._row_to_result(r) for r in rows]
    
    def _row_to_result(self, row: sqlite3.Row) -> ValidationResult:
        return ValidationResult(
            id=row["id"],
            job_id=row["job_id"],
            device_id=row["device_id"],
            paper_id=row["paper_id"],
            rule_id=row["rule_id"],
            section_id=row["section_id"],
            is_match=bool(row["is_match"]),
            similarity=row["similarity"],
            confidence=row["confidence"],
            points_earned=row["points_earned"],
            time_taken_ms=row["time_taken_ms"],
        )
    
    # ==================== CONSENSUS ====================
    
    def update_consensus(self, paper_id: str, rule_id: str) -> Optional[PaperConsensus]:
        """Calculate and update consensus for paper-rule combination"""
        results = self.get_results_for_paper_rule(paper_id, rule_id)
        
        if not results:
            return None
        
        vote_count = len(results)
        matches = [r for r in results if r.is_match]
        match_count = len(matches)
        
        avg_similarity = sum(r.similarity for r in results) / vote_count
        avg_confidence = sum(r.confidence for r in results) / vote_count
        agreement_ratio = max(match_count, vote_count - match_count) / vote_count
        
        # Majority vote
        is_match = match_count > (vote_count / 2)
        
        # Sections where matches were found
        sections = list(set(r.section_id for r in results if r.is_match))
        
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_consensus (paper_id, rule_id, is_match, avg_similarity,
                                            avg_confidence, vote_count, agreement_ratio,
                                            found_in_sections, is_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, rule_id) DO UPDATE SET
                    is_match = excluded.is_match,
                    avg_similarity = excluded.avg_similarity,
                    avg_confidence = excluded.avg_confidence,
                    vote_count = excluded.vote_count,
                    agreement_ratio = excluded.agreement_ratio,
                    found_in_sections = excluded.found_in_sections,
                    is_validated = excluded.is_validated,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (paper_id, rule_id, is_match, avg_similarity, avg_confidence,
                 vote_count, agreement_ratio, json.dumps(sections),
                 vote_count >= 3)  # min 3 votes for validation
            )
        
        return PaperConsensus(
            paper_id=paper_id,
            rule_id=rule_id,
            is_match=is_match,
            avg_similarity=avg_similarity,
            avg_confidence=avg_confidence,
            vote_count=vote_count,
            agreement_ratio=agreement_ratio,
            found_in_sections=json.dumps(sections),
            is_validated=vote_count >= 3,
        )
    
    def get_consensus(self, paper_id: str, rule_id: str) -> Optional[PaperConsensus]:
        """Get consensus for paper-rule"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM paper_consensus WHERE paper_id = ? AND rule_id = ?",
                (paper_id, rule_id)
            ).fetchone()
            if row:
                return PaperConsensus(
                    id=row["id"],
                    paper_id=row["paper_id"],
                    rule_id=row["rule_id"],
                    is_match=bool(row["is_match"]) if row["is_match"] is not None else None,
                    avg_similarity=row["avg_similarity"],
                    avg_confidence=row["avg_confidence"],
                    vote_count=row["vote_count"],
                    agreement_ratio=row["agreement_ratio"],
                    found_in_sections=row["found_in_sections"],
                    is_validated=bool(row["is_validated"]),
                )
        return None
    
    # ==================== LEADERBOARD ====================
    
    def update_leaderboard(self, device_id: str, points: int, found_match: bool) -> Leaderboard:
        """Update leaderboard for a device"""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO leaderboard (device_id, total_points, papers_validated, matches_found)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    total_points = total_points + excluded.total_points,
                    papers_validated = papers_validated + 1,
                    matches_found = matches_found + excluded.matches_found,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (device_id, points, 1 if found_match else 0)
            )
            
            # Update ranks
            conn.execute(
                """
                UPDATE leaderboard SET rank = (
                    SELECT COUNT(*) + 1 FROM leaderboard l2 
                    WHERE l2.total_points > leaderboard.total_points
                )
                """
            )
            
            row = conn.execute(
                "SELECT * FROM leaderboard WHERE device_id = ?",
                (device_id,)
            ).fetchone()
            
            return Leaderboard(
                id=row["id"],
                device_id=row["device_id"],
                player_name=row["player_name"],
                total_points=row["total_points"],
                papers_validated=row["papers_validated"],
                matches_found=row["matches_found"],
                rank=row["rank"],
            )
    
    def get_leaderboard(self, limit: int = 100) -> List[Leaderboard]:
        """Get top players"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM leaderboard ORDER BY total_points DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [
                Leaderboard(
                    id=r["id"],
                    device_id=r["device_id"],
                    player_name=r["player_name"],
                    total_points=r["total_points"],
                    papers_validated=r["papers_validated"],
                    matches_found=r["matches_found"],
                    rank=r["rank"],
                )
                for r in rows
            ]
    
    # ==================== STATS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            
            # Paper counts
            row = conn.execute(
                "SELECT status, COUNT(*) as count FROM papers GROUP BY status"
            ).fetchall()
            stats["papers"] = {r["status"]: r["count"] for r in row}
            
            # Job counts
            row = conn.execute(
                "SELECT status, COUNT(*) as count FROM validation_jobs GROUP BY status"
            ).fetchall()
            stats["jobs"] = {r["status"]: r["count"] for r in row}
            
            # Device count
            row = conn.execute(
                "SELECT COUNT(*) as count FROM device_registry WHERE is_active = 1"
            ).fetchone()
            stats["active_devices"] = row["count"]
            
            # Rule count
            row = conn.execute(
                "SELECT COUNT(*) as count FROM rules WHERE is_active = 1"
            ).fetchone()
            stats["active_rules"] = row["count"]
            
            # Validated papers
            row = conn.execute(
                "SELECT COUNT(DISTINCT paper_id) as count FROM paper_consensus WHERE is_validated = 1"
            ).fetchone()
            stats["validated_papers"] = row["count"]
            
            return stats
