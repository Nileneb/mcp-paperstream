"""
Device Handler - Android device registration and management

Handles:
- POST /api/devices/register - Register new device
- GET /api/devices/{device_id} - Get device info
- GET /api/devices - List active devices
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from ..db import get_db, DeviceRegistry

logger = logging.getLogger(__name__)


class DeviceHandler:
    """Handler for Android device management"""
    
    def __init__(self):
        self.db = get_db()
        # Consider device inactive if not seen for 5 minutes
        self.inactive_threshold_minutes = 5
    
    def register_device(
        self,
        device_id: str,
        device_name: Optional[str] = None,
        device_model: Optional[str] = None,
        os_version: Optional[str] = None,
        app_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new Android device or update existing.
        
        Called by Android app: POST /api/devices/register
        
        Args:
            device_id: Unique device identifier
            device_name: User-defined device name
            device_model: Device model (e.g., "Pixel 6")
            os_version: Android version
            app_version: PaperRun app version
        
        Returns:
            {
                "status": "registered" | "updated",
                "device_id": str,
                "device": {...}
            }
        """
        # Check if device exists
        existing = self._get_device_by_id(device_id)
        
        device = DeviceRegistry(
            device_id=device_id,
            device_name=device_name,
            device_model=device_model,
            os_version=os_version,
            app_version=app_version
        )
        
        device = self.db.register_device(device)
        
        status = "updated" if existing else "registered"
        logger.info(f"Device {status}: {device_id} ({device_model})")
        
        return {
            "status": status,
            "device_id": device_id,
            "device": device.to_dict()
        }
    
    def _get_device_by_id(self, device_id: str) -> Optional[DeviceRegistry]:
        """Get device by ID"""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM device_registry WHERE device_id = ?",
                (device_id,)
            ).fetchone()
            if row:
                return DeviceRegistry(
                    id=row["id"],
                    device_id=row["device_id"],
                    device_name=row["device_name"],
                    device_model=row["device_model"],
                    os_version=row["os_version"],
                    app_version=row["app_version"],
                    jobs_completed=row["jobs_completed"],
                    accuracy=row["accuracy"],
                    is_active=bool(row["is_active"]),
                )
        return None
    
    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device info with stats"""
        device = self._get_device_by_id(device_id)
        if device:
            # Get leaderboard info
            with self.db.get_connection() as conn:
                lb_row = conn.execute(
                    "SELECT * FROM leaderboard WHERE device_id = ?",
                    (device_id,)
                ).fetchone()
            
            result = device.to_dict()
            if lb_row:
                result["leaderboard"] = {
                    "rank": lb_row["rank"],
                    "total_points": lb_row["total_points"],
                    "papers_validated": lb_row["papers_validated"],
                    "matches_found": lb_row["matches_found"]
                }
            return result
        return None
    
    def list_devices(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all devices"""
        devices = self.db.get_active_devices() if active_only else []
        
        if not active_only:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM device_registry ORDER BY last_seen DESC"
                ).fetchall()
                devices = [
                    DeviceRegistry(
                        id=r["id"],
                        device_id=r["device_id"],
                        device_name=r["device_name"],
                        device_model=r["device_model"],
                        jobs_completed=r["jobs_completed"],
                        is_active=bool(r["is_active"]),
                    )
                    for r in rows
                ]
        
        return [d.to_dict() for d in devices]
    
    def deactivate_device(self, device_id: str) -> bool:
        """Mark device as inactive"""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE device_registry SET is_active = 0 WHERE device_id = ?",
                (device_id,)
            )
            return conn.total_changes > 0
    
    def cleanup_inactive_devices(self) -> int:
        """Mark devices as inactive if not seen recently"""
        threshold = datetime.now() - timedelta(minutes=self.inactive_threshold_minutes)
        
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE device_registry 
                SET is_active = 0 
                WHERE last_seen < ? AND is_active = 1
                """,
                (threshold,)
            )
            count = conn.total_changes
        
        if count > 0:
            logger.info(f"Marked {count} devices as inactive")
        
        return count
    
    def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a device"""
        with self.db.get_connection() as conn:
            # Job stats
            job_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_assigned,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'expired' OR status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN status = 'completed' THEN similarity ELSE NULL END) as avg_similarity
                FROM validation_jobs
                WHERE assigned_to = ?
                """,
                (device_id,)
            ).fetchone()
            
            # Result stats
            result_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_results,
                    SUM(points_earned) as total_points,
                    AVG(time_taken_ms) as avg_time_ms,
                    SUM(CASE WHEN is_match THEN 1 ELSE 0 END) as matches_found
                FROM validation_results
                WHERE device_id = ?
                """,
                (device_id,)
            ).fetchone()
            
            return {
                "device_id": device_id,
                "jobs": {
                    "total_assigned": job_stats["total_assigned"],
                    "completed": job_stats["completed"],
                    "failed": job_stats["failed"],
                    "completion_rate": (
                        job_stats["completed"] / job_stats["total_assigned"] 
                        if job_stats["total_assigned"] > 0 else 0
                    )
                },
                "results": {
                    "total": result_stats["total_results"],
                    "total_points": result_stats["total_points"] or 0,
                    "avg_time_ms": result_stats["avg_time_ms"],
                    "matches_found": result_stats["matches_found"]
                },
                "avg_similarity": job_stats["avg_similarity"]
            }


# Singleton instance
_device_handler: Optional[DeviceHandler] = None


def get_device_handler() -> DeviceHandler:
    """Get singleton device handler"""
    global _device_handler
    if _device_handler is None:
        _device_handler = DeviceHandler()
    return _device_handler
