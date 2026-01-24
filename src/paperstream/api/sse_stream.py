"""
Unity SSE Stream

Server-Sent Events endpoint for real-time updates to Unity game client.

Events:
- paper_validated: When consensus is reached for a paper
- leaderboard_update: When leaderboard changes
- new_paper: When a new paper is ready for validation
- device_joined: When a new device registers
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator, Set
from datetime import datetime
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Server-Sent Event"""
    event: str
    data: Dict[str, Any]
    id: Optional[str] = None
    retry: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    
    def format(self) -> str:
        """Format as SSE message"""
        lines = []
        
        if self.id:
            lines.append(f"id: {self.id}")
        
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        lines.append(f"event: {self.event}")
        lines.append(f"data: {json.dumps(self.data)}")
        
        return "\n".join(lines) + "\n\n"


class UnitySSEStream:
    """
    Manages SSE connections and event broadcasting to Unity clients.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._event_id = 0
        self._running = False
        self._heartbeat_interval = 30  # seconds
    
    async def subscribe(self, client_id: str) -> AsyncGenerator[str, None]:
        """
        Subscribe to SSE stream.
        
        Args:
            client_id: Unique client identifier
        
        Yields:
            SSE formatted messages
        """
        queue = asyncio.Queue()
        self._subscribers[client_id] = queue
        
        logger.info(f"Unity client subscribed: {client_id}")
        
        # Send initial connection event
        yield SSEEvent(
            event="connected",
            data={
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to PaperStream SSE"
            },
            id=str(self._next_event_id())
        ).format()
        
        try:
            while True:
                try:
                    # Wait for events with timeout for heartbeat
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self._heartbeat_interval
                    )
                    yield event.format()
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield SSEEvent(
                        event="heartbeat",
                        data={"timestamp": datetime.now().isoformat()},
                        id=str(self._next_event_id())
                    ).format()
                    
        except asyncio.CancelledError:
            pass
        finally:
            self._subscribers.pop(client_id, None)
            logger.info(f"Unity client disconnected: {client_id}")
    
    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast event to all connected Unity clients.
        
        Args:
            event_type: Event type name
            data: Event data
        """
        if not self._subscribers:
            return
        
        event = SSEEvent(
            event=event_type,
            data=data,
            id=str(self._next_event_id())
        )
        
        for client_id, queue in list(self._subscribers.items()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for client {client_id}, dropping event")
    
    # =========================
    # Event Emitters
    # =========================
    
    async def emit_paper_validated(
        self,
        paper_id: str,
        title: str,
        rules_results: Dict[str, bool],
        voxel_data: Optional[Dict[str, Any]] = None,
        thumbnail_base64: Optional[str] = None
    ):
        """Emit when paper validation is complete"""
        payload = {
            "paper_id": paper_id,
            "title": title,
            "rules_results": rules_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include voxel data for Unity mesh generation
        if voxel_data:
            payload["voxel_data"] = voxel_data
        
        # Include thumbnail for Unity UI
        if thumbnail_base64:
            payload["thumbnail_base64"] = thumbnail_base64
        
        await self.broadcast("paper_validated", payload)
    
    async def emit_leaderboard_update(
        self,
        top_players: list,
        changed_ranks: list
    ):
        """Emit when leaderboard changes"""
        await self.broadcast("leaderboard_update", {
            "top_players": top_players,
            "changed_ranks": changed_ranks,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_new_paper(
        self,
        paper_id: str,
        title: str,
        sections_count: int
    ):
        """Emit when new paper is ready for validation"""
        await self.broadcast("new_paper", {
            "paper_id": paper_id,
            "title": title,
            "sections_count": sections_count,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_device_joined(
        self,
        device_id: str,
        device_name: str,
        total_devices: int
    ):
        """Emit when new device registers"""
        await self.broadcast("device_joined", {
            "device_id": device_id,
            "device_name": device_name,
            "total_devices": total_devices,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_validation_progress(
        self,
        paper_id: str,
        rule_id: str,
        vote_count: int,
        votes_needed: int
    ):
        """Emit validation progress update"""
        await self.broadcast("validation_progress", {
            "paper_id": paper_id,
            "rule_id": rule_id,
            "vote_count": vote_count,
            "votes_needed": votes_needed,
            "progress_percent": min(100, int(vote_count / max(votes_needed, 1) * 100)),
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_stats_update(self, stats: Dict[str, Any]):
        """Emit periodic stats update"""
        await self.broadcast("stats_update", {
            **stats,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self._subscribers)
    
    def get_connected_clients(self) -> list:
        """Get list of connected client IDs"""
        return list(self._subscribers.keys())


# Singleton instance
_sse_stream: Optional[UnitySSEStream] = None


def get_unity_sse_stream() -> UnitySSEStream:
    """Get singleton SSE stream"""
    global _sse_stream
    if _sse_stream is None:
        _sse_stream = UnitySSEStream()
    return _sse_stream
