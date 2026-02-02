"""
Moltbook Identity Authentication Handler
========================================
Verifiziert Moltbook Identity Tokens und erstellt Game Sessions.
"""

import os
import time
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

# Konfiguration
MOLTBOOK_APP_KEY = os.getenv("MOLTBOOK_APP_KEY", "")  # Developer App Key
MOLTBOOK_API_BASE = "https://www.moltbook.com/api/v1"
GAME_AUDIENCE = os.getenv("GAME_AUDIENCE", "mcp.linn.games")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))


@dataclass
class MoltbookAgent:
    """Verifizierter Moltbook Agent"""
    id: str
    name: str
    description: str = ""
    karma: int = 0
    avatar_url: Optional[str] = None
    is_claimed: bool = False
    follower_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    owner: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameSession:
    """Aktive Spielsession"""
    session_id: str
    player_id: str
    moltbook_id: str
    agent_name: str
    karma: int
    created_at: float
    expires_at: float
    
    def is_valid(self) -> bool:
        return time.time() < self.expires_at
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "agent_name": self.agent_name,
            "karma": self.karma,
            "expires_in": int(self.expires_at - time.time())
        }


class PlayerData(BaseModel):
    """Spielerdaten in der DB"""
    player_id: str
    moltbook_id: str
    name: str
    karma: int
    avatar_url: Optional[str] = None
    owner_handle: Optional[str] = None
    
    # Game Stats
    total_score: int = 0
    matches_found: int = 0
    papers_validated: int = 0
    accuracy: float = 0.0
    
    # Timestamps
    created_at: str = ""
    last_login: str = ""


# In-Memory Stores (später: Redis/DB)
_sessions: Dict[str, GameSession] = {}
_players: Dict[str, dict] = {}


async def verify_moltbook_token(identity_token: str) -> Optional[MoltbookAgent]:
    """
    Verifiziert ein Moltbook Identity Token via API.
    
    Returns:
        MoltbookAgent bei Erfolg, None bei Fehler
    """
    if not MOLTBOOK_APP_KEY:
        print("⚠️  MOLTBOOK_APP_KEY nicht konfiguriert - Auth deaktiviert")
        return None
    
    url = f"{MOLTBOOK_API_BASE}/agents/verify-identity"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers={"X-Moltbook-App-Key": MOLTBOOK_APP_KEY},
                json={
                    "token": identity_token,
                    "audience": GAME_AUDIENCE
                },
                timeout=10.0
            )
        except httpx.RequestError as e:
            print(f"❌ Moltbook API error: {e}")
            return None
    
    if response.status_code != 200:
        print(f"❌ Moltbook verification failed: {response.status_code}")
        return None
    
    data = response.json()
    
    if not data.get("valid"):
        error = data.get("error", "Unknown")
        print(f"❌ Token invalid: {error}")
        return None
    
    agent_data = data.get("agent", {})
    
    return MoltbookAgent(
        id=agent_data.get("id", ""),
        name=agent_data.get("name", "Unknown"),
        description=agent_data.get("description", ""),
        karma=agent_data.get("karma", 0),
        avatar_url=agent_data.get("avatar_url"),
        is_claimed=agent_data.get("is_claimed", False),
        follower_count=agent_data.get("follower_count", 0),
        stats=agent_data.get("stats", {}),
        owner=agent_data.get("owner", {})
    )


def create_session(agent: MoltbookAgent) -> GameSession:
    """Erstellt eine neue Game Session für einen Agent."""
    session_id = secrets.token_urlsafe(32)
    player_id = f"mb_{agent.id[:8]}"
    
    now = time.time()
    expires_at = now + (SESSION_TTL_HOURS * 3600)
    
    session = GameSession(
        session_id=session_id,
        player_id=player_id,
        moltbook_id=agent.id,
        agent_name=agent.name,
        karma=agent.karma,
        created_at=now,
        expires_at=expires_at
    )
    
    _sessions[session_id] = session
    
    # Player in DB anlegen/updaten
    if player_id not in _players:
        _players[player_id] = {
            "player_id": player_id,
            "moltbook_id": agent.id,
            "name": agent.name,
            "karma": agent.karma,
            "avatar_url": agent.avatar_url,
            "owner_handle": agent.owner.get("x_handle"),
            "total_score": 0,
            "matches_found": 0,
            "papers_validated": 0,
            "accuracy": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
        print(f"✅ New player registered: {agent.name} (Karma: {agent.karma})")
    else:
        _players[player_id].update({
            "name": agent.name,
            "karma": agent.karma,
            "avatar_url": agent.avatar_url,
            "owner_handle": agent.owner.get("x_handle"),
            "last_login": datetime.utcnow().isoformat()
        })
        print(f"✅ Player logged in: {agent.name} (Karma: {agent.karma})")
    
    return session


def validate_session(session_token: str) -> Optional[GameSession]:
    """Prüft ob ein Session Token gültig ist."""
    session = _sessions.get(session_token)
    
    if not session:
        return None
    
    if not session.is_valid():
        # Abgelaufene Session entfernen
        del _sessions[session_token]
        return None
    
    return session


def get_player(player_id: str) -> Optional[dict]:
    """Holt Spielerdaten."""
    return _players.get(player_id)


def update_player_stats(player_id: str, score: int, is_match: bool) -> bool:
    """Aktualisiert Spielerstatistiken nach Validation."""
    player = _players.get(player_id)
    if not player:
        return False
    
    player["total_score"] += score
    player["papers_validated"] += 1
    if is_match:
        player["matches_found"] += 1
    
    # Accuracy berechnen
    if player["papers_validated"] > 0:
        player["accuracy"] = player["matches_found"] / player["papers_validated"]
    
    return True


def get_leaderboard(limit: int = 10) -> list:
    """Top Spieler nach Score."""
    sorted_players = sorted(
        _players.values(),
        key=lambda p: p.get("total_score", 0),
        reverse=True
    )[:limit]
    
    return [
        {
            "rank": i + 1,
            "name": p.get("name", "Unknown"),
            "score": p.get("total_score", 0),
            "karma": p.get("karma", 0),
            "accuracy": round(p.get("accuracy", 0) * 100, 1)
        }
        for i, p in enumerate(sorted_players)
    ]


# === Dev/Test Mode: Skip Moltbook ===

def create_dev_session(device_id: str, name: str = "TestPlayer") -> GameSession:
    """
    Erstellt eine Dev-Session ohne Moltbook-Verification.
    NUR FÜR ENTWICKLUNG!
    """
    session_id = f"dev_{secrets.token_urlsafe(16)}"
    player_id = f"dev_{device_id[:8]}"
    
    now = time.time()
    
    session = GameSession(
        session_id=session_id,
        player_id=player_id,
        moltbook_id="",
        agent_name=name,
        karma=0,
        created_at=now,
        expires_at=now + 86400  # 24h
    )
    
    _sessions[session_id] = session
    
    if player_id not in _players:
        _players[player_id] = {
            "player_id": player_id,
            "moltbook_id": "",
            "name": name,
            "karma": 0,
            "total_score": 0,
            "matches_found": 0,
            "papers_validated": 0,
            "accuracy": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
    
    print(f"🔧 Dev session created: {name} ({player_id})")
    return session


def cleanup_expired_sessions():
    """Entfernt abgelaufene Sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if s.expires_at < now]
    for sid in expired:
        del _sessions[sid]
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired sessions")
