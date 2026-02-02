"""
Unity Authentication Handler
============================
Verifiziert Unity Gaming Services idTokens via JWKS.
"""

import os
import time
import secrets
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import httpx
import jwt
from jwt import PyJWKClient

# Konfiguration
UNITY_PROJECT_ID = os.getenv("UNITY_PROJECT_ID", "")  # Deine Unity Project ID
UNITY_JWKS_URL = "https://player-auth.services.api.unity.com/.well-known/jwks.json"
UNITY_ISSUER = "https://player-auth.services.api.unity.com"
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))

# JWKS Client (cached)
_jwks_client: Optional[PyJWKClient] = None
_jwks_last_refresh: float = 0
JWKS_REFRESH_INTERVAL = 8 * 3600  # 8 Stunden


@dataclass
class UnityPlayer:
    """Verifizierter Unity Player"""
    player_id: str
    project_id: str
    environment_id: Optional[str] = None
    external_ids: list = field(default_factory=list)
    token_id: Optional[str] = None  # jti claim


@dataclass
class GameSession:
    """Aktive Spielsession"""
    session_id: str
    player_id: str
    unity_player_id: str
    created_at: float
    expires_at: float
    
    def is_valid(self) -> bool:
        return time.time() < self.expires_at
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "unity_player_id": self.unity_player_id,
            "expires_in": int(self.expires_at - time.time())
        }


# In-Memory Stores (später: Redis/DB)
_sessions: Dict[str, GameSession] = {}
_players: Dict[str, dict] = {}


def _get_jwks_client() -> PyJWKClient:
    """Holt oder erstellt JWKS Client mit Caching."""
    global _jwks_client, _jwks_last_refresh
    
    now = time.time()
    
    # Refresh JWKS alle 8 Stunden
    if _jwks_client is None or (now - _jwks_last_refresh) > JWKS_REFRESH_INTERVAL:
        _jwks_client = PyJWKClient(UNITY_JWKS_URL)
        _jwks_last_refresh = now
        print(f"🔑 JWKS refreshed from Unity")
    
    return _jwks_client


async def verify_unity_token(id_token: str) -> Optional[UnityPlayer]:
    """
    Verifiziert ein Unity idToken via JWKS.
    
    Args:
        id_token: JWT idToken von Unity Authentication
        
    Returns:
        UnityPlayer bei Erfolg, None bei Fehler
    """
    try:
        # JWKS Client holen
        jwks_client = _get_jwks_client()
        
        # Signing Key aus Token Header extrahieren
        signing_key = jwks_client.get_signing_key_from_jwt(id_token)
        
        # Token verifizieren und decodieren
        payload = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=UNITY_ISSUER,
            options={
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iss": True,
                "require": ["exp", "nbf", "sub", "iss"]
            }
        )
        
        # Optional: Project ID prüfen
        token_project_id = payload.get("project_id", "")
        if UNITY_PROJECT_ID and token_project_id != UNITY_PROJECT_ID:
            print(f"⚠️ Project ID mismatch: {token_project_id} != {UNITY_PROJECT_ID}")
            # Wir loggen nur, blockieren nicht (für Dev-Flexibilität)
        
        player = UnityPlayer(
            player_id=payload.get("sub", ""),
            project_id=token_project_id,
            environment_id=payload.get("environment_id"),
            token_id=payload.get("jti")
        )
        
        print(f"✅ Unity token verified: {player.player_id[:8]}...")
        return player
        
    except jwt.ExpiredSignatureError:
        print("❌ Unity token expired")
        return None
    except jwt.InvalidIssuerError:
        print("❌ Unity token invalid issuer")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Unity token invalid: {e}")
        return None
    except Exception as e:
        print(f"❌ Unity token verification error: {e}")
        return None


def create_session(player: UnityPlayer) -> GameSession:
    """Erstellt eine neue Game Session für einen Unity Player."""
    session_id = secrets.token_urlsafe(32)
    # Player ID = Unity Player ID (unique per project)
    player_id = f"unity_{player.player_id[:16]}"
    
    now = time.time()
    expires_at = now + (SESSION_TTL_HOURS * 3600)
    
    session = GameSession(
        session_id=session_id,
        player_id=player_id,
        unity_player_id=player.player_id,
        created_at=now,
        expires_at=expires_at
    )
    
    _sessions[session_id] = session
    
    # Player in DB anlegen/updaten
    if player_id not in _players:
        _players[player_id] = {
            "player_id": player_id,
            "unity_player_id": player.player_id,
            "project_id": player.project_id,
            "total_score": 0,
            "matches_found": 0,
            "papers_validated": 0,
            "accuracy": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
        print(f"✅ New Unity player registered: {player_id}")
    else:
        _players[player_id]["last_login"] = datetime.utcnow().isoformat()
        print(f"✅ Unity player logged in: {player_id}")
    
    return session


def validate_session(session_token: str) -> Optional[GameSession]:
    """Prüft ob ein Session Token gültig ist."""
    session = _sessions.get(session_token)
    
    if not session:
        return None
    
    if not session.is_valid():
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
            "player_id": p.get("player_id", "Unknown")[:12] + "...",
            "score": p.get("total_score", 0),
            "accuracy": round(p.get("accuracy", 0) * 100, 1)
        }
        for i, p in enumerate(sorted_players)
    ]


# === Dev Mode: Anonymous without Unity ===

def create_anonymous_session(device_id: str) -> GameSession:
    """
    Erstellt eine anonyme Session ohne Unity Auth.
    Für lokales Testing.
    """
    session_id = f"anon_{secrets.token_urlsafe(16)}"
    player_id = f"anon_{device_id[:12]}"
    
    now = time.time()
    
    session = GameSession(
        session_id=session_id,
        player_id=player_id,
        unity_player_id="",
        created_at=now,
        expires_at=now + 86400
    )
    
    _sessions[session_id] = session
    
    if player_id not in _players:
        _players[player_id] = {
            "player_id": player_id,
            "unity_player_id": "",
            "project_id": "",
            "total_score": 0,
            "matches_found": 0,
            "papers_validated": 0,
            "accuracy": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
    
    print(f"🔧 Anonymous session created: {player_id}")
    return session


def cleanup_expired_sessions():
    """Entfernt abgelaufene Sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if s.expires_at < now]
    for sid in expired:
        del _sessions[sid]
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired sessions")
