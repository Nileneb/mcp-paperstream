
"""
DiffusionBERTScore IoT Server
=============================
Basierend auf server_paperstream.py von Nileneb/bitvavo-mcp
Erweitert für verteilte BERTScore-Berechnung auf IoT-Geräten

Architektur:
- Coordinator (dieses Skript): Verteilt Embedding-Berechnung
- IoT Workers: Berechnen Teilembeddings mit TinyBERT
- Aggregator: Führt BERTScore aus den Teilembeddings zusammen
"""

from __future__ import annotations
import os, json, time, asyncio, hmac, hashlib
from typing import Any, Dict, List, Optional, Tuple, cast
from enum import Enum
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse

load_dotenv()

# Lazy import für Handler (vermeidet Circular Imports)
_biobert_handler = None

def _get_biobert_handler():
    """Lazy-load BioBERT Handler"""
    global _biobert_handler
    if _biobert_handler is None:
        try:
            from .handlers.biobert_handler import get_handler
            _biobert_handler = get_handler()
        except ImportError:
            _biobert_handler = None
    return _biobert_handler

# Config laden
def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}

# ========================= 
# MCP Server Setup
# =========================
mcp = FastMCP(
    name="diffusion-bertscore-iot",
    instructions="Verteilter BERTScore für IoT-Geräte. SSE für Task-Verteilung, REST für Ergebnisse.",
    version="0.1.0",
)

# ========================= 
# Konfiguration
# =========================
HOST = os.getenv("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.getenv("FASTMCP_PORT", "8089"))
SSE_PATH = os.getenv("SSE_PATH", "/sse-bertscore")
RESULT_PATH = os.getenv("RESULT_PATH", "/bert-result")
HEALTH_PATH = os.getenv("HEALTH_PATH", "/health")
HMAC_SECRET = os.getenv("BERTSCORE_HMAC", "")

# IoT-spezifische Konfiguration
ASSIGN_TTL = int(os.getenv("ASSIGN_TTL", "30"))  # Kürzere TTL für IoT
MAX_INFLIGHT_PER_CLIENT = int(os.getenv("MAX_INFLIGHT_PER_CLIENT", "1"))  # IoT = weniger parallel
MIN_CLIENTS_FOR_DISTRIBUTED = int(os.getenv("MIN_CLIENTS", "2"))  # Mind. 2 Clients für Verteilung

# TinyBERT Layer-Verteilung (6 Layer bei TinyBERT)
TINYBERT_LAYERS = 6
EMBEDDING_DIM = 312  # TinyBERT hidden size

# =========================
# Enums und Modelle
# =========================
class TaskType(str, Enum):
    EMBEDDING = "embedding"      # Berechne Embeddings für Tokens
    ATTENTION = "attention"      # Berechne Attention-Block
    POOLING = "pooling"          # Finale Aggregation
    BERTSCORE = "bertscore"      # BERTScore aus Embeddings

class DeviceCapability(str, Enum):
    LOW = "low"          # ESP32, Raspberry Pi Zero (~1 Layer)
    MEDIUM = "medium"    # Raspberry Pi 4, alte Smartphones (~2-3 Layer)
    HIGH = "high"        # Moderne Smartphones, Tablets (alle Layer)

class IoTClient(BaseModel):
    """Repräsentiert ein verbundenes IoT-Gerät"""
    client_id: str
    device_type: str = "unknown"
    capability: DeviceCapability = DeviceCapability.MEDIUM
    assigned_layers: List[int] = Field(default_factory=lambda: cast(List[int], []))
    connected_at: float = Field(default_factory=lambda: time.time())
    last_heartbeat: float = Field(default_factory=lambda: time.time())
    tasks_completed: int = 0
    avg_latency_ms: float = 0.0

class EmbeddingTask(BaseModel):
    """Eine Teilaufgabe für IoT-Geräte"""
    task_id: str
    job_id: str
    task_type: TaskType
    layer_range: Tuple[int, int] = (0, 6)  # Welche Layer berechnen
    tokens: List[str] = Field(default_factory=lambda: cast(List[str], []))
    token_ids: List[int] = Field(default_factory=lambda: cast(List[int], []))
    status: str = "queued"
    client_id: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    deadline: float = 0.0
    result: Optional[Dict[str, Any]] = None

class BERTScoreJob(BaseModel):
    """Ein vollständiger BERTScore-Berechnungsjob"""
    job_id: str
    reference_text: str
    candidate_text: str
    tasks: List[EmbeddingTask] = Field(default_factory=lambda: cast(List[EmbeddingTask], []))
    partial_embeddings: Dict[str, Any] = Field(default_factory=lambda: cast(Dict[str, Any], {}))
    final_score: Optional[Dict[str, Any]] = None
    status: str = "pending"
    created_at: float = Field(default_factory=lambda: time.time())

# =========================
# Globaler State
# =========================
clients: Dict[str, IoTClient] = {}
clients_queues: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}
clients_inflight: Dict[str, int] = {}
jobs: Dict[str, BERTScoreJob] = {}
pending_tasks: List[EmbeddingTask] = []

# =========================
# Hilfsfunktionen
# =========================
def _now_ms() -> int:
    return int(time.time() * 1000)

def _hmac_sign(payload: str) -> str:
    if not HMAC_SECRET:
        return ""
    return hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _tokenize(text: str) -> Tuple[List[str], List[int]]:
    """
    Tokenisiert Text mit BioBERT-Tokenizer.
    
    Falls BioBERT nicht verfügbar, Fallback auf einfache Tokenisierung.
    
    Returns:
        (tokens, token_ids): Token-Liste und deren IDs
    """
    handler = _get_biobert_handler()
    if handler is not None:
        return handler.tokenize(text)
    
    # Fallback: einfache Tokenisierung (ohne echte IDs)
    tokens = text.lower().split()
    token_ids = list(range(len(tokens)))  # Pseudo-IDs
    return tokens, token_ids


def _tokenize_simple(text: str) -> List[str]:
    """Legacy-Funktion für Kompatibilität"""
    tokens, _ = _tokenize(text)
    return tokens

def _get_best_client(task_type: TaskType, required_layers: Tuple[int, int]) -> Optional[str]:
    """
    Wählt den besten Client basierend auf:
    1. Capability (kann die Layer berechnen?)
    2. Aktuelle Last (weniger inflight = besser)
    3. Latenz-Historie (schnellere Geräte bevorzugen)
    """
    if not clients_queues:
        return None
    
    suitable_clients: List[Tuple[str, IoTClient]] = []
    for cid, client in clients.items():
        if cid not in clients_queues:
            continue
        
        # Prüfe ob Client die benötigten Layer berechnen kann
        layer_count = required_layers[1] - required_layers[0]
        if client.capability == DeviceCapability.LOW and layer_count > 1:
            continue
        if client.capability == DeviceCapability.MEDIUM and layer_count > 3:
            continue
        
        # Prüfe ob Client nicht überlastet
        if clients_inflight.get(cid, 0) >= MAX_INFLIGHT_PER_CLIENT:
            continue
            
        suitable_clients.append((cid, client))
    
    if not suitable_clients:
        return None
    
    # Sortiere nach: inflight ASC, avg_latency ASC
    suitable_clients.sort(key=lambda x: (
        clients_inflight.get(x[0], 0),
        x[1].avg_latency_ms
    ))
    
    return suitable_clients[0][0]

def _split_into_layer_chunks(num_clients: int) -> List[Tuple[int, int]]:
    """
    Teilt die TinyBERT-Layer auf verfügbare Clients auf
    
    Beispiel mit 4 Clients und 6 Layern:
    - Client 1: Layer 0-1 (Embedding + erste Attention)
    - Client 2: Layer 2-3
    - Client 3: Layer 4-5
    - Client 4: Pooling/Aggregation
    """
    if num_clients <= 1:
        return [(0, TINYBERT_LAYERS)]
    
    # Reserviere einen Client für finale Aggregation
    compute_clients = num_clients - 1
    layers_per_client = TINYBERT_LAYERS // compute_clients
    
    chunks: List[Tuple[int, int]] = []
    for i in range(compute_clients):
        start = i * layers_per_client
        end = start + layers_per_client if i < compute_clients - 1 else TINYBERT_LAYERS
        chunks.append((start, end))
    
    return chunks

# =========================
# Task-Verteilung
# =========================
async def _assign_task(task: EmbeddingTask):
    """Weist Task dem besten verfügbaren Client zu"""
    client_id = _get_best_client(task.task_type, task.layer_range)
    
    if client_id is None:
        pending_tasks.append(task)
        return
    
    await _deliver_task(client_id, task)

async def _deliver_task(client_id: str, task: EmbeddingTask):
    """Sendet Task an Client über SSE"""
    clients_inflight[client_id] = clients_inflight.get(client_id, 0) + 1
    task.client_id = client_id
    task.status = "assigned"
    task.deadline = time.time() + ASSIGN_TTL
    
    message: Dict[str, Any] = {
        "type": "task",
        "task_id": task.task_id,
        "job_id": task.job_id,
        "task_type": task.task_type.value,
        "layer_range": task.layer_range,
        "tokens": task.tokens,
        "token_ids": task.token_ids,
        "ttl_s": ASSIGN_TTL,
        "sig": _hmac_sign(task.task_id),
    }
    
    await clients_queues[client_id].put(message)
    asyncio.create_task(_watch_task_expiry(task))

async def _watch_task_expiry(task: EmbeddingTask):
    """Überwacht Task-Timeout und reassigned bei Bedarf"""
    await asyncio.sleep(max(1, ASSIGN_TTL))
    
    if task.status == "done":
        return
    
    task.status = "expired"
    if task.client_id:
        clients_inflight[task.client_id] = max(0, clients_inflight.get(task.client_id, 1) - 1)
    
    # Retry mit neuem Task
    job = jobs.get(task.job_id)
    if job and job.status != "completed":
        new_task = EmbeddingTask(
            task_id=f"t_{_now_ms()}",
            job_id=task.job_id,
            task_type=task.task_type,
            layer_range=task.layer_range,
            tokens=task.tokens,
            token_ids=task.token_ids,
        )
        job.tasks.append(new_task)
        await _assign_task(new_task)

async def _drain_pending_tasks():
    """Verteilt wartende Tasks wenn Clients verfügbar werden"""
    remaining: List[EmbeddingTask] = []
    for task in pending_tasks:
        client_id = _get_best_client(task.task_type, task.layer_range)
        if client_id is None:
            remaining.append(task)
        else:
            await _deliver_task(client_id, task)
    
    pending_tasks.clear()
    pending_tasks.extend(remaining)

# =========================
# Embedding-Aggregation
# =========================
def _aggregate_embeddings(job: BERTScoreJob) -> Dict[str, Any]:
    """
    Kombiniert Teil-Embeddings von verschiedenen IoT-Geräten
    zu einem vollständigen Embedding-Vektor
    """
    # Sammle alle Teil-Embeddings sortiert nach Layer
    all_embeddings: List[Dict[str, Any]] = []
    for task in job.tasks:
        if task.status == "done" and task.result:
            all_embeddings.append({
                "layer_range": task.layer_range,
                "embedding": task.result.get("embedding", [])
            })
    
    # Sortiere nach Layer-Start
    all_embeddings.sort(key=lambda x: x["layer_range"][0])
    
    # Kombiniere (in echt: komplexere Aggregation nötig)
    combined: List[float] = []
    for emb in all_embeddings:
        combined.extend(emb.get("embedding", []))
    
    return {
        "combined_embedding": combined,
        "num_parts": len(all_embeddings),
        "total_dim": len(combined)
    }

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Berechnet Cosine Similarity zwischen zwei Vektoren"""
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _calculate_bertscore_from_embeddings(
    ref_embeddings: List[List[float]], 
    cand_embeddings: List[List[float]]
) -> Dict[str, Any]:
    """
    Berechnet ECHTEN BERTScore aus Token-Level Embeddings.
    
    Formel (Zhang et al., 2019):
    - Precision = (1/|cand|) * Σ max_j(cos_sim(cand_i, ref_j))
    - Recall = (1/|ref|) * Σ max_i(cos_sim(ref_j, cand_i))  
    - F1 = 2 * P * R / (P + R)
    
    Args:
        ref_embeddings: Liste von Embedding-Vektoren für Reference-Tokens
        cand_embeddings: Liste von Embedding-Vektoren für Candidate-Tokens
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    if not ref_embeddings or not cand_embeddings:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "Empty embeddings"
        }
    
    # Precision: Für jeden Candidate-Token, finde max Similarity zu irgendeinem Reference-Token
    precision_scores = []
    for cand_emb in cand_embeddings:
        max_sim = max(_cosine_similarity(cand_emb, ref_emb) for ref_emb in ref_embeddings)
        precision_scores.append(max_sim)
    precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    
    # Recall: Für jeden Reference-Token, finde max Similarity zu irgendeinem Candidate-Token
    recall_scores = []
    for ref_emb in ref_embeddings:
        max_sim = max(_cosine_similarity(ref_emb, cand_emb) for cand_emb in cand_embeddings)
        recall_scores.append(max_sim)
    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _compute_bertscore_real(reference: str, candidate: str) -> Dict[str, Any]:
    """
    Berechnet ECHTEN BERTScore mit BioBERT-Embeddings.
    
    Diese Funktion ersetzt den Placeholder und macht die eigentliche Arbeit.
    """
    handler = _get_biobert_handler()
    
    if handler is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "BioBERT handler not available",
            "mode": "failed"
        }
    
    # Tokenisiere beide Texte
    ref_tokens, ref_ids = handler.tokenize(reference)
    cand_tokens, cand_ids = handler.tokenize(candidate)
    
    # Berechne Token-Level Embeddings
    ref_embeddings = handler.embed_tokens(ref_ids)
    cand_embeddings = handler.embed_tokens(cand_ids)
    
    # Berechne BERTScore
    scores = _calculate_bertscore_from_embeddings(ref_embeddings, cand_embeddings)
    scores["mode"] = "local_biobert"
    scores["ref_tokens"] = len(ref_tokens)
    scores["cand_tokens"] = len(cand_tokens)
    
    return scores


def _calculate_bertscore(ref_embedding: List[float], cand_embedding: List[float]) -> Dict[str, Any]:
    """
    Legacy-Funktion für Kompatibilität mit verteilter Berechnung.
    Wird verwendet wenn Teil-Embeddings von IoT-Geräten aggregiert wurden.
    """
    if not ref_embedding or not cand_embedding:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "computed_locally": True,
        }
    
    # Einfache Cosine Similarity für aggregierte Embeddings
    similarity = _cosine_similarity(ref_embedding, cand_embedding)
    
    return {
        "precision": round(similarity, 4),
        "recall": round(similarity, 4),
        "f1": round(similarity, 4),
        "computed_locally": False,
    }

# =========================
# MCP Tools
# =========================
@mcp.tool(tags={"public"})
async def bertscore_compute(
    reference: str,
    candidate: str,
    distributed: bool = True
) -> Dict[str, Any]:
    """
    Berechnet BERTScore für Reference vs. Candidate Text.
    
    Bei distributed=True wird die Berechnung auf IoT-Geräte verteilt.
    Bei distributed=False oder wenn keine Worker verbunden sind, läuft alles lokal.
    
    Args:
        reference: Der Referenztext (z.B. Original-Paper-Abschnitt)
        candidate: Der zu bewertende Text (z.B. Review-Kommentar)
        distributed: Ob auf IoT-Geräte verteilt werden soll
    
    Returns:
        {
            "job_id": str,
            "status": "processing" | "completed",
            "score": {"precision": float, "recall": float, "f1": float} | None,
            "distributed_to": int (Anzahl Clients)
        }
    """
    job_id = f"job_{_now_ms()}"
    
    job = BERTScoreJob(
        job_id=job_id,
        reference_text=reference,
        candidate_text=candidate,
    )
    jobs[job_id] = job
    
    num_clients = len(clients_queues)
    
    # Wenn keine IoT-Worker oder distributed=False: Lokale Berechnung mit echtem BioBERT
    if not distributed or num_clients < MIN_CLIENTS_FOR_DISTRIBUTED:
        # ECHTE Berechnung statt Placeholder!
        job.final_score = _compute_bertscore_real(reference, candidate)
        job.status = "completed"
        
        if num_clients < MIN_CLIENTS_FOR_DISTRIBUTED:
            job.final_score["reason"] = f"Only {num_clients} IoT clients connected, computed locally"
        
        return {
            "job_id": job_id,
            "status": "completed",
            "score": job.final_score,
            "distributed_to": 0,
        }
    
    # Verteilte Berechnung
    ref_tokens = _tokenize_simple(reference)
    cand_tokens = _tokenize_simple(candidate)
    layer_chunks = _split_into_layer_chunks(num_clients)
    
    # Erstelle Tasks für Reference-Embeddings
    for i, chunk in enumerate(layer_chunks):
        task = EmbeddingTask(
            task_id=f"t_ref_{_now_ms()}_{i}",
            job_id=job_id,
            task_type=TaskType.EMBEDDING,
            layer_range=chunk,
            tokens=ref_tokens,
        )
        job.tasks.append(task)
        await _assign_task(task)
    
    # Erstelle Tasks für Candidate-Embeddings
    for i, chunk in enumerate(layer_chunks):
        task = EmbeddingTask(
            task_id=f"t_cand_{_now_ms()}_{i}",
            job_id=job_id,
            task_type=TaskType.EMBEDDING,
            layer_range=chunk,
            tokens=cand_tokens,
        )
        job.tasks.append(task)
        await _assign_task(task)
    
    job.status = "processing"
    
    return {
        "job_id": job_id,
        "status": "processing",
        "score": None,
        "distributed_to": num_clients,
        "tasks_created": len(job.tasks),
    }

@mcp.tool(tags={"public"})
async def bertscore_status(job_id: str) -> Dict[str, Any]:
    """
    Prüft den Status eines laufenden BERTScore-Jobs.
    
    Returns:
        {
            "job_id": str,
            "status": str,
            "tasks_total": int,
            "tasks_done": int,
            "score": dict | None
        }
    """
    job = jobs.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    
    tasks_done = sum(1 for t in job.tasks if t.status == "done")
    tasks_total = len(job.tasks)
    
    # Prüfe ob alle Tasks fertig
    if tasks_done == tasks_total and tasks_total > 0 and job.status != "completed":
        # Aggregiere Embeddings und berechne Score
        ref_agg = _aggregate_embeddings(job)
        job.final_score = _calculate_bertscore(
            ref_agg.get("combined_embedding", []),
            ref_agg.get("combined_embedding", [])  # Vereinfacht
        )
        job.final_score["mode"] = "distributed"
        job.status = "completed"
    
    return {
        "job_id": job_id,
        "status": job.status,
        "tasks_total": tasks_total,
        "tasks_done": tasks_done,
        "score": job.final_score,
    }

@mcp.tool(tags={"public"})
async def register_iot_client(
    client_id: str,
    device_type: str = "unknown",
    capability: str = "medium"
) -> Dict[str, Any]:
    """
    Registriert ein neues IoT-Gerät als Worker.
    
    Args:
        client_id: Eindeutige ID des Geräts
        device_type: z.B. "raspberry_pi", "smartphone", "esp32"
        capability: "low", "medium", oder "high"
    
    Returns:
        {"registered": bool, "assigned_layers": list}
    """
    cap = DeviceCapability(capability) if capability in ["low", "medium", "high"] else DeviceCapability.MEDIUM
    
    client = IoTClient(
        client_id=client_id,
        device_type=device_type,
        capability=cap,
    )
    
    # Weise Layer basierend auf Capability zu
    if cap == DeviceCapability.LOW:
        client.assigned_layers = [0]  # Nur Embedding-Layer
    elif cap == DeviceCapability.MEDIUM:
        client.assigned_layers = [0, 1, 2]  # Erste Hälfte
    else:
        client.assigned_layers = list(range(TINYBERT_LAYERS))  # Alle Layer
    
    clients[client_id] = client
    clients_queues[client_id] = asyncio.Queue()
    clients_inflight[client_id] = 0
    
    # Versuche wartende Tasks zu verteilen
    await _drain_pending_tasks()
    
    return {
        "registered": True,
        "client_id": client_id,
        "capability": cap.value,
        "assigned_layers": client.assigned_layers,
    }

@mcp.tool(tags={"internal"})
async def submit_task_result(
    task_id: str,
    job_id: str,
    client_id: str,
    embedding: List[float],
    latency_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Empfängt Ergebnis einer Task-Berechnung von einem IoT-Gerät.
    
    Args:
        task_id: ID der abgeschlossenen Task
        job_id: ID des übergeordneten Jobs
        client_id: ID des Clients der die Task bearbeitet hat
        embedding: Berechnetes (Teil-)Embedding
        latency_ms: Zeit die die Berechnung gedauert hat
    
    Returns:
        {"accepted": bool, "job_status": str}
    """
    job = jobs.get(job_id)
    if not job:
        return {"accepted": False, "error": "Job not found"}
    
    # Finde die Task
    task = next((t for t in job.tasks if t.task_id == task_id), None)
    if not task:
        return {"accepted": False, "error": "Task not found"}
    
    # Update Task
    task.status = "done"
    task.result = {"embedding": embedding}
    
    # Update Client-Stats
    if client_id in clients:
        client = clients[client_id]
        client.tasks_completed += 1
        # Running average für Latenz
        client.avg_latency_ms = (
            (client.avg_latency_ms * (client.tasks_completed - 1) + latency_ms) 
            / client.tasks_completed
        )
    
    # Decrement inflight
    clients_inflight[client_id] = max(0, clients_inflight.get(client_id, 1) - 1)
    
    # Drain pending tasks
    await _drain_pending_tasks()
    
    return {
        "accepted": True,
        "job_status": job.status,
        "tasks_remaining": sum(1 for t in job.tasks if t.status != "done"),
    }

@mcp.tool(tags={"public"})
async def get_system_stats() -> Dict[str, Any]:
    """
    Gibt Statistiken über das verteilte System zurück.
    
    Returns:
        {
            "connected_clients": int,
            "clients_by_capability": dict,
            "pending_tasks": int,
            "active_jobs": int,
            "total_tasks_completed": int
        }
    """
    by_capability = {"low": 0, "medium": 0, "high": 0}
    total_completed = 0
    
    for client in clients.values():
        by_capability[client.capability.value] += 1
        total_completed += client.tasks_completed
    
    active_jobs = sum(1 for j in jobs.values() if j.status == "processing")
    
    return {
        "connected_clients": len(clients),
        "clients_by_capability": by_capability,
        "pending_tasks": len(pending_tasks),
        "active_jobs": active_jobs,
        "total_tasks_completed": total_completed,
        "jobs_total": len(jobs),
    }


@mcp.tool(tags={"public"})
async def get_paper_thumbnail(
    paper_id: str,
    width: int = 200,
    height: int = 280,
    regenerate: bool = False
) -> Dict[str, Any]:
    """
    Generiert oder liefert ein Thumbnail (PNG) der ersten PDF-Seite.
    
    Für Unity-Integration: Gibt Base64-encoded PNG zurück.
    Thumbnails werden gecacht für schnellere Abfragen.
    
    Args:
        paper_id: Paper ID (z.B. DOI oder PMC-ID)
        width: Thumbnail-Breite in Pixel (default: 200)
        height: Thumbnail-Höhe in Pixel (default: 280)
        regenerate: True um Cache zu ignorieren und neu zu generieren
    
    Returns:
        {
            "status": "success" | "error" | "no_pdf",
            "paper_id": str,
            "thumbnail_base64": str (PNG als Base64),
            "thumbnail_path": str (Dateipfad),
            "width": int,
            "height": int,
            "cached": bool
        }
    """
    from .api.paper_handler import get_paper_handler
    handler = get_paper_handler()
    
    result = handler.get_thumbnail(
        paper_id=paper_id,
        width=width,
        height=height,
        as_base64=True,
        regenerate=regenerate
    )
    
    return result

# =========================
# SSE Endpoint für IoT-Clients
# =========================
async def sse_handler(request: Request):
    """Server-Sent Events endpoint für IoT-Clients"""
    client_id = request.query_params.get("client_id")
    
    if not client_id or client_id not in clients:
        return PlainTextResponse("Client not registered", status_code=401)
    
    async def event_stream():
        queue = clients_queues.get(client_id)
        if not queue:
            return
        
        # Initial heartbeat
        yield f"data: {json.dumps({'type': 'connected', 'client_id': client_id})}\n\n"
        
        while True:
            try:
                # Warte auf Tasks mit Timeout für Heartbeats
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat senden
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': _now_ms()})}\n\n"
                    if client_id in clients:
                        clients[client_id].last_heartbeat = time.time()
            except Exception:
                break
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Route registrieren (FastMCP 2.x verwendet custom_route_handler)
@mcp.custom_route(SSE_PATH, methods=["GET"])
async def sse_route(request: Request):
    return await sse_handler(request)

# =========================
# ASGI App für uvicorn
# =========================
# FastMCP 2.x braucht http_app() für uvicorn
app = mcp.http_app()

# =========================
# Main
# =========================
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting DiffusionBERTScore IoT Server on {HOST}:{PORT}")
    print(f"📡 SSE Endpoint: {SSE_PATH}")
    print(f"⏱️  Task TTL: {ASSIGN_TTL}s")
    print(f"📊 Max inflight per client: {MAX_INFLIGHT_PER_CLIENT}")
    uvicorn.run(app, host=HOST, port=PORT)
