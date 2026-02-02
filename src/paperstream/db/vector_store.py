"""
Qdrant Vector Store Integration

Provides efficient vector storage and similarity search for:
- Paper embeddings (BioBERT 768-dim)
- Rule embeddings (pos/neg)
- Batch processing and similarity queries
"""

import os
import logging
import base64
import struct
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Qdrant client (lazy loaded)
_qdrant_client = None

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Collection names
PAPERS_COLLECTION = "papers"
RULES_COLLECTION = "rules"

# BioBERT embedding dimension
VECTOR_DIM = 768


def _to_qdrant_id(string_id: str) -> str:
    """Convert string ID to valid Qdrant UUID"""
    import hashlib
    import uuid
    # Create deterministic UUID from string
    hash_bytes = hashlib.md5(string_id.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


@dataclass
class VectorMatch:
    """Result from similarity search"""
    id: str
    score: float
    payload: Dict[str, Any]


def get_qdrant_client():
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        except ImportError:
            logger.error("qdrant-client not installed. Run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    return _qdrant_client


def init_collections():
    """Initialize Qdrant collections for papers and rules"""
    from qdrant_client.models import Distance, VectorParams
    
    client = get_qdrant_client()
    
    # Papers collection
    if not client.collection_exists(PAPERS_COLLECTION):
        client.create_collection(
            collection_name=PAPERS_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {PAPERS_COLLECTION}")
    
    # Rules collection (stores both pos and neg embeddings)
    if not client.collection_exists(RULES_COLLECTION):
        client.create_collection(
            collection_name=RULES_COLLECTION,
            vectors_config={
                "positive": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                "negative": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            }
        )
        logger.info(f"Created collection: {RULES_COLLECTION}")
    
    return True


def decode_embedding(embedding_b64: str) -> List[float]:
    """Decode base64 embedding to float list"""
    raw = base64.b64decode(embedding_b64)
    return list(struct.unpack(f'{len(raw)//4}f', raw))


def encode_embedding(embedding: List[float]) -> str:
    """Encode float list to base64"""
    raw = struct.pack(f'{len(embedding)}f', *embedding)
    return base64.b64encode(raw).decode('ascii')


# =========================
# Paper Operations
# =========================

def upsert_paper_embedding(
    paper_id: str,
    embedding: List[float],
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store or update a paper embedding in Qdrant.
    
    Args:
        paper_id: Unique paper identifier
        embedding: 768-dim BioBERT embedding
        payload: Optional metadata (title, doi, etc.)
    """
    from qdrant_client.models import PointStruct
    
    client = get_qdrant_client()
    
    # Store original paper_id in payload for retrieval
    full_payload = payload or {}
    full_payload["original_id"] = paper_id
    
    point = PointStruct(
        id=_to_qdrant_id(paper_id),  # Convert to UUID
        vector=embedding,
        payload=full_payload
    )
    
    client.upsert(
        collection_name=PAPERS_COLLECTION,
        points=[point]
    )
    
    logger.debug(f"Upserted paper embedding: {paper_id}")
    return True


def upsert_paper_embedding_b64(
    paper_id: str,
    embedding_b64: str,
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """Store paper embedding from base64 encoded string"""
    embedding = decode_embedding(embedding_b64)
    return upsert_paper_embedding(paper_id, embedding, payload)


def batch_upsert_papers(
    papers: List[Tuple[str, List[float], Optional[Dict]]]
) -> int:
    """
    Batch insert/update multiple paper embeddings.
    
    Args:
        papers: List of (paper_id, embedding, payload) tuples
    
    Returns:
        Number of papers upserted
    """
    from qdrant_client.models import PointStruct
    
    client = get_qdrant_client()
    
    points = []
    for paper_id, embedding, payload in papers:
        full_payload = payload or {}
        full_payload["original_id"] = paper_id
        points.append(PointStruct(
            id=_to_qdrant_id(paper_id),  # Convert to UUID
            vector=embedding,
            payload=full_payload
        ))
    
    client.upsert(
        collection_name=PAPERS_COLLECTION,
        points=points
    )
    
    logger.info(f"Batch upserted {len(points)} paper embeddings")
    return len(points)


def search_similar_papers(
    query_embedding: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[VectorMatch]:
    """
    Find papers similar to query embedding.
    
    Args:
        query_embedding: 768-dim query vector
        limit: Max results to return
        score_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of VectorMatch with paper_id, score, payload
    """
    client = get_qdrant_client()
    
    results = client.search(
        collection_name=PAPERS_COLLECTION,
        query_vector=query_embedding,
        limit=limit,
        score_threshold=score_threshold
    )
    
    return [
        VectorMatch(
            id=str(hit.id),
            score=hit.score,
            payload=hit.payload or {}
        )
        for hit in results
    ]


def get_paper_embedding(paper_id: str) -> Optional[List[float]]:
    """Retrieve a paper's embedding from Qdrant"""
    client = get_qdrant_client()
    
    results = client.retrieve(
        collection_name=PAPERS_COLLECTION,
        ids=[_to_qdrant_id(paper_id)],  # Convert to UUID
        with_vectors=True
    )
    
    if results:
        return results[0].vector
    return None


# =========================
# Rule Operations
# =========================

def upsert_rule_embeddings(
    rule_id: str,
    pos_embedding: List[float],
    neg_embedding: List[float],
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store rule embeddings (positive and negative) in Qdrant.
    
    Args:
        rule_id: Unique rule identifier
        pos_embedding: Positive example embedding
        neg_embedding: Negative example embedding
        payload: Optional metadata (rule_key, description, etc.)
    """
    from qdrant_client.models import PointStruct
    
    client = get_qdrant_client()
    
    # Store original rule_id in payload
    full_payload = payload or {}
    full_payload["original_id"] = rule_id
    
    point = PointStruct(
        id=_to_qdrant_id(rule_id),  # Convert to UUID
        vector={
            "positive": pos_embedding,
            "negative": neg_embedding
        },
        payload=full_payload
    )
    
    client.upsert(
        collection_name=RULES_COLLECTION,
        points=[point]
    )
    
    logger.debug(f"Upserted rule embeddings: {rule_id}")
    return True


def upsert_rule_embeddings_b64(
    rule_id: str,
    pos_embedding_b64: str,
    neg_embedding_b64: str,
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """Store rule embeddings from base64 encoded strings"""
    pos_embedding = decode_embedding(pos_embedding_b64)
    neg_embedding = decode_embedding(neg_embedding_b64)
    return upsert_rule_embeddings(rule_id, pos_embedding, neg_embedding, payload)


def match_paper_to_rules(
    paper_embedding: List[float],
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Match a paper embedding against all rules.
    
    Returns rules sorted by positive similarity - negative similarity.
    
    Args:
        paper_embedding: Paper's BioBERT embedding
        limit: Max rules to return
    
    Returns:
        List of dicts with rule_id, pos_score, neg_score, delta
    """
    client = get_qdrant_client()
    
    # Get all rules with their embeddings
    rules = client.scroll(
        collection_name=RULES_COLLECTION,
        limit=1000,
        with_vectors=True
    )[0]
    
    results = []
    for rule in rules:
        pos_vec = rule.vector.get("positive", [])
        neg_vec = rule.vector.get("negative", [])
        
        # Compute cosine similarities
        pos_score = _cosine_similarity(paper_embedding, pos_vec)
        neg_score = _cosine_similarity(paper_embedding, neg_vec)
        delta = pos_score - neg_score
        
        results.append({
            "rule_id": str(rule.id),
            "pos_score": pos_score,
            "neg_score": neg_score,
            "delta": delta,
            "payload": rule.payload or {}
        })
    
    # Sort by delta (highest first = best match to positive)
    results.sort(key=lambda x: x["delta"], reverse=True)
    
    return results[:limit]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


# =========================
# Batch Processing
# =========================

def sync_papers_from_sqlite(db_path: str = "data/paperstream.db") -> int:
    """
    Sync all paper embeddings from SQLite to Qdrant.
    
    Useful for initial migration or re-sync.
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get paper sections with embeddings
    cursor.execute("""
        SELECT ps.paper_id, p.title, ps.embedding 
        FROM paper_sections ps
        LEFT JOIN papers p ON ps.paper_id = p.paper_id
        WHERE ps.embedding IS NOT NULL
    """)
    
    # Group sections by paper and average embeddings
    paper_embeddings = {}
    for row in cursor.fetchall():
        paper_id = row["paper_id"]
        if row["embedding"]:
            try:
                embedding = decode_embedding(row["embedding"])
                if paper_id not in paper_embeddings:
                    paper_embeddings[paper_id] = {
                        "embeddings": [],
                        "title": row["title"] or ""
                    }
                paper_embeddings[paper_id]["embeddings"].append(embedding)
            except Exception as e:
                logger.warning(f"Failed to decode embedding for paper {paper_id}: {e}")
    
    conn.close()
    
    # Average embeddings per paper and upsert
    papers = []
    for paper_id, data in paper_embeddings.items():
        if data["embeddings"]:
            avg_embedding = [sum(x)/len(x) for x in zip(*data["embeddings"])]
            papers.append((
                paper_id,
                avg_embedding,
                {"title": data["title"], "paper_id": paper_id}
            ))
    
    if papers:
        return batch_upsert_papers(papers)
    return 0


def sync_rules_from_sqlite(db_path: str = "data/paperstream.db") -> int:
    """
    Sync all rule embeddings from SQLite to Qdrant.
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check schema - try both possible column names
    cursor.execute("PRAGMA table_info(rules)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Build query based on available columns
    if "rule_key" in columns:
        key_col = "rule_key"
    elif "rule_id" in columns:
        key_col = "rule_id"
    else:
        key_col = "id"
    
    desc_col = "description" if "description" in columns else "question"
    
    cursor.execute(f"""
        SELECT id, {key_col} as rule_key, {desc_col} as description, pos_embedding, neg_embedding 
        FROM rules 
        WHERE pos_embedding IS NOT NULL
    """)
    
    count = 0
    for row in cursor.fetchall():
        try:
            # Handle raw bytes vs base64 encoded
            pos_emb = row["pos_embedding"]
            neg_emb = row["neg_embedding"]
            
            # Try to decode as base64, otherwise treat as raw bytes
            try:
                if isinstance(pos_emb, str):
                    pos_list = decode_embedding(pos_emb)
                else:
                    pos_list = list(np.frombuffer(pos_emb, dtype=np.float32))
                
                if neg_emb:
                    if isinstance(neg_emb, str):
                        neg_list = decode_embedding(neg_emb)
                    else:
                        neg_list = list(np.frombuffer(neg_emb, dtype=np.float32))
                else:
                    neg_list = [0.0] * 768
            except Exception:
                continue
            
            upsert_rule_embeddings(
                rule_id=str(row["id"]),
                pos_embedding=pos_list,
                neg_embedding=neg_list,
                payload={
                    "rule_key": row["rule_key"],
                    "description": row["description"]
                }
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to sync rule {row['id']}: {e}")
    
    conn.close()
    
    logger.info(f"Synced {count} rules to Qdrant")
    return count


# =========================
# Health Check
# =========================

def health_check() -> Dict[str, Any]:
    """Check Qdrant connection and collection status"""
    try:
        client = get_qdrant_client()
        
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        result = {
            "status": "healthy",
            "host": f"{QDRANT_HOST}:{QDRANT_PORT}",
            "collections": collection_names
        }
        
        # Get collection stats
        for name in [PAPERS_COLLECTION, RULES_COLLECTION]:
            if name in collection_names:
                info = client.get_collection(name)
                result[f"{name}_count"] = info.points_count
        
        return result
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
