# DATAMODEL.md — Shared Contract

**Version:** 0.1.0  
**Status:** Draft  
**Repos:** `mcp-paperstream` (Python) | `ValidationGame` (Unity/C#)

---

## Ziel

Spieler erkennen **visuell** ob ein Paper zu einer Rule matcht.  
Kein LLM-Aufruf pro Paper. Humans als Compute-Ressource.

---

## Hierarchie

```
TEXT ──────► EMBEDDING ──────► CHUNK ──────► VOXELS
             (768-dim)        (Container)   (8×8×12 Grid)
                                  │
                              MOLECULE
                         (verbundene Chunks)
```

---

## 1. Embedding

**Quelle:** BioBERT (`dmis-lab/biobert-base-cased-v1.2`)  
**Dimension:** 768 floats  
**Transport:** Base64-encoded float32 (`embedding_b64`)

```python
# Python: Encode
embedding_b64 = base64.b64encode(np.array(emb, dtype=np.float32).tobytes()).decode()

# C#: Decode
byte[] bytes = Convert.FromBase64String(embedding_b64);
float[] embedding = new float[768];
Buffer.BlockCopy(bytes, 0, embedding, 0, bytes.Length);
```

---

## 2. Voxel Grid

**Dimensionen:** 8 × 8 × 12 = 768 Voxels (= Embedding-Dimension)  
**Mapping:** `embedding[i]` → `voxel[x][y][z]` wobei `i = x + y*8 + z*64`

```
X: 0-7   (Breite, 8 Spalten)
Y: 0-7   (Höhe, 8 Schichten)  
Z: 0-11  (Tiefe, 12 Reihen)
```

### Embedding → Voxel Transformation

```python
def embedding_to_voxel_grid(embedding: np.ndarray, threshold: float = 0.3) -> List[Voxel]:
    """
    Deterministisch: Gleiches Embedding = Gleiches Grid
    
    1. Normalize to [0, 1]
    2. Reshape to 8×8×12
    3. Threshold: value >= threshold → Voxel aktiv
    """
    emb = np.array(embedding).flatten()[:768]
    
    # Normalize
    emb_min, emb_max = emb.min(), emb.max()
    normalized = (emb - emb_min) / (emb_max - emb_min + 1e-8)
    
    # Reshape: [768] → [8, 8, 12]
    grid = normalized.reshape((8, 8, 12))
    
    # Create voxels
    voxels = []
    for x in range(8):
        for y in range(8):
            for z in range(12):
                value = grid[x, y, z]
                if value >= threshold:
                    voxels.append(Voxel(x, y, z, value))
    
    return voxels
```

### Visuelle Unterscheidbarkeit

**Problem:** Raw Embeddings sehen oft ähnlich aus.  
**Lösung:** Transformation die semantische Unterschiede verstärkt.

```python
def enhance_visual_contrast(embedding: np.ndarray) -> np.ndarray:
    """
    Verstärkt Unterschiede für bessere visuelle Erkennbarkeit.
    """
    emb = np.array(embedding)
    
    # 1. Center around mean
    centered = emb - emb.mean()
    
    # 2. Amplify differences (sigmoid-like)
    amplified = np.tanh(centered * 2.0)
    
    # 3. Rescale to [0, 1]
    result = (amplified + 1) / 2
    
    return result
```

---

## 3. Chunk

Ein Chunk ist ein **Container** für ein Embedding + seine Voxels.

```json
{
  "chunk_id": 0,
  "chunk_type": "abstract",           // oder "positive", "negative", etc.
  "text_preview": "First 500 chars...",
  "embedding_b64": "base64...",
  "voxels": {
    "grid_size": [8, 8, 12],
    "voxels": [[0,0,0,0.85], [0,0,1,0.72], ...],  // [x,y,z,value]
    "voxel_count": 312
  },
  "color": {"r": 0.2, "g": 0.6, "b": 0.9},
  "position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "connects_to": [1]
}
```

### Unity Rendering

```
Chunk (invisible GameObject)
  └─ position = world position
  └─ Voxels[] (visible cubes)
       └─ localPosition = voxel.x, voxel.y, voxel.z
       └─ color = chunk.color * voxel.value (intensity)
```

---

## 4. Molecule

**Paper Molecule:** Chain von Section-Chunks  
**Rule Molecule:** Dipole von Pos/Neg-Chunks

### Paper Molecule

```json
{
  "molecule_id": "paper_12345",
  "molecule_type": "paper",
  "title": "Effectiveness of...",
  "layout": "chain",
  "connection_type": "sequential",
  "chunks": [
    {"chunk_id": 0, "chunk_type": "abstract", ...},
    {"chunk_id": 1, "chunk_type": "methods", ...},
    {"chunk_id": 2, "chunk_type": "results", ...}
  ]
}
```

```
[Abstract] ──wire── [Methods] ──wire── [Results]
  Chunk 0            Chunk 1            Chunk 2
```

### Rule Molecule

```json
{
  "molecule_id": "rule_rct",
  "molecule_type": "rule",
  "title": "Is this a Randomized Controlled Trial?",
  "layout": "dipole",
  "connection_type": "polar",
  "chunks": [
    {"chunk_id": 0, "chunk_type": "positive", ...},
    {"chunk_id": 1, "chunk_type": "negative", ...}
  ]
}
```

```
[Positive] ←──polar──→ [Negative]
  Chunk 0                Chunk 1
   grün                   rot
```

---

## 5. Farben

| chunk_type   | RGB              | Hex      |
|--------------|------------------|----------|
| abstract     | (0.2, 0.6, 0.9)  | #3399E6  |
| introduction | (0.3, 0.8, 0.3)  | #4DCC4D  |
| methods      | (0.9, 0.7, 0.2)  | #E6B233  |
| results      | (0.8, 0.3, 0.3)  | #CC4D4D  |
| discussion   | (0.7, 0.4, 0.9)  | #B266E6  |
| positive     | (0.2, 0.9, 0.3)  | #33E64D  |
| negative     | (0.9, 0.2, 0.2)  | #E63333  |

---

## 6. Game Flow

### Setup
```
1. Server sendet aktive Rule (Molecule mit pos/neg Chunks)
2. Unity zeigt Rule-Shapes als Referenz (oben im UI)
3. Server streamt Paper-Jobs via SSE
```

### Gameplay
```
4. Paper spawnt als Voxel-Shape (1 Chunk = Abstract oder relevante Section)
5. Spieler sieht: "Sieht das aus wie die grüne Referenz?"
6. Spieler sammelt ein (→ implizites JA) oder ignoriert (→ implizites NEIN)
```

### Validation
```
7. Unity sendet: { job_id, action: "collect" | "skip", time_ms }
8. Server aggregiert: Mehrere Spieler → Consensus
9. Consensus erreicht → Paper wird klassifiziert
```

---

## 7. API Endpoints

### GET /api/rule/active
```json
{
  "rule": { /* Rule Molecule */ },
  "threshold": 0.7,
  "question": "Is this a Randomized Controlled Trial?"
}
```

### SSE /api/jobs/stream
```json
{
  "event": "job",
  "data": {
    "job_id": "job_abc123",
    "paper_id": "paper_12345",
    "chunk": { /* Single Chunk to validate */ },
    "timeout_ms": 10000
  }
}
```

### POST /api/jobs/{job_id}/response
```json
{
  "device_id": "unity_client_xyz",
  "action": "collect",        // or "skip"
  "response_time_ms": 2340
}
```

---

## 8. Invarianten

1. **Determinismus:** `embedding_to_voxel(E)` gibt immer gleiches Ergebnis
2. **Symmetrie:** Python und C# produzieren identische Voxel-Grids
3. **Threshold:** Standardwert 0.3 — Server kann pro Rule anpassen
4. **Chunk-Größe:** Immer 8×8×12, nie anders

---

## 9. TODO

- [ ] `enhance_visual_contrast()` Parameter tunen
- [ ] Unity: `EmbeddingToVoxel.cs` an dieses Modell anpassen
- [ ] Python: `data_model.py` Threshold konfigurierbar machen
- [ ] Wire-Rendering in Unity (Chunk-Verbindungen)
- [ ] Test: Verschiedene Paper-Typen visuell unterscheidbar?

---

## Changelog

- **0.1.0** (2026-02-02): Initial draft
