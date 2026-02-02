# Copilot Instructions for mcp-paperstream

## Project Overview
**mcp-paperstream** is a distributed MCP server for scientific paper review combining:
1. **BERTScore computation** distributed across IoT edge devices (ESP32, RPi, smartphones)
2. **Stable Diffusion integration** for scientific visualizations
3. **BiomedCLIP validation** for text-image semantic alignment


# Core Data Model (CRITICAL!)

## Hierarchy: Text → Embedding → Chunk → Voxels

```
┌─────────────────────────────────────────────────────────────────┐
│  TEXT (Papers & Rules)                                          │
│  - Papers: PDF sections (abstract, methods, results, etc.)      │
│  - Rules: Positive/negative phrase lists                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ BioBERT (768-dim)
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDING (768 dimensions)                                      │
│  - Semantic vector representation                                │
│  - Base64 encoded for transport: embedding_b64                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ Container
┌─────────────────────────────────────────────────────────────────┐
│  CHUNK (Unity: INVISIBLE cube at position)                       │
│  - Container for embedding + voxels                              │
│  - Paper: N chunks (one per section)                             │
│  - Rule: 2 chunks (positive=green, negative=red)                 │
│  - Position defines local origin (0,0,0) for voxels inside       │
│  - connects_to: Array of chunk_ids for wire/lane connections     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ Reshape 768 → 8x8x12
┌─────────────────────────────────────────────────────────────────┐
│  VOXELS (Unity: VISIBLE cubes for interaction)                   │
│  - 8x8x12 grid = 768 values (matches embedding dim)              │
│  - Each voxel = one embedding dimension visualized               │
│  - Rendered relative to chunk's position                         │
│  - voxels: {grid_size: [8,8,12], voxels: [[x,y,z,value],...]}   │
└─────────────────────────────────────────────────────────────────┘
```

## Unified Chunk Structure (BOTH Papers AND Rules)
```json
{
  "chunk_id": 0,
  "chunk_type": "abstract" | "methods" | "positive" | "negative",
  "text_preview": "First 500 chars...",
  "embedding_b64": "base64-encoded-768-float32",
  "voxels": {
    "grid_size": [8, 8, 12],
    "voxels": [[x, y, z, value], ...],
    "voxel_count": 384,
    "fill_ratio": 0.5
  },
  "color": {"r": 0.2, "g": 0.9, "b": 0.3},
  "position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "connects_to": [1, 2]
}
```

## Wires/Lanes (WICHTIG!)
**Wires verbinden VOXELS innerhalb eines Chunks - NICHT Chunks untereinander!**

- Wires = Linien zwischen Embedding-Cubes (Voxels) innerhalb eines Chunks
- Zeigen Zusammenhang der 768 Embedding-Dimensionen
- Helfen beim visuellen Zuordnen welche Cubes zum gleichen Embedding gehören
- Ein Chunk kann mehrere Embeddings enthalten → Wires gruppieren zugehörige Voxels

```
Chunk (unsichtbar)
├── Voxel[0,0,0] ──Wire── Voxel[1,0,0] ──Wire── Voxel[2,0,0]
├── Voxel[0,1,0] ──Wire── Voxel[1,1,0] ──Wire── Voxel[2,1,0]
└── ... (8x8x12 Voxels verbunden durch Wires)
```

## Unity Rendering Flow
1. **Spawn Chunk** = invisible cube at `position` (container)
2. **Spawn Voxels** = visible cubes at `chunk.position + voxel[x,y,z]`
3. **Draw Wires** = LineRenderer zwischen Voxels INNERHALB des Chunks (zeigen Embedding-Struktur)


## Architecture Essentials

┌─────────────────────────────────────────────────────────────────────────┐
│                         PAPER-VALIDATION-PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. N8N-AGENT (paper-search-mcp)                                         │
│     └── Sucht Paper → Lädt PDF → Speichert in /shared/papers             │
│                                                                          │
│  2. PAPERSTREAM-MCP (Docker-Server)                                      │
│     ├── Erstellt ERST alle Rule-Embeddings (= Preview-Container)         │
│     ├── Normalisiert PDF → Extrahiert Sections                          │
│     ├── Erzeugt Paper-Embeddings (= Job-Embeddings)                     │
│     ├── Mappt Embeddings auf 8×8×12-Voxel-Grid                          │
│     └── Sendet Jobs (Embeddings + Rules) an Android-Devices             │
│                                                                          │
│  3. ANDROID-DEVICES                                                      │
│     ├── Empfangen: Paper-Embeddings (Base64) + Rule-Embeddings (Base64)  │
│     ├── Vergleichen per Cosine-Similarity                               │
│     └── Senden zurück: "Paper X enthält Rule 1,3,5 in Section Y,Z"      │
│                                                                          │
│  4. UNITY-GAME                                                           │
│     ├── Spawnt Voxel-Figuren aus validiertem Embedding-Grid             │
│     ├── RulePreview zeigt Ziel-Figur als 3D-Referenz                    │
│     └── Spieler sammelt passende Voxels → Punkte                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘



WICHTIG: der server läuft mit Docker compose !!!!!!
Ablauf pro Paper

    load_default_rules() → Lädt 17 vordefinierte Rule-Embeddings
    search_*() + download_*() → Paper finden und PDF speichern
    submit_paper() + link_paper_pdf() → Paper registrieren
    process_paper() → PDF→Sections→Embeddings→Voxels→Jobs
    Android empfängt Jobs → Vergleicht Paper- mit Rule-Embeddings
    Android sendet Ergebnis → "Rule X gefunden in Section Y"
    Unity spawnt validierte Figuren → Spieler interagiert



```
src/paperstream/
├── server.py           # MCP server - task distribution, SSE, job management
├── config.yaml         # Central configuration (paths, ports, IoT settings)
├── handlers/
│   ├── biobert_handler.py    # BioBERT tokenization & embeddings
│   ├── biomedclip_handler.py # Text-image similarity (requires open_clip)
│   ├── sd_api_client.py      # AUTOMATIC1111 SD WebUI API client
│   └── download_model.py     # Model download utility
└── prompts/
    ├── scientific_templates.py # SD prompt templates (cell_diagram, molecular, etc.)
    └── term_mappings.json      # Scientific terms → visual descriptors
```

### Coordinator-Worker Pattern
The server uses SSE for downstream task distribution and REST for result submission:
- **Coordinator** (`server.py`): Manages jobs, distributes tasks, aggregates embeddings
- **IoT Workers**: Compute TinyBERT layer subsets based on device capability
- Task flow: `queued → assigned → (timeout → retry) | done`

### Device Capability Mapping
```python
LOW    (ESP32):     [0]        # Only embedding layer
MEDIUM (RPi4):      [0,1,2]    # First 3 layers
HIGH   (modern):    [0-5]      # All 6 TinyBERT layers
```

## Configuration
All settings in `config.yaml`. Key sections:
- `server`: Host, port, SSE/result paths
- `models.biobert`: Model path and HuggingFace name
- `models.biomedclip`: BiomedCLIP config (optional)

- `iot`: TTL, inflight limits, layer count

Environment variables override config: `FASTMCP_HOST`, `FASTMCP_PORT`, `BERTSCORE_HMAC`, etc.

## Handler Usage Patterns

### BioBERT Handler
```python
from paperstream.handlers import get_biobert_handler
handler = get_biobert_handler()
tokens, token_ids = handler.tokenize("scientific text")
embedding = handler.embed("text", layer_range=(0, 3))  # Partial layers for IoT
```



## Development Workflow

### Setup
```bash
uv pip install transformers torch open_clip_torch pillow httpx pyyaml
python -m paperstream.handlers.download_model all  # Download models
```

### Run Server
```bash
uvicorn src.paperstream.server:mcp --host 0.0.0.0 --port 8089
```

### Test Flow
1. `register_iot_client(client_id="test", capability="high")`
2. `bertscore_compute(reference="...", candidate="...", distributed=True)`
3. `bertscore_status(job_id)` - poll until completed


