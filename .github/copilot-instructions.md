# Copilot Instructions for mcp-paperstream

## Project Overview

**mcp-paperstream** is a distributed paper validation system where:
1. **BioBERT** generates 768-dim embeddings for scientific papers
2. **Embeddings become visual voxel structures** (8×8×12 grids)
3. **Humans play a game** comparing paper voxels to rule voxels
4. **Consensus validation** aggregates player decisions

**Core insight:** Humans visually pattern-match instead of expensive LLM calls.

---

## DATAMODEL.md Contract (CRITICAL!)

The `DATAMODEL.md` file is the **single source of truth** shared between Python backend and Unity frontend.

### Hierarchy: Text → Embedding → Chunk → Voxels → Molecule

```
TEXT (Paper/Rule)
    ↓ BioBERT (768-dim)
EMBEDDING (float32 array)
    ↓ enhance_visual_contrast() + threshold
VOXEL GRID (8×8×12 = 768 positions)
    ↓ package with metadata
CHUNK (container: embedding + voxels + color + position)
    ↓ collect related chunks
MOLECULE (Paper = chain of section chunks, Rule = dipole of pos/neg chunks)
```

### Voxel Mapping Formula
```python
# Embedding index → 3D position
i = x + y*8 + z*64
# where x ∈ [0,7], y ∈ [0,7], z ∈ [0,11]

# Iteration order (MUST match C#!)
for z in range(12):
    for y in range(8):
        for x in range(8):
            i = x + y*8 + z*64
```

### Visual Contrast Enhancement
```python
def enhance_visual_contrast(embedding):
    centered = emb - mean(emb)
    amplified = tanh(centered * 2.0)
    return (amplified + 1.0) / 2.0
```

**This function MUST produce identical results in Python and C#!**

---

## Key Files

```
src/paperstream/
├── core/
│   └── data_model.py      # ⭐ DATAMODEL.md implementation
├── server.py              # MCP server with FastAPI
├── server_integrated.py   # Full server with all endpoints
├── processing/
│   └── pdf_voxelizer.py   # PDF → sections → embeddings → voxels
└── handlers/
    └── biobert_handler.py # BioBERT tokenization & embeddings

tests/
├── test_visual_embeddings.py  # Generates test data for Unity
└── test_data/                 # Sample JSON files
```

---

## API Endpoints (Game Integration)

### GET /api/rule/active
Returns the currently active Rule as a Molecule for display in Unity.

```json
{
  "rule": { /* Molecule object */ },
  "threshold": 0.7,
  "question": "Is this a Randomized Controlled Trial?"
}
```

### SSE /api/jobs/stream
Server-Sent Events stream delivering paper chunks for validation.

```
event: job
data: {"job_id": "...", "paper_id": "...", "chunk": {...}, "timeout_ms": 30000}
```

### POST /api/jobs/{job_id}/response
Player submits their decision.

```json
{
  "job_id": "abc123",
  "device_id": "player-device-id",
  "action": "collect",  // or "skip"
  "response_time_ms": 1234
}
```

---

## Game Flow

1. Server loads Rule → creates pos/neg Molecule with voxel grids
2. Unity displays Rule shapes as 3D reference
3. Papers get processed → sections become Chunks with voxels
4. Server streams paper Chunks to Unity as "jobs"
5. Player sees paper voxels, compares to Rule reference
6. Player collects (= YES vote) or skips (= NO vote)
7. Server aggregates consensus from multiple players
8. Paper gets classified when threshold reached

**NO client-side cosine similarity! Humans do the visual matching.**

---

## Development

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Test Data
```bash
PYTHONPATH=src python tests/test_visual_embeddings.py -o tests/test_data
# Add --no-biobert for mock embeddings (faster, no model download)
```

### Run Server
```bash
docker-compose up
# or directly:
uvicorn src.paperstream.server_integrated:app --host 0.0.0.0 --port 8089
```

---

## Code Conventions

- **Determinism is critical:** `embedding_to_voxel_grid()` must produce identical output every time
- **German comments OK:** Team is German-speaking
- **Type hints:** Use them for all public functions
- **Tests:** Run `pytest tests/` before committing

---

## Common Tasks

### Adding a new Rule type
1. Define positive/negative phrase lists
2. Generate embeddings with BioBERT
3. Create Molecule via `create_rule_molecule()`
4. Test voxel output visually in Unity

### Changing the voxel transformation
1. Update `enhance_visual_contrast()` in `data_model.py`
2. Update matching function in Unity `EmbeddingToVoxel.cs`
3. Verify with `test_visual_embeddings.py --no-biobert`
4. Compare voxel counts between Python and C#

### Debugging job flow
1. Check `/api/jobs/stream` SSE connection in browser
2. Monitor server logs for job assignment
3. Verify device registration at `/api/devices`
