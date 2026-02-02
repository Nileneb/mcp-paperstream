# mcp-paperstream

Distributed scientific paper validation through visual voxel comparison.

**Core idea:** BioBERT embeddings (768-dim) become 8×8×12 voxel grids. Humans visually compare paper shapes to rule shapes — no LLM calls needed for classification.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PAPER VALIDATION FLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. INPUT (n8n / MCP Client)                                         │
│     └── Submit Paper → Process PDF → Extract Sections                │
│                                                                      │
│  2. EMBEDDING (BioBERT)                                              │
│     └── Section Text → 768-dim Embedding → Base64                    │
│                                                                      │
│  3. VOXELIZATION                                                     │
│     └── Embedding → enhance_visual_contrast() → 8×8×12 Grid          │
│                                                                      │
│  4. DISTRIBUTION                                                     │
│     └── Jobs via SSE → Unity Game / Agent Swarm                      │
│                                                                      │
│  5. VALIDATION                                                       │
│     └── Human/Agent compares → Collect/Skip → Vote                   │
│                                                                      │
│  6. CONSENSUS                                                        │
│     └── Aggregate votes → Classification → Done                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/Nileneb/mcp-paperstream.git
cd mcp-paperstream
python -m venv .venv && source .venv/bin/activate

# Install
pip install -r requirements.txt

# Run
docker-compose up
# or: uvicorn src.paperstream.server_integrated:app --host 0.0.0.0 --port 8089
```

---

## API Endpoints

### MCP Integration (n8n)
| Endpoint | Description |
|----------|-------------|
| `GET /sse` | SSE transport for MCP client |
| `POST /messages` | MCP JSON-RPC messages |

### Game Integration (Unity)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/rule/active` | Current rule as Molecule |
| `SSE` | `/api/jobs/stream` | Paper chunks for validation |
| `POST` | `/api/jobs/{id}/response` | Player action (collect/skip) |

### Paper Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/papers/submit` | Submit new paper |
| `GET` | `/api/papers/{id}` | Paper status |
| `POST` | `/api/papers/{id}/process` | Trigger processing |

### Rules
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rules/create` | Create validation rule |
| `GET` | `/api/rules` | List all rules |
| `POST` | `/api/rules/load-defaults` | Load default rules |

---

## MCP Tools

For n8n or any MCP client:

| Tool | Description |
|------|-------------|
| `submit_paper` | Submit paper for validation |
| `create_rule` | Create rule with pos/neg phrases |
| `process_paper` | Process paper (PDF → Sections → Voxels) |
| `get_paper_status` | Check validation progress |
| `get_system_stats` | Server statistics |

### Example: Create Rule
```json
{
  "name": "create_rule",
  "arguments": {
    "rule_id": "is_rct",
    "question": "Is this a randomized controlled trial?",
    "positive_phrases": ["randomized controlled trial", "RCT", "double-blind"],
    "negative_phrases": ["systematic review", "meta-analysis", "case report"],
    "threshold": 0.7
  }
}
```

---

## DATAMODEL.md Contract

The `DATAMODEL.md` file defines the shared contract with Unity.

### Voxel Transformation
```python
# Embedding → Voxel position
i = x + y*8 + z*64  # where x,y ∈ [0,7], z ∈ [0,11]

# Visual contrast enhancement (MUST match C#!)
def enhance_visual_contrast(embedding):
    centered = emb - mean(emb)
    amplified = tanh(centered * 2.0)
    return (amplified + 1.0) / 2.0
```

### Data Hierarchy
```
Molecule (Paper or Rule)
├── molecule_id, molecule_type, title
└── chunks[]
    ├── chunk_id, chunk_type
    ├── embedding_b64 (768 floats, base64)
    ├── voxels {grid_size, voxels[[x,y,z,v],...]}
    ├── color {r,g,b}
    └── position {x,y,z}
```

---

## Project Structure

```
src/paperstream/
├── server_integrated.py    # Main server (REST + MCP + SSE)
├── core/
│   └── data_model.py       # ⭐ DATAMODEL.md implementation
├── api/
│   ├── paper_handler.py
│   ├── rule_handler.py
│   ├── job_handler.py
│   └── sse_stream.py
├── pipeline/
│   ├── paper_processor.py  # PDF → Sections → Voxels
│   └── consensus_engine.py
├── handlers/
│   └── biobert_handler.py  # BioBERT embeddings
└── db/
    └── database.py         # SQLite

tests/
├── test_visual_embeddings.py  # Generate test data
└── test_data/                 # Sample JSON for Unity
```

---

## Configuration

`src/paperstream/config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8089

models:
  biobert:
    name: "dmis-lab/biobert-base-cased-v1.2"

consensus:
  min_votes: 3
  agreement_threshold: 0.6

voxel:
  threshold: 0.3
  enhance_contrast: true
```

---

## Generate Test Data

```bash
# With BioBERT (downloads ~400MB model)
PYTHONPATH=src python tests/test_visual_embeddings.py -o tests/test_data

# Mock embeddings (faster, no download)
PYTHONPATH=src python tests/test_visual_embeddings.py --no-biobert -o tests/test_data
```

Output goes to `tests/test_data/` and can be copied to Unity's `StreamingAssets/TestData/`.

---

## Docker

```bash
docker-compose up -d
```

Services:
- `paperstream`: Main server on port 8089
- `db`: SQLite volume for persistence

---

## Links

- **Unity Game:** [ValidationGame](https://github.com/Nileneb/ValidationGame)
- **Contract:** See `DATAMODEL.md` in repo root
