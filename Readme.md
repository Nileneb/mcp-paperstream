# 🧬 MCP-PaperStream

**Distributed MCP Server for Scientific Paper Validation** - A gamified platform where Android devices validate scientific papers through crowdsourced BERTScore computation.

## 🎯 Core Concept

```
n8n Workflow → Submit Papers → MCP Server → Create Jobs
                                    ↓
              Android Devices ← Fetch Jobs ← SQLite DB
                    ↓
            Validate Sections → Submit Results → Consensus
                                                    ↓
                              Unity Game ← SSE Updates ← Leaderboard
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PaperStream MCP Server v1.0                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  REST API   │    │   MCP API   │    │  Unity SSE  │              │
│  │  /api/*     │    │   /mcp      │    │  /stream    │              │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘              │
│         │                  │                   │                     │
│  ┌──────┴──────────────────┴───────────────────┴──────┐             │
│  │                   Starlette App                     │             │
│  └─────────────────────────┬───────────────────────────┘             │
│                            │                                         │
│  ┌─────────────────────────┴───────────────────────────┐             │
│  │                     Handlers                         │             │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │             │
│  │  │  Paper   │  │   Rule   │  │   Job    │  │Device│ │             │
│  │  │ Handler  │  │ Handler  │  │ Handler  │  │Handler│             │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬───┘ │             │
│  └───────┼─────────────┼─────────────┼────────────┼─────┘             │
│          │             │             │            │                   │
│  ┌───────┴─────────────┴─────────────┴────────────┴─────┐             │
│  │                    SQLite Database                    │             │
│  │  papers | rules | jobs | devices | results | consensus│             │
│  └───────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐          │
│  │      BioBERT Handler    │    │   Paper Processor       │          │
│  │   (Embeddings, 768-dim) │    │ (PDF → Sections → Voxels)│          │
│  └─────────────────────────┘    └─────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- ~2GB disk space for models
- (Optional) AUTOMATIC1111 Stable Diffusion WebUI

### Installation

```bash
# Clone repository
git clone https://github.com/Nileneb/mcp-paperstream.git
cd mcp-paperstream

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers fastmcp pyyaml httpx pillow python-dotenv aiohttp uvicorn starlette requests

# Optional: PDF processing
pip install PyMuPDF

# Download models (auto-downloads from HuggingFace on first use)
python -m src.paperstream.handlers.download_model all
```

### Start Server

```bash
# Integrated server (recommended)
./start_server.sh

# Or manually:
python -m uvicorn src.paperstream.server_integrated:app --host 0.0.0.0 --port 8089

# MCP-only mode (BERTScore IoT):
./start_server.sh mcp
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8089/health

# Submit paper (REST API)
curl -X POST http://localhost:8089/api/papers/submit \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "PMC12345", "title": "My Paper", "priority": 7}'

# Get stats
curl http://localhost:8089/api/stats
```

---

## 🔌 n8n MCP Integration

The server exposes MCP tools via **SSE Transport** for n8n integration.

### n8n Configuration

1. Add **MCP Client** node in n8n
2. Set URL: `http://YOUR_IP:8089/sse`
3. Transport: **SSE (Server-Sent Events)**

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `submit_paper` | Submit new paper for processing |
| `create_rule` | Create validation rule with BioBERT embeddings |
| `process_paper` | Process paper (extract sections, embeddings) |
| `get_paper_status` | Get paper validation status |
| `get_leaderboard` | Get gamification leaderboard |
| `get_system_stats` | Get system statistics |
| `load_default_rules` | Load default validation rules |

### Example: Submit Paper via MCP

```json
{
  "name": "submit_paper",
  "arguments": {
    "paper_id": "PMC12345",
    "title": "My Scientific Paper",
    "pdf_url": "https://example.com/paper.pdf",
    "priority": 8,
    "source": "n8n"
  }
}
```

### Example: Create Rule via MCP

```json
{
  "name": "create_rule",
  "arguments": {
    "rule_id": "is_rct",
    "question": "Is this a randomized controlled trial?",
    "positive_phrases": ["randomized controlled trial", "RCT", "clinical trial"],
    "negative_phrases": ["review", "meta-analysis"],
    "threshold": 0.75
  }
}
```

---

## 📁 Project Structure

```
src/paperstream/
├── server_integrated.py   # Main server (REST + MCP + SSE)
├── server.py              # MCP-only server (BERTScore IoT)
├── config.yaml            # Central configuration
│
├── db/                    # Database Layer
│   ├── database.py        # SQLite manager
│   ├── models.py          # Dataclass models
│   └── migrations/        # SQL schema
│
├── api/                   # API Handlers
│   ├── paper_handler.py   # Paper CRUD
│   ├── rule_handler.py    # Rule management + BioBERT
│   ├── job_handler.py     # Job distribution
│   ├── device_handler.py  # Device registration
│   └── sse_stream.py      # Unity SSE stream
│
├── pipeline/              # Processing Pipeline
│   ├── paper_processor.py # PDF → Sections → Voxels
│   └── consensus_engine.py# Result aggregation
│
├── handlers/              # ML Handlers
│   ├── biobert_handler.py # BioBERT embeddings
│   ├── sd_api_client.py   # Stable Diffusion API
│   └── biomedclip_handler.py # BiomedCLIP (optional)
│
└── prompts/               # Prompt Templates
    ├── scientific_templates.py
    └── term_mappings.json
```

---

## 🔌 API Endpoints

### MCP (n8n Integration)
| Endpoint | Description |
|----------|-------------|
| GET `/sse` | SSE stream for MCP client connection |
| POST `/messages` | MCP message endpoint (JSON-RPC) |
| Mount `/mcp` | Streamable HTTP for MCP Inspector |

### Papers (REST API)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/papers/submit` | Submit new paper |
| GET | `/api/papers` | List all papers |
| GET | `/api/papers/{id}` | Get paper details |

### Rules
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rules/create` | Create validation rule |
| GET | `/api/rules` | List active rules |

### Jobs (Android)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs/next?device_id=xxx` | Get next jobs |
| POST | `/api/validation/submit` | Submit results |

### Devices
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/devices/register` | Register device |
| GET | `/api/devices/{id}` | Get device info |

### Unity
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stream/unity` | SSE stream |
| GET | `/api/consensus/{paper_id}` | Validation status |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/stats` | System statistics |

See [docs/API.md](docs/API.md) for full documentation.

---

## 🎮 Gamification

### Point System
- Base: 10 points per validation
- Similarity bonus: up to 40 points
- Confidence bonus: up to 30 points
- Match found: +20 points

### Leaderboard
- Tracks total points, papers validated, matches found
- Real-time updates via SSE to Unity

---

## 🔧 Configuration

Edit `src/paperstream/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8089

models:
  biobert:
    name: "nlpie/distil-biobert"
    path: "./src/paperstream/models/biobert"

consensus:
  min_votes: 3
  agreement_threshold: 0.6

jobs:
  ttl_seconds: 300
  max_per_device: 5
```

---

## 📋 Data Flow

1. **n8n submits paper** → `POST /api/papers/submit`
2. **Paper processing** → Extract sections, generate embeddings, create voxels
3. **Job creation** → One job per (paper × section × rule)
4. **Android fetches jobs** → `GET /api/jobs/next`
5. **Android validates** → Compare embeddings locally
6. **Submit results** → `POST /api/validation/submit`
7. **Consensus calculation** → Majority vote after 3+ submissions
8. **Unity notified** → SSE event: `paper_validated`

---

## 🧪 Testing

```bash
# Test all modules
python -c "
from src.paperstream.db import get_db
from src.paperstream.api import get_paper_handler, get_rule_handler
from src.paperstream.pipeline import get_consensus_engine
print('All modules OK')
"

# Initialize database
python -c "
from src.paperstream.db import get_db
db = get_db()
db.initialize()
print(f'DB: {db.db_path}')
"

# Load default rules
curl http://localhost:8089/api/stats
```

---

## 📄 License

MIT License - see LICENSE file.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

---

## 📞 Support

Open an issue on GitHub for bugs or feature requests.
