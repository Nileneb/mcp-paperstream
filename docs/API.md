# PaperStream API Documentation

## Base URL
```
http://localhost:8089
```

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    n8n      │────▶│ PaperStream │◀────│   Unity     │
│  (MCP/REST) │     │   Server    │     │   (SSE)     │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │ SQLite  │ │ Qdrant  │ │ BioBERT │
        │   DB    │ │ Vector  │ │ (768d)  │
        └─────────┘ └─────────┘ └─────────┘
```

---

## Endpoints Overview

### MCP (n8n Integration)
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/sse\` | GET | SSE stream for MCP client |
| \`/messages\` | POST | MCP JSON-RPC messages |
| \`/mcp\` | * | Streamable HTTP endpoint |

### Papers API
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/papers/submit\` | POST | Submit new paper |
| \`/api/papers\` | GET | List all papers |
| \`/api/papers/{paper_id}\` | GET | Get paper details |
| \`/api/papers/{paper_id}/chunks\` | GET | Get paper as molecule chunks |
| \`/api/chunks?paper_id=...\` | GET | Alternative chunk endpoint (for DOIs) |

### Rules API
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/rules/create\` | POST | Create validation rule |
| \`/api/rules/chunks\` | GET | Get ALL rules as molecule chunks |
| \`/api/rules/active\` | GET | Get all rules with embeddings |
| \`/api/rules/{rule_id}/chunks\` | GET | Get single rule as chunks |
| \`/api/rules\` | GET | List all active rules |
| \`/api/rule-chunks?rule_id=...\` | GET | Alternative chunk endpoint |

### Jobs API (Android/Unity) - Round-Robin
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/jobs/stats\` | GET | Job queue statistics |
| \`/api/jobs/next\` | GET | Get next job (round-robin) |
| \`/api/jobs/submit\` | POST | Submit validation results |
| \`/api/validation/submit\` | POST | Alias for submit |

### Qdrant Vector Store
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/qdrant/status\` | GET | Qdrant connection status |
| \`/api/qdrant/sync\` | POST | Sync SQLite → Qdrant |
| \`/api/qdrant/search/papers\` | POST | Semantic paper search |
| \`/api/qdrant/match/rules\` | POST | Match embedding to rules |

### Other
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/health\` | GET | Health check |
| \`/api/stats\` | GET | System statistics |
| \`/api/devices/register\` | POST | Register device |
| \`/api/devices/{device_id}\` | GET | Get device info |
| \`/api/consensus/{paper_id}\` | GET | Get validation status |
| \`/api/stream/unity\` | GET | SSE stream for Unity |

---

## Papers API

### Submit Paper
\`\`\`http
POST /api/papers/submit
Content-Type: application/json

{
  "paper_id": "PMC12345",
  "title": "Randomized Trial of...",
  "pdf_url": "https://...",
  "priority": 7,
  "source": "n8n"
}
\`\`\`

**Response:**
\`\`\`json
{
  "status": "accepted",
  "paper_id": "PMC12345",
  "message": "Paper queued for processing"
}
\`\`\`

### List Papers
\`\`\`http
GET /api/papers?status=pending&limit=100
\`\`\`

### Get Paper Chunks (Molecule Visualization)
\`\`\`http
GET /api/papers/{paper_id}/chunks
GET /api/chunks?paper_id=DOI:10.1234/example  # For DOIs with special chars
\`\`\`

**Response:**
\`\`\`json
{
  "paper_id": "2601.16040v1",
  "title": "Can Platform Design Encourage Curiosity?",
  "chunks": [
    {
      "chunk_id": 0,
      "section_name": "abstract",
      "text_preview": "First 500 chars of abstract...",
      "embedding_b64": "base64-768-dim-float32",
      "color": {"r": 0.2, "g": 0.6, "b": 0.9},
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "connects_to": [1]
    },
    {
      "chunk_id": 1,
      "section_name": "introduction",
      "text_preview": "First 500 chars of intro...",
      "embedding_b64": "...",
      "color": {"r": 0.3, "g": 0.8, "b": 0.3},
      "position": {"x": 2.0, "y": 0.0, "z": 0.0},
      "connects_to": [2]
    }
  ],
  "chunks_count": 5,
  "molecule_config": {
    "embedding_dim": 768,
    "layout": "chain",
    "scale": 1.0,
    "connection_type": "sequential"
  }
}
\`\`\`

**Section Colors:**
| Section | Color (RGB) |
|---------|-------------|
| abstract | (0.2, 0.6, 0.9) Blue |
| introduction | (0.3, 0.8, 0.3) Green |
| methods | (0.9, 0.7, 0.2) Yellow |
| results | (0.8, 0.3, 0.3) Red |
| discussion | (0.7, 0.4, 0.9) Purple |
| conclusion | (0.5, 0.5, 0.5) Gray |

---

## Rules API

### Create Rule
\`\`\`http
POST /api/rules/create
Content-Type: application/json

{
  "rule_id": "is_rct",
  "question": "Is this a Randomized Controlled Trial?",
  "positive_phrases": ["randomized controlled trial", "RCT", "random allocation"],
  "negative_phrases": ["observational", "retrospective", "case series"],
  "threshold": 0.75
}
\`\`\`

**Response:**
\`\`\`json
{
  "status": "created",
  "rule_id": "is_rct",
  "message": "Rule created with 3 positive phrases",
  "embedding_dim": 768,
  "qdrant_synced": true
}
\`\`\`

### Get All Rules with Embeddings
\`\`\`http
GET /api/rules/active
\`\`\`

**Response:**
\`\`\`json
{
  "rules": [
    {
      "rule_id": "is_rct",
      "question": "Is this a Randomized Controlled Trial?",
      "threshold": 0.7,
      "is_active": true,
      "pos_embedding_b64": "base64-768-dim",
      "neg_embedding_b64": "base64-768-dim"
    }
  ],
  "count": 8
}
\`\`\`

### Get Rule Chunks (Dipole Molecule)
\`\`\`http
GET /api/rules/{rule_id}/chunks
GET /api/rules/chunks        # ALL rules as chunks
GET /api/rule-chunks?rule_id=is_rct
\`\`\`

**Response:**
\`\`\`json
{
  "rule_id": "is_rct",
  "question": "Is this a Randomized Controlled Trial?",
  "threshold": 0.7,
  "chunks": [
    {
      "chunk_id": 0,
      "chunk_type": "positive",
      "text_preview": "randomized controlled trial, RCT, random allocation...",
      "embedding_b64": "base64-768-dim",
      "color": {"r": 0.2, "g": 0.9, "b": 0.3},
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "connects_to": [1]
    },
    {
      "chunk_id": 1,
      "chunk_type": "negative",
      "text_preview": "observational, retrospective, case series...",
      "embedding_b64": "base64-768-dim",
      "color": {"r": 0.9, "g": 0.2, "b": 0.2},
      "position": {"x": 2.0, "y": 0.0, "z": 0.0},
      "connects_to": []
    }
  ],
  "chunks_count": 2,
  "molecule_config": {
    "embedding_dim": 768,
    "layout": "dipole",
    "scale": 1.0,
    "connection_type": "polar"
  }
}
\`\`\`

**Rule Molecule Structure:**
\`\`\`
┌─────────────┐         ┌─────────────┐
│   POSITIVE  │ ──────► │  NEGATIVE   │
│   (Green)   │         │   (Red)     │
└─────────────┘         └─────────────┘
\`\`\`

---

## Jobs API (Round-Robin)

### Job Distribution Principle

**Round-Robin ohne Blockierung:**
- Multiple devices can validate the same paper simultaneously
- All validations are documented for consensus building
- Papers with fewest validations get priority

### Get Job Queue Stats
\`\`\`http
GET /api/jobs/stats
\`\`\`

**Response:**
\`\`\`json
{
  "paper_status": {
    "pending": 14,
    "ready": 8,
    "failed": 1
  },
  "ready_with_embeddings": 8,
  "total_validations": 7,
  "active_assignments": 2,
  "validation_distribution": {
    "PMC12345": {"title": "...", "validations": 3},
    "PMC67890": {"title": "...", "validations": 1}
  },
  "round_robin_enabled": true
}
\`\`\`

### Get Next Job
\`\`\`http
GET /api/jobs/next?device_id=android_abc123
\`\`\`

**Response:**
\`\`\`json
{
  "status": "assigned",
  "job": {
    "paper_id": "PMC12345",
    "title": "A Randomized Controlled Trial of...",
    "authors": "Smith J, Jones M",
    "journal": "Nature Medicine",
    "paper_embedding_b64": "base64-encoded-768-dim-float32-array",
    "paper_text": "Abstract text for keyword matching...",
    "chunks": [
      {
        "chunk_id": 0,
        "section_name": "abstract",
        "text_preview": "...",
        "embedding_b64": "...",
        "color": {"r": 0.2, "g": 0.6, "b": 0.9},
        "position": {"x": 0, "y": 0, "z": 0},
        "connects_to": [1]
      }
    ],
    "chunks_count": 5,
    "expires_in_seconds": 600
  },
  "device_id": "android_abc123"
}
\`\`\`

**Priority Logic:**
1. Papers this device hasn't validated yet
2. Papers with fewest total validations (fairness)
3. By priority/age

### Submit Validation Results
\`\`\`http
POST /api/jobs/submit
POST /api/validation/submit  # Alias
Content-Type: application/json

{
  "device_id": "android_abc123",
  "paper_id": "PMC12345",
  "results": [
    {
      "rule_id": "is_rct",
      "matched": true,
      "confidence": 0.89,
      "regions": [[100, 200, 300, 400]]
    },
    {
      "rule_id": "has_placebo",
      "matched": false,
      "confidence": 0.45,
      "regions": []
    }
  ]
}
\`\`\`

**Response:**
\`\`\`json
{
  "accepted": true,
  "paper_id": "PMC12345",
  "rules_matched": 1,
  "rules_checked": 2,
  "points_earned": 14
}
\`\`\`

---

## Qdrant Vector Store API

### Check Status
\`\`\`http
GET /api/qdrant/status
\`\`\`

**Response:**
\`\`\`json
{
  "status": "connected",
  "collections": {
    "papers": {"vectors_count": 8, "status": "green"},
    "rules": {"vectors_count": 16, "status": "green"}
  }
}
\`\`\`

### Sync SQLite → Qdrant
\`\`\`http
POST /api/qdrant/sync
\`\`\`

**Response:**
\`\`\`json
{
  "status": "synced",
  "papers_synced": 8,
  "rules_synced": 8
}
\`\`\`

### Semantic Paper Search
\`\`\`http
POST /api/qdrant/search/papers
Content-Type: application/json

{
  "embedding": [0.1, 0.2, ...],
  "limit": 10
}
\`\`\`

### Match Embedding to Rules
\`\`\`http
POST /api/qdrant/match/rules
Content-Type: application/json

{
  "embedding": [0.1, 0.2, ...],
  "limit": 5
}
\`\`\`

---

## Unity SSE Stream

### Connect to Stream
\`\`\`http
GET /api/stream/unity?client_id=unity_game
Accept: text/event-stream
\`\`\`

**Events:**
| Event | Description |
|-------|-------------|
| \`connected\` | Initial connection |
| \`paper_validated\` | Paper validation complete |
| \`leaderboard_update\` | Leaderboard changed |
| \`new_paper\` | New paper ready |
| \`device_joined\` | New device registered |
| \`validation_progress\` | Progress update |
| \`heartbeat\` | Keep-alive (30s) |

**Example Event:**
\`\`\`
id: 1
event: paper_validated
data: {"paper_id": "PMC12345", "title": "...", "rules_results": {"is_rct": true}}
\`\`\`

---

## MCP Integration (n8n)

### Connection Flow
1. Connect to \`GET /sse\` → Receive \`session_id\`
2. Send JSON-RPC messages to \`POST /messages?session_id=...\`
3. Receive responses via SSE stream

### Available MCP Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| \`submit_paper\` | paper_id, title?, pdf_url?, priority?, source? | Submit paper |
| \`create_rule\` | rule_id, question, positive_phrases, negative_phrases?, threshold? | Create rule |
| \`process_paper\` | paper_id | Process paper sections |
| \`get_paper_status\` | paper_id | Get validation status |
| \`get_leaderboard\` | limit? | Get leaderboard |
| \`get_system_stats\` | - | Get system stats |
| \`load_default_rules\` | - | Load default rules |

### Example: Tool Call
\`\`\`json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "submit_paper",
    "arguments": {
      "paper_id": "PMC12345",
      "title": "My Paper",
      "priority": 8
    }
  }
}
\`\`\`

---

## Molecule Visualization

### Paper Molecule (Chain)
\`\`\`
[Abstract] ─► [Intro] ─► [Methods] ─► [Results] ─► [Discussion]
   Blue       Green      Yellow        Red         Purple
\`\`\`

Each chunk contains:
- \`embedding_b64\`: 768-dim BioBERT embedding
- \`color\`: RGB for rendering
- \`position\`: 3D coordinates
- \`connects_to\`: Links to next chunk

### Rule Molecule (Dipole)
\`\`\`
[POSITIVE] ◄──► [NEGATIVE]
   Green           Red
\`\`\`

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (missing parameters) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## n8n Webhook Examples

### Paper Submission
\`\`\`
URL: http://server:8089/api/papers/submit
Method: POST
Headers: Content-Type: application/json
Body: 
{
  "paper_id": "{{$json.pmcid}}",
  "title": "{{$json.title}}",
  "pdf_url": "{{$json.pdf_url}}",
  "priority": 7,
  "source": "n8n"
}
\`\`\`

### Rule Creation
\`\`\`
URL: http://server:8089/api/rules/create
Method: POST
Body:
{
  "rule_id": "custom_rule",
  "question": "...",
  "positive_phrases": [...],
  "negative_phrases": [...],
  "threshold": 0.75
}
\`\`\`

---

## Embedding Format

All embeddings are:
- **768 dimensions** (BioBERT)
- **Float32** values
- **Base64 encoded** for transport

To decode in Python:
\`\`\`python
import base64
import numpy as np

embedding = np.frombuffer(
    base64.b64decode(embedding_b64), 
    dtype=np.float32
)
\`\`\`

To decode in C# (Unity):
\`\`\`csharp
byte[] bytes = Convert.FromBase64String(embedding_b64);
float[] embedding = new float[768];
Buffer.BlockCopy(bytes, 0, embedding, 0, bytes.Length);
\`\`\`
