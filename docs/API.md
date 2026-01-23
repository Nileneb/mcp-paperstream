# PaperStream API Documentation

## Base URL
```
http://localhost:8089
```

## Endpoints Overview

### MCP (n8n Integration)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sse` | GET | SSE stream for MCP client |
| `/messages` | POST | MCP JSON-RPC messages |
| `/mcp` | * | Streamable HTTP endpoint |

### REST API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | System statistics |
| `/api/papers/submit` | POST | Submit new paper |
| `/api/papers` | GET | List all papers |
| `/api/papers/{paper_id}` | GET | Get paper details |
| `/api/rules/create` | POST | Create validation rule |
| `/api/rules` | GET | List all active rules |
| `/api/jobs/next` | GET | Get jobs for Android device |
| `/api/validation/submit` | POST | Submit validation results |
| `/api/devices/register` | POST | Register Android device |
| `/api/devices/{device_id}` | GET | Get device info |
| `/api/consensus/{paper_id}` | GET | Get validation status |
| `/api/stream/unity` | GET | SSE stream for Unity |

---

## MCP Integration (n8n)

The server exposes MCP tools via **SSE Transport** at `/sse`.

### Connection Flow
1. Connect to `GET /sse` → Receive `session_id`
2. Send JSON-RPC messages to `POST /messages?session_id=...`
3. Receive responses via SSE stream

### Available MCP Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `submit_paper` | paper_id, title?, pdf_url?, priority?, source? | Submit paper |
| `create_rule` | rule_id, question, positive_phrases, negative_phrases?, threshold? | Create rule |
| `process_paper` | paper_id | Process paper sections |
| `get_paper_status` | paper_id | Get validation status |
| `get_leaderboard` | limit? | Get leaderboard |
| `get_system_stats` | - | Get system stats |
| `load_default_rules` | - | Load default rules |

### Example: Tool Call
```json
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
```

---

## Papers API

### Submit Paper
```http
POST /api/papers/submit
Content-Type: application/json

{
  "paper_id": "PMC12345",
  "title": "Randomized Trial of...",
  "pdf_url": "https://...",
  "priority": 7,
  "source": "n8n"
}
```

**Response:**
```json
{
  "status": "accepted",
  "paper_id": "PMC12345",
  "message": "Paper queued for processing"
}
```

### List Papers
```http
GET /api/papers?status=pending&limit=100
```

**Response:**
```json
{
  "papers": [
    {
      "id": 1,
      "paper_id": "PMC12345",
      "title": "...",
      "status": "pending",
      "priority": 7
    }
  ]
}
```

---

## Rules API

### Create Rule
```http
POST /api/rules/create
Content-Type: application/json

{
  "rule_id": "is_rct",
  "question": "Is this a Randomized Controlled Trial?",
  "positive_phrases": ["randomized controlled trial", "RCT"],
  "negative_phrases": ["observational", "retrospective"],
  "threshold": 0.75
}
```

**Response:**
```json
{
  "status": "created",
  "rule_id": "is_rct",
  "message": "Rule created with 2 positive phrases",
  "embedding_dim": 768
}
```

### List Rules
```http
GET /api/rules
```

---

## Jobs API (Android)

### Get Next Jobs
```http
GET /api/jobs/next?device_id=android_abc123&limit=5
```

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_abc123",
      "paper_id": "PMC12345",
      "section_id": 1,
      "rule_id": "is_rct",
      "question": "Is this a RCT?",
      "threshold": 0.75,
      "section_text": "...",
      "voxel_data": "[[...]]",
      "section_embedding_b64": "...",
      "pos_embedding_b64": "...",
      "neg_embedding_b64": "..."
    }
  ],
  "assigned_count": 1,
  "device_id": "android_abc123",
  "expires_in_seconds": 300
}
```

### Submit Validation Results
```http
POST /api/validation/submit
Content-Type: application/json

{
  "device_id": "android_abc123",
  "results": [
    {
      "job_id": "job_abc123",
      "is_match": true,
      "similarity": 0.89,
      "confidence": 0.92,
      "time_taken_ms": 1500
    }
  ]
}
```

**Response:**
```json
{
  "accepted": 1,
  "rejected": 0,
  "total_points": 89
}
```

---

## Devices API

### Register Device
```http
POST /api/devices/register
Content-Type: application/json

{
  "device_id": "android_abc123",
  "device_name": "My Phone",
  "device_model": "Pixel 6",
  "os_version": "Android 13",
  "app_version": "1.0.0"
}
```

### Get Device Info
```http
GET /api/devices/android_abc123
```

---

## Consensus API

### Get Paper Validation Status
```http
GET /api/consensus/PMC12345
```

**Response:**
```json
{
  "paper_id": "PMC12345",
  "rules_checked": 6,
  "rules_validated": 4,
  "overall_status": "incomplete",
  "rules": [
    {
      "rule_id": "is_rct",
      "question": "Is this a RCT?",
      "is_match": true,
      "is_validated": true,
      "vote_count": 5,
      "agreement_ratio": 0.8
    }
  ]
}
```

---

## Unity SSE Stream

### Connect to Stream
```http
GET /api/stream/unity?client_id=unity_game
Accept: text/event-stream
```

**Events:**
- `connected` - Initial connection
- `paper_validated` - Paper validation complete
- `leaderboard_update` - Leaderboard changed
- `new_paper` - New paper ready
- `device_joined` - New device registered
- `validation_progress` - Progress update
- `heartbeat` - Keep-alive

**Example Event:**
```
id: 1
event: paper_validated
data: {"paper_id": "PMC12345", "title": "...", "rules_results": {"is_rct": true}}
```

---

## MCP Tools

Available via MCP endpoint `/mcp`:

| Tool | Description |
|------|-------------|
| `submit_paper` | Submit paper for processing |
| `create_rule` | Create validation rule |
| `get_paper_status` | Get paper validation status |
| `get_leaderboard` | Get gamification leaderboard |
| `get_system_stats` | Get system statistics |
| `process_paper` | Manually process a paper |
| `load_default_rules` | Load default validation rules |

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (missing parameters) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## n8n Integration

### Webhook for Paper Submission
```
URL: http://your-server:8089/api/papers/submit
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
```

### Webhook for Rule Creation
```
URL: http://your-server:8089/api/rules/create
Method: POST
Body:
{
  "rule_id": "custom_rule",
  "question": "...",
  "positive_phrases": [...],
  "threshold": 0.75
}
```
