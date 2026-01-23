# Copilot Instructions for mcp-paperstream

## Project Overview
**mcp-paperstream** is a distributed MCP server for scientific paper review combining:
1. **BERTScore computation** distributed across IoT edge devices (ESP32, RPi, smartphones)
2. **Stable Diffusion integration** for scientific visualizations
3. **BiomedCLIP validation** for text-image semantic alignment

## Architecture Essentials
WICHTIG!!!! PRÜFE IMMER OB DAS CONDA ENVIROMENT AKTIVIERT IST MIT NEM PYTHON 3.12 UND .VENV!!!!!!!
CONDA ENVIROMENT: "paperstream"
PYTHON VERSION: 3.12
PYTHON ENV: ".venv"
### Core Components
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
- `stable_diffusion`: API URL, timeout
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

### SD API Client
```python
from paperstream.handlers import get_sd_client
client = get_sd_client()
result = await client.txt2img(prompt="cell diagram of neuron", steps=20)
# result["images"][0] is PIL.Image
```

### Prompt Templates
```python
from paperstream.prompts import get_template, get_visual_terms
prompt = get_template("cell_diagram", {"cell_type": "neuron"})
terms = get_visual_terms("neuron")  # Returns synonyms, visual_descriptors
```

## Critical State Dictionaries
Three dicts in `server.py` must stay synchronized:
- `clients`: Device metadata, latency stats
- `clients_inflight`: Active task count (max 1 for IoT)
- `clients_queues`: Async queues for SSE delivery

## Common Pitfalls
- **Task timeout**: Expired tasks auto-reassign but need `bertscore_status()` call for aggregation
- **SSE cleanup**: Disconnected clients stay registered; stale entries in `get_system_stats()`
- **Model loading**: Handlers lazy-load models; first call is slow
- **open_clip optional**: BiomedCLIP requires `pip install open_clip_torch`

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

## File Quick Reference
| File | Purpose |
|------|---------|
| [server.py](src/paperstream/server.py) | MCP tools, SSE endpoint, task distribution |
| [config.yaml](src/paperstream/config.yaml) | All configurable settings |
| [biobert_handler.py](src/paperstream/handlers/biobert_handler.py) | `BioBERTHandler.tokenize()`, `.embed()` |
| [sd_api_client.py](src/paperstream/handlers/sd_api_client.py) | `StableDiffusionClient.txt2img()` |
| [scientific_templates.py](src/paperstream/prompts/scientific_templates.py) | `TEMPLATES` dict, `get_template()` |
| [term_mappings.json](src/paperstream/prompts/term_mappings.json) | Scientific vocabulary mappings |

