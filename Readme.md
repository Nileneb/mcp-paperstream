# 🧬 MCP-PaperStream

**Distributed MCP Server for Scientific Paper Review** combining BERTScore computation across IoT edge devices with Stable Diffusion integration for scientific visualizations.

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP SERVER (FastMCP 2.x)                    │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │  BioBERT/       │   │  Prompt Builder   │   │  SD API      │  │
│  │  DistilBioBERT  │──▶│  (8 Templates)    │──▶│  Client      │  │
│  │  Tokenizer      │   │                   │   │              │  │
│  └─────────────────┘   └──────────────────┘   └──────┬───────┘  │
│           │                                          │          │
│           │            ┌─────────────────┐           │          │
│           └───────────▶│  BiomedCLIP     │◀──────────┘          │
│                        │  (Validation)    │                      │
│                        └─────────────────┘                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Distributed BERTScore (IoT Workers)               ││
│  │   ESP32 ──▶ Layer 0    RPi4 ──▶ Layer 0-2    Phone ──▶ 0-5  ││
│  │   (LOW)                (MEDIUM)              (HIGH)          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│           AUTOMATIC1111 STABLE DIFFUSION WEBUI                  │
│                    http://127.0.0.1:7860                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- AUTOMATIC1111 Stable Diffusion WebUI running with `--api`
- ~2GB disk space for models

### Installation

```bash
# Clone repository
git clone https://github.com/Nileneb/mcp-paperstream.git
cd mcp-paperstream

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers fastmcp pyyaml httpx pillow python-dotenv

# Optional: BiomedCLIP support
pip install open_clip_torch

# Download models (first run will auto-download from HuggingFace)
python -m src.paperstream.handlers.download_model all
```

### Start Server

```bash
# Make sure SD WebUI is running with --api flag first!
./start_server.sh

# Or manually:
.venv/bin/uvicorn src.paperstream.server:mcp --host 0.0.0.0 --port 8089
```

---

## 📁 Project Structure

```
src/paperstream/
├── server.py              # MCP server - task distribution, SSE, job management
├── config.yaml            # Central configuration
├── handlers/
│   ├── biobert_handler.py    # BioBERT tokenization & embeddings
│   ├── biomedclip_handler.py # Text-image similarity (optional)
│   ├── sd_api_client.py      # AUTOMATIC1111 SD WebUI API client
│   └── download_model.py     # Model download utility
├── models/                # Local model cache
│   ├── biobert/
│   └── biomedclip/
└── prompts/
    ├── scientific_templates.py  # 8 SD prompt templates
    └── term_mappings.json       # Scientific vocabulary mappings
```

---

## 🔧 Configuration

All settings in `src/paperstream/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8089
  sse_path: "/sse-bertscore"

models:
  biobert:
    model_name: "nlpie/distil-biobert"
  biomedclip:
    model_name: "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

stable_diffusion:
  api_url: "http://127.0.0.1:7860"
  timeout: 120

iot:
  assign_ttl: 30
  tinybert_layers: 6
```

Environment variables override config:
- `FASTMCP_HOST`, `FASTMCP_PORT`
- `BERTSCORE_HMAC` (for task signing)

---

## 🛠️ MCP Tools

| Tool | Description |
|------|-------------|
| `bertscore_compute` | Compute BERTScore (distributed or local) |
| `bertscore_status` | Check job status |
| `register_iot_client` | Register IoT device as worker |
| `submit_task_result` | Submit embedding result from worker |
| `get_system_stats` | Get system statistics |

### Example: BERTScore Computation

```python
# Via MCP client
result = await client.call_tool("bertscore_compute", {
    "reference": "The mitochondria is the powerhouse of the cell.",
    "candidate": "Mitochondria produce ATP through cellular respiration.",
    "distributed": True
})
```

---

## 🎨 Prompt Templates

8 scientific visualization templates available:

| Template | Use Case |
|----------|----------|
| `cell_diagram` | Cell structure diagrams |
| `molecular_structure` | Molecular/chemical structures |
| `anatomical` | Anatomical illustrations |
| `process_flow` | Biological process flows |
| `microscopy` | Microscopy-style images |
| `protein_structure` | Protein/enzyme structures |
| `pathway_diagram` | Metabolic/signaling pathways |
| `tissue_section` | Histological sections |

```python
from src.paperstream.prompts import get_template

prompt = get_template('cell_diagram', {
    'cell_type': 'neuron',
    'organelles': 'axon, dendrites, nucleus'
})
# Returns: {'prompt': '...', 'negative_prompt': '...', 'steps': 25, ...}
```

---

## 📡 IoT Worker Integration

### Device Capabilities

| Capability | Devices | Assigned Layers |
|------------|---------|-----------------|
| `LOW` | ESP32, RPi Zero | Layer 0 only |
| `MEDIUM` | RPi 4, old phones | Layers 0-2 |
| `HIGH` | Modern phones, tablets | All 6 layers |

### Register Worker

```python
result = await client.call_tool("register_iot_client", {
    "client_id": "rpi4-kitchen",
    "device_type": "raspberry_pi",
    "capability": "medium"
})
```

### SSE Task Stream

Workers connect to `/sse-bertscore?client_id=<id>` to receive tasks.

---

## 🧪 Testing

```bash
# Run all tests
.venv/bin/python -c "
from src.paperstream import mcp
from src.paperstream.handlers import get_biobert_handler, get_sd_client
from src.paperstream.prompts import TEMPLATES

print(f'✅ MCP Server: {mcp.name}')
print(f'✅ Templates: {len(TEMPLATES)}')

handler = get_biobert_handler()
tokens, ids = handler.tokenize('DNA replication')
print(f'✅ BioBERT: {len(tokens)} tokens')

client = get_sd_client()
print(f'✅ SD Client: {client.api_url}')
"
```

---

## 📚 Models Used

| Model | HuggingFace ID | Size | Purpose |
|-------|----------------|------|---------|
| **DistilBioBERT** | `nlpie/distil-biobert` | 265MB | Tokenization & Embeddings |
| **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | ~400MB | Image-Text Validation |

---

## 📄 License

MIT License - see LICENSE file.

---

## 🔗 Links

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [DistilBioBERT](https://huggingface.co/nlpie/distil-biobert)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
