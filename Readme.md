
# 🎯 paperstreamReview Server-Side TODO
## Ziel: MCP Server mit BioBERT + Stable Diffusion API Integration

---

## 📊 ARCHITEKTUR-ÜBERSICHT

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP SERVER                                │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │  BioBERT/       │   │  Prompt Builder   │   │  SD API      │  │
│  │  DistilBioBERT  │──▶│  (Term Expansion) │──▶│  Client      │  │
│  │  (Text)         │   │                   │   │              │  │
│  └─────────────────┘   └──────────────────┘   └──────┬───────┘  │
│           │                                          │          │
│           │            ┌─────────────────┐           │          │
│           └───────────▶│  BiomedCLIP     │◀──────────┘          │
│                        │  (Validation)    │                      │
│                        └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│           AUTOMATIC1111 STABLE DIFFUSION WEBUI (WSL)            │
│                    http://127.0.0.1:7860                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │ /sdapi/v1/  │   │ /sdapi/v1/  │   │ Models: SD 1.5/2.1  │    │
│  │ txt2img     │   │ img2img     │   │ + Science LoRAs     │    │
│  └─────────────┘   └─────────────┘   └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ PHASE 1: Stable Diffusion Server Setup (WSL)

### 1.1 AUTOMATIC1111 Installation
- [x] Repository klonen (bereits erledigt?)
- [x] Python venv erstellen
- [x] Dependencies installieren
- [x] **API aktivieren**: `webui.sh --api --listen`

### 1.2 Modelle herunterladen (→ `models/Stable-diffusion/`)
**Checkpoint-Modelle:**
```bash
# Pfad: stable-diffusion-webui/models/Stable-diffusion/
```

### 1.3 API testen
```bash
# Teste ob API läuft >> läuft
curl http://127.0.0.1:7860/sdapi/v1/sd-models
```

---

## ✅ PHASE 2: NLP-Modelle Download (HuggingFace)

### 2.1 BioBERT Modelle (→ `./models/biobert/`)
```bash
# Erstelle Modell-Verzeichnis
mkdir -p models/biobert models/biomedclip models/blip
```

### 2.2 Download via Python
```python
from transformers import AutoTokenizer, AutoModel

# Option A: Full BioBERT (440MB)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

# Option B: DistilBioBERT (265MB) - EMPFOHLEN für Server
tokenizer = AutoTokenizer.from_pretrained("nlpie/distil-biobert")
model = AutoModel.from_pretrained("nlpie/distil-biobert")

# Option C: TinyBioBERT (56MB) - Für IoT/Edge
tokenizer = AutoTokenizer.from_pretrained("nlpie/tiny-biobert")
model = AutoModel.from_pretrained("nlpie/tiny-biobert")
```

---

## ✅ PHASE 3: MCP Server Anpassung

### 3.1 Neue Abhängigkeiten
```bash
pip install transformers torch open_clip_torch requests pillow
```

### 3.2 Server-Struktur
```
mcp_paperstream_server/
├── __init__.py
├── server.py              # Hauptserver (MCP-CLi/Uvicorn)
├── handlers/
│   ├── __init__.py
│   ├── biobert_handler.py # BioBERT Embeddings + Term Expansion
│   ├── biomedclip_handler.py # Bildvalidierung
│   └── sd_api_client.py   # AUTOMATIC1111 API Client
├── models/
├── prompts/
│   ├── __init__.py
│   ├── scientific_templates.py  # Prompt-Vorlagen
│   └── term_mappings.json       # Bio-Term → Visual-Term
└── config.yaml
```

### 3.3 Kernfunktionen implementieren

**A) BioBERT Handler:**
- [ ] Biomedical Term Extraction
- [ ] Semantic Similarity für Prompt-Refinement
- [ ] Term Expansion (z.B. "mitosis" → ["cell division", "chromosomes", "spindle fibers"])

**B) SD API Client:**
- [ ] txt2img Wrapper
- [ ] img2img Wrapper  
- [ ] Parameter-Presets für wissenschaftliche Diagramme

**C) BiomedCLIP Handler (optional):**
- [ ] Generiertes Bild validieren
- [ ] Text-Bild Alignment prüfen

---

## ✅ PHASE 4: Integration & Testing

### 4.1 End-to-End Flow testen
```
Input: "Generate diagram of CRISPR-Cas9 mechanism"
   ↓
BioBERT: Extract terms → ["CRISPR", "Cas9", "guide RNA", "DNA cleavage"]
   ↓
Prompt Builder: "scientific diagram, CRISPR-Cas9 mechanism, guide RNA 
                 binding to DNA, Cas9 protein cleaving double helix, 
                 molecular biology illustration, clean lines, labeled components"
   ↓
SD API: POST /sdapi/v1/txt2img
   ↓
(Optional) BiomedCLIP: Validate output
   ↓
Return: Generated image
```

### 4.2 Qualitätskriterien
- [ ] Bildgenerierung < 30s
- [ ] Relevanz-Score (BiomedCLIP) > 0.7
- [ ] Keine anatomischen Fehler

---

## 📥 VALIDE DOWNLOAD-LINKS

### Stable Diffusion Checkpoints

| Modell | Link | Größe | Empfehlung |
|--------|------|-------|------------|
| **SD 1.5** | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors | 4.27 GB | ✅ Standard |
| **SD 2.1 Base** | https://huggingface.co/sd-research/stable-diffusion-2-1-base | 5 GB | Alternative |

### BioBERT Familie (HuggingFace)

| Modell | HuggingFace ID | Parameter | Empfehlung |
|--------|----------------|-----------|------------|
| **BioBERT Base v1.2** | `dmis-lab/biobert-base-cased-v1.2` | 110M | Full-Feature |
| **BioBERT Base v1.1** | `dmis-lab/biobert-base-cased-v1.1` | 110M | Alternative |
| **DistilBioBERT** ⭐ | `nlpie/distil-biobert` | 65M | ✅ EMPFOHLEN |
| **TinyBioBERT** | `nlpie/tiny-biobert` | 15M | Edge/IoT |
| **CompactBioBERT** | `nlpie/compact-biobert` | 65M | Balanced |

### Vision-Language Modelle (Optional)

| Modell | HuggingFace ID | Funktion |
|--------|----------------|----------|
| **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Bild-Text Validierung |
| **BLIP Image Captioning** | `Salesforce/blip-image-captioning-base` | Caption Generation |
| **PubMedBERT** | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Alternative zu BioBERT |

### LoRAs für wissenschaftliche Bilder
- Civitai durchsuchen nach: "scientific", "diagram", "medical illustration"
- Pfad: `stable-diffusion-webui/models/Lora/`

---

## 📋 QUICK START COMMANDS

```bash
# 1. AUTOMATIC1111 starten (in WSL)
cd stable-diffusion-webui
./webui.sh --api --listen --xformers

# 2. API Docs öffnen
# http://127.0.0.1:7860/docs

# 3. Modelle downloaden (Python)
python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('nlpie/distil-biobert')
model = AutoModel.from_pretrained('nlpie/distil-biobert')
model.save_pretrained('./models/distil-biobert')
tokenizer.save_pretrained('./models/distil-biobert')
print('DistilBioBERT heruntergeladen')
"
```

### Test API Call (Python)
```python
import requests
import base64
from PIL import Image
import io

url = "http://127.0.0.1:7860"
payload = {
    "prompt": "scientific diagram of DNA double helix structure, clean lines, labeled, educational illustration",
    "negative_prompt": "blurry, text, watermark",
    "steps": 20,
    "width": 512,
    "height": 512,
    "sampler_name": "Euler"
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
r = response.json()

# Decode image
image_data = base64.b64decode(r['images'][0])
image = Image.open(io.BytesIO(image_data))
image.save('test_output.png')
```

---

## ⏰ PRIORITÄTS-REIHENFOLGE

1. **JETZT**: SD WebUI mit `--api` starten, API testen
2. **DANN**: DistilBioBERT herunterladen und lokal cachen
3. **DANN**: SD API Client im MCP Server implementieren
4. **DANN**: BioBERT-Prompt-Pipeline bauen
5. **OPTIONAL**: BiomedCLIP für Validierung

---

Erstellt: 2026-01-23 11:08
