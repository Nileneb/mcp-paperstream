# 📋 mcp-paperstream TODO-Liste & Datenfluss-Dokumentation

> **Stand:** 23.01.2026  
> **Status:** 🔴 Viele Handler noch leer, server.py funktionsfähig aber nicht integriert

---

## 🏗️ Projektstruktur

```
mcp-paperstream/
├── smythery.yaml          ⚠️ Prüfen auf optimierungsmöglichkeiten
├── uv.lock                ⚠️ Prüfen auf optimierungsmöglichkeiten
└── src/paperstream/
    ├── __init__.py        ❌ LEER
    ├── config.yaml        ❌ LEER
    ├── server.py          ✅ FUNKTIONIERT (DiffusionBERTScore IoT Server)
    ├── handlers/
    │   ├── __init__.py            ❌ LEER
    │   ├── biobert_handler.py     ❌ LEER
    │   ├── biomedclip_handler.py  ❌ LEER
    │   ├── download_model.py      ⚠️ STANDALONE (nicht als Modul nutzbar)
    │   └── sd_api_client.py       ❌ LEER
    └── prompts/
        ├── __init__.py            ❌ LEER
        ├── scientific_templates.py ❌ LEER
        └── term_mappings.json      ❌ LEER
```

---

## 🚨 KRITISCHE INKONSISTENZEN



### 2. Modell-Pfad Inkonsistenz
| Datei | Pfad | Problem |
|-------|------|---------|
| `download_model.py` | `../models/biobert/distil-biobert` | Relativer Pfad, hängt vom Ausführungsort ab |
| `server.py` | Kein Modell-Pfad definiert | Nutzt noch Placeholder-Tokenizer |

**FIX:** Absoluten Pfad in `config.yaml` definieren, beide Skripte lesen daraus

### 3. server.py nutzt NICHT die Handler
`server.py` hat eigene `_tokenize_simple()` Funktion statt `biobert_handler.py` zu nutzen!

---

## 📁 DATEI-SPEZIFISCHE TODOs

---

### 📄 `src/paperstream/__init__.py`

**Aufgabe:** Package initialisieren, Submodule exportieren

**INPUT:** Keine  
**VERARBEITUNG:** Imports definieren  
**OUTPUT:** Verfügbare Module/Klassen

```python
# TODO: Implementieren
from .server import mcp
from .handlers import biobert_handler, biomedclip_handler, sd_api_client

__version__ = "0.1.0"
__all__ = ["mcp", "biobert_handler", "biomedclip_handler", "sd_api_client"]
```

---

### 📄 `src/paperstream/config.yaml`

**Aufgabe:** Zentrale Konfiguration für alle Module

**INPUT:** Keine (wird von anderen Modulen gelesen)  
**VERARBEITUNG:** YAML-Parsing  
**OUTPUT:** Konfigurationswerte

```yaml
# TODO: Implementieren
server:
  host: "0.0.0.0"
  port: 8089
  sse_path: "/sse-bertscore"
  result_path: "/bert-result"

models:
  biobert:
    path: "./models/biobert/distil-biobert"
    model_name: "nlpie/distil-biobert"
  biomedclip:
    path: "./models/biomedclip"
    model_name: "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

stable_diffusion:
  api_url: "http://127.0.0.1:7860"
  timeout: 60

iot:
  assign_ttl: 30
  max_inflight_per_client: 1
  min_clients_for_distributed: 2
  tinybert_layers: 6
  embedding_dim: 312
```

---

### 📄 `src/paperstream/server.py` ✅

**Status:** Funktioniert, aber verwendet Placeholder statt echte Handler

**Aufgabe:** MCP Server für verteilte BERTScore-Berechnung

**INPUT:**  
- REST-Requests (BERTScore-Anfragen)
- SSE-Verbindungen (IoT-Clients)
- Task-Results (von IoT-Workern)

**VERARBEITUNG:**  
- Job erstellen → Tasks aufteilen → an IoT-Clients verteilen
- Embeddings aggregieren → BERTScore berechnen

**OUTPUT:**  
- SSE-Events (Tasks an Clients)
- JSON-Responses (Job-Status, Scores)

**TODOs:**
| Zeile | Problem | Fix |
|-------|---------|-----|
| 72 | `_tokenize_simple()` = Placeholder | `biobert_handler.tokenize()` nutzen |
| - | Kein Config-Loader | `config.yaml` einlesen |
| - | Hardcoded Konstanten | Aus Config laden |

```python
# ÄNDERN von:
def _tokenize_simple(text: str) -> List[str]:
    return text.lower().split()

# ZU:
from .handlers.biobert_handler import BioBERTHandler
biobert = BioBERTHandler()

def _tokenize(text: str) -> Tuple[List[str], List[int]]:
    return biobert.tokenize(text)
```

---

### 📄 `src/paperstream/handlers/__init__.py`

**Aufgabe:** Handler-Submodule exportieren

```python
# TODO: Implementieren
from .biobert_handler import BioBERTHandler
from .biomedclip_handler import BiomedCLIPHandler
from .sd_api_client import StableDiffusionClient

__all__ = ["BioBERTHandler", "BiomedCLIPHandler", "StableDiffusionClient"]
```

---

### 📄 `src/paperstream/handlers/biobert_handler.py` ❌

**Aufgabe:** BioBERT/TinyBERT Tokenisierung & Embedding-Berechnung

**INPUT:**  
- Text (str)
- Optional: Layer-Range für partielle Berechnung

**VERARBEITUNG:**  
- Tokenisierung mit BioBERT-Tokenizer
- Embedding-Berechnung (optional: nur bestimmte Layer)

**OUTPUT:**  
- Token-Liste (List[str])
- Token-IDs (List[int])
- Embeddings (List[float] oder torch.Tensor)

```python
# TODO: Implementieren
"""
BioBERT Handler für Tokenisierung und Embedding-Berechnung
"""
import os
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel

class BioBERTHandler:
    """Handler für distil-biobert Modell"""
    
    def __init__(self, model_path: str = "./models/biobert/distil-biobert"):
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_path)
            self._model.eval()
        return self._model
    
    def tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Tokenisiert Text mit BioBERT-Tokenizer.
        
        Args:
            text: Eingabetext
            
        Returns:
            (tokens, token_ids): Liste der Tokens und deren IDs
        """
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        token_ids = encoded["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens, token_ids
    
    def embed(
        self, 
        text: str, 
        layer_range: Optional[Tuple[int, int]] = None
    ) -> List[float]:
        """
        Berechnet Embeddings für Text.
        
        Args:
            text: Eingabetext
            layer_range: Optional (start, end) für partielle Layer-Ausgabe
            
        Returns:
            Embedding-Vektor als Liste von Floats
        """
        with torch.no_grad():
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**encoded, output_hidden_states=True)
            
            if layer_range:
                # Nur bestimmte Layer ausgeben
                hidden_states = outputs.hidden_states[layer_range[0]:layer_range[1]]
                # Mean über Layer und Tokens
                embedding = torch.stack(hidden_states).mean(dim=[0, 2])
            else:
                # Letzter Layer, Mean über Tokens
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            return embedding.squeeze().tolist()

# Singleton für einfachen Import
_handler: Optional[BioBERTHandler] = None

def get_handler() -> BioBERTHandler:
    global _handler
    if _handler is None:
        _handler = BioBERTHandler()
    return _handler
```

---

### 📄 `src/paperstream/handlers/biomedclip_handler.py` ❌

**Aufgabe:** BiomedCLIP für Text-Bild-Ähnlichkeit

**INPUT:**  
- Text (str) ODER
- Bild (PIL.Image oder Pfad)

**VERARBEITUNG:**  
- Text-/Bild-Encoding mit BiomedCLIP
- Similarity-Score berechnen

**OUTPUT:**  
- Text-Embedding (List[float])
- Bild-Embedding (List[float])
- Similarity-Score (float)

```python
# TODO: Implementieren
"""
BiomedCLIP Handler für Text-Bild-Vergleiche
"""
from typing import List, Union, Optional
from PIL import Image
import torch

class BiomedCLIPHandler:
    """Handler für BiomedCLIP Modell"""
    
    def __init__(self, model_path: str = "./models/biomedclip"):
        self.model_path = model_path
        self._model = None
        self._processor = None
    
    def encode_text(self, text: str) -> List[float]:
        """Berechnet Text-Embedding"""
        # TODO: Implementieren
        raise NotImplementedError("BiomedCLIP noch nicht implementiert")
    
    def encode_image(self, image: Union[str, Image.Image]) -> List[float]:
        """Berechnet Bild-Embedding"""
        # TODO: Implementieren
        raise NotImplementedError("BiomedCLIP noch nicht implementiert")
    
    def similarity(self, text: str, image: Union[str, Image.Image]) -> float:
        """Berechnet Text-Bild-Ähnlichkeit (0-1)"""
        # TODO: Implementieren
        raise NotImplementedError("BiomedCLIP noch nicht implementiert")
```

---

### 📄 `src/paperstream/handlers/sd_api_client.py` ❌

**Aufgabe:** Client für AUTOMATIC1111 Stable Diffusion WebUI API

**INPUT:**  
- Prompt (str)
- Negative Prompt (str)
- Parameter (CFG Scale, Steps, Sampler, etc.)

**VERARBEITUNG:**  
- HTTP-Request an SD WebUI API
- Bild aus Base64 dekodieren

**OUTPUT:**  
- Generiertes Bild (PIL.Image)
- Seed (int)
- Generation-Info (dict)

```python
# TODO: Implementieren
"""
Stable Diffusion API Client für AUTOMATIC1111 WebUI
"""
import base64
import httpx
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image

class StableDiffusionClient:
    """Client für SD WebUI API"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        self.api_url = api_url.rstrip("/")
        self.timeout = 120.0
    
    async def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        sampler: str = "DPM++ 2M Karras",
        seed: int = -1,
    ) -> Dict[str, Any]:
        """
        Generiert Bild aus Text-Prompt.
        
        Returns:
            {
                "image": PIL.Image,
                "seed": int,
                "info": dict
            }
        """
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "sampler_name": sampler,
            "seed": seed,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_url}/sdapi/v1/txt2img",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        # Bild dekodieren
        img_data = base64.b64decode(result["images"][0])
        image = Image.open(BytesIO(img_data))
        
        return {
            "image": image,
            "seed": result.get("seed", seed),
            "info": result.get("info", {}),
        }
    
    async def health_check(self) -> bool:
        """Prüft ob SD WebUI erreichbar ist"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_url}/sdapi/v1/options")
                return response.status_code == 200
        except Exception:
            return False
```

---

### 📄 `src/paperstream/handlers/download_model.py` ⚠️

**Status:** Funktioniert, aber nicht als Modul nutzbar

**Aufgabe:** Modelle von HuggingFace herunterladen und lokal speichern

**FIX:**
```python
# ÄNDERN von:
tokenizer.save_pretrained("../models/biobert/distil-biobert")

# ZU:
"""
Model Download Utility
Nutzt Pfade aus config.yaml
"""
import os
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

def get_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

def download_biobert(save_path: str = None):
    """Lädt distil-biobert herunter und speichert lokal"""
    config = get_config()
    
    model_name = config.get("models", {}).get("biobert", {}).get(
        "model_name", "nlpie/distil-biobert"
    )
    save_path = save_path or config.get("models", {}).get("biobert", {}).get(
        "path", "./models/biobert/distil-biobert"
    )
    
    print(f"📥 Lade {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"✅ Gespeichert in {save_path}")

if __name__ == "__main__":
    download_biobert()
```

---

### 📄 `src/paperstream/prompts/__init__.py`

```python
# TODO: Implementieren
from .scientific_templates import get_template, TEMPLATES
from .term_mappings import get_visual_terms

__all__ = ["get_template", "TEMPLATES", "get_visual_terms"]
```

---

### 📄 `src/paperstream/prompts/scientific_templates.py` ❌

**Aufgabe:** Prompt-Templates für wissenschaftliche Visualisierungen

**INPUT:**  
- Template-Name (str)
- Variablen (dict)

**OUTPUT:**  
- Fertiger SD-Prompt (str)

```python
# TODO: Implementieren
"""
Scientific Prompt Templates für Stable Diffusion
"""
from typing import Dict, Any

TEMPLATES = {
    "cell_diagram": {
        "base": "scientific diagram of {cell_type} cell, labeled parts, "
                "textbook illustration style, white background, high detail",
        "negative": "photo, realistic, blurry, text, watermark",
    },
    "molecular_structure": {
        "base": "3D molecular structure of {molecule}, ball-and-stick model, "
                "scientific visualization, clean background",
        "negative": "cartoon, sketch, blurry",
    },
    "anatomical": {
        "base": "medical illustration of {organ}, anatomical cross-section, "
                "labeled diagram, textbook style",
        "negative": "photo, x-ray, blurry, gore",
    },
    "process_flow": {
        "base": "scientific flowchart showing {process}, arrows, labeled steps, "
                "infographic style, clean design",
        "negative": "photo, 3D, complex background",
    },
}

def get_template(name: str, variables: Dict[str, Any]) -> Dict[str, str]:
    """
    Füllt Template mit Variablen.
    
    Args:
        name: Template-Name (z.B. "cell_diagram")
        variables: Dict mit Platzhaltern (z.B. {"cell_type": "neuron"})
    
    Returns:
        {"prompt": str, "negative_prompt": str}
    """
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    
    template = TEMPLATES[name]
    return {
        "prompt": template["base"].format(**variables),
        "negative_prompt": template.get("negative", ""),
    }
```

---

### 📄 `src/paperstream/prompts/term_mappings.json` ❌

**Aufgabe:** Mapping von wissenschaftlichen Begriffen zu visuellen Beschreibungen

```json
{
    "_meta": {
        "description": "Maps scientific terms to visual descriptors for SD prompts",
        "version": "0.1.0"
    },
    "cell_types": {
        "neuron": ["nerve cell", "neural cell", "brain cell"],
        "erythrocyte": ["red blood cell", "RBC"],
        "leukocyte": ["white blood cell", "WBC", "immune cell"]
    },
    "molecules": {
        "DNA": ["double helix", "deoxyribonucleic acid"],
        "ATP": ["adenosine triphosphate", "energy molecule"],
        "glucose": ["blood sugar", "C6H12O6"]
    },
    "visual_styles": {
        "textbook": ["educational", "labeled", "diagram"],
        "research": ["detailed", "high resolution", "publication quality"],
        "simplified": ["basic", "schematic", "overview"]
    }
}
```

---

## 📊 DATENFLUSS-ÜBERSICHT

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCP-PAPERSTREAM                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  config.yaml                                                                 │
│  ═══════════                                                                 │
│  • Server-Einstellungen (Host, Port, Paths)                                 │
│  • Model-Pfade (BioBERT, BiomedCLIP)                                        │
│  • SD API URL                                                                │
│  • IoT-Konfiguration                                                         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌─────────────────────┐ ┌───────────────┐ ┌─────────────────────┐
│   biobert_handler   │ │  sd_api_client │ │  biomedclip_handler │
│   ═══════════════   │ │  ═════════════ │ │  ════════════════   │
│                     │ │                │ │                     │
│ IN: Text            │ │ IN: Prompt     │ │ IN: Text + Image    │
│     Layer-Range     │ │     Params     │ │                     │
│                     │ │                │ │ OUT: Similarity     │
│ OUT: Tokens         │ │ OUT: Image     │ │      Score (0-1)    │
│      Token-IDs      │ │      Seed      │ │                     │
│      Embeddings     │ │      Info      │ │ [NICHT IMPLEMENTIERT]│
└──────────┬──────────┘ └───────┬───────┘ └─────────────────────┘
           │                    │
           └──────────┬─────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  server.py (MCP Server)                                                      │
│  ══════════════════════                                                      │
│                                                                              │
│  TOOLS:                                                                      │
│  • bertscore_compute(reference, candidate) → Job-ID + Score                 │
│  • bertscore_status(job_id) → Status + Score                                │
│  • register_iot_client(client_id, capability) → Registration                │
│  • submit_task_result(task_id, embedding) → Accepted                        │
│  • get_system_stats() → Client-Stats                                        │
│                                                                              │
│  ENDPOINTS:                                                                  │
│  • GET /sse-bertscore?client_id=X → SSE Task Stream                         │
│  • POST /bert-result → Task Result Submission                                │
│  • GET /health → Health Check                                                │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  IoT Workers (externe Geräte)                                                │
│  ════════════════════════════                                                │
│                                                                              │
│  • Verbinden via SSE                                                         │
│  • Empfangen EmbeddingTasks                                                  │
│  • Berechnen Teil-Embeddings                                                 │
│  • Senden Ergebnis zurück                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ PRIORITÄTEN-LISTE

### Sofort (Blocking)
1. [ ] `smythery.yaml` ← Inhalt aus `uv.lock` verschieben
2. [ ] `uv.lock` neu generieren mit `uv lock`
3. [ ] `config.yaml` ausfüllen

### Hoch (Core Functionality)
4. [ ] `biobert_handler.py` implementieren
5. [ ] `server.py` anpassen: `_tokenize_simple()` → `biobert_handler.tokenize()`
6. [ ] `download_model.py` refactoren (config.yaml nutzen)

### Mittel (Extended Features)
7. [ ] `sd_api_client.py` implementieren
8. [ ] `scientific_templates.py` implementieren
9. [ ] `term_mappings.json` ausfüllen

### Niedrig (Optional)
10. [ ] `biomedclip_handler.py` implementieren
11. [ ] Unit Tests hinzufügen
12. [ ] Dokumentation (README.md) schreiben

---

## 🔗 NAMING-KONVENTIONEN

| Typ | Konvention | Beispiel |
|-----|------------|----------|
| Klassen | PascalCase | `BioBERTHandler`, `IoTClient` |
| Funktionen | snake_case | `tokenize()`, `get_handler()` |
| Konstanten | UPPER_SNAKE | `TINYBERT_LAYERS`, `EMBEDDING_DIM` |
| Module | snake_case | `biobert_handler`, `sd_api_client` |
| Config-Keys | snake_case | `model_path`, `api_url` |

**Inkonsistenzen gefunden:**
- `_tokenize_simple` in server.py → sollte `_tokenize` heißen oder Handler nutzen

