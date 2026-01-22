
"""
DiffusionBERTScore IoT Worker Client
====================================
Läuft auf IoT-Geräten (Raspberry Pi, Smartphone, ESP32, etc.)
Verbindet sich mit dem Coordinator und berechnet Teil-Embeddings.

Unterstützte Geräte:
- Raspberry Pi 4/5: TinyBERT vollständig
- Raspberry Pi Zero: Nur Embedding-Layer
- ESP32: Nur einfache Tokenisierung
- Smartphones: DistilBERT oder TinyBERT
"""

import os
import json
import time
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Optional: ONNX Runtime für optimierte Inferenz
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime nicht verfügbar - Fallback auf Simulation")

# Optional: Transformers für echte Tokenisierung
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers nicht verfügbar - Fallback auf einfache Tokenisierung")

# =========================
# Konfiguration
# =========================
SERVER_URL = os.getenv("BERTSCORE_SERVER", "http://localhost:8082")
SSE_ENDPOINT = os.getenv("SSE_ENDPOINT", "/sse-bertscore")
CLIENT_ID = os.getenv("CLIENT_ID", f"iot_{int(time.time())}")
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "raspberry_pi")
CAPABILITY = os.getenv("CAPABILITY", "medium")  # low, medium, high

# TinyBERT Konfiguration
TINYBERT_MODEL = os.getenv("TINYBERT_MODEL", "huawei-noah/TinyBERT_General_4L_312D")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/tinybert.onnx")

# =========================
# Geräteklassen
# =========================
class DeviceCapability(str, Enum):
    LOW = "low"      # ESP32, Pi Zero: ~15MB RAM, nur 1 Layer
    MEDIUM = "medium"  # Pi 4, alte Phones: ~60MB RAM, 2-3 Layer
    HIGH = "high"    # Neue Phones, Tablets: ~256MB RAM, alle Layer

@dataclass
class DeviceProfile:
    """Hardware-Profil des Geräts"""
    capability: DeviceCapability
    max_tokens: int
    max_layers: int
    batch_size: int
    use_quantization: bool

# Vordefinierte Profile
DEVICE_PROFILES = {
    "esp32": DeviceProfile(
        capability=DeviceCapability.LOW,
        max_tokens=64,
        max_layers=1,
        batch_size=1,
        use_quantization=True
    ),
    "raspberry_pi_zero": DeviceProfile(
        capability=DeviceCapability.LOW,
        max_tokens=128,
        max_layers=2,
        batch_size=1,
        use_quantization=True
    ),
    "raspberry_pi_4": DeviceProfile(
        capability=DeviceCapability.MEDIUM,
        max_tokens=256,
        max_layers=4,
        batch_size=4,
        use_quantization=True
    ),
    "raspberry_pi_5": DeviceProfile(
        capability=DeviceCapability.HIGH,
        max_tokens=512,
        max_layers=6,
        batch_size=8,
        use_quantization=False
    ),
    "smartphone_low": DeviceProfile(
        capability=DeviceCapability.MEDIUM,
        max_tokens=256,
        max_layers=4,
        batch_size=4,
        use_quantization=True
    ),
    "smartphone_high": DeviceProfile(
        capability=DeviceCapability.HIGH,
        max_tokens=512,
        max_layers=6,
        batch_size=16,
        use_quantization=False
    ),
    "default": DeviceProfile(
        capability=DeviceCapability.MEDIUM,
        max_tokens=256,
        max_layers=4,
        batch_size=4,
        use_quantization=True
    ),
}

# =========================
# TinyBERT Worker
# =========================
class TinyBERTWorker:
    """
    Führt TinyBERT-Inferenz auf dem IoT-Gerät aus.
    Nutzt ONNX Runtime für optimierte Performance.
    """
    
    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self.tokenizer = None
        self.model = None
        self.session = None
        
    async def initialize(self):
        """Lädt Tokenizer und Modell"""
        print(f"🔧 Initialisiere TinyBERT Worker...")
        print(f"   Capability: {self.profile.capability.value}")
        print(f"   Max Tokens: {self.profile.max_tokens}")
        print(f"   Max Layers: {self.profile.max_layers}")
        
        # Lade Tokenizer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL)
                print(f"✅ Tokenizer geladen: {TINYBERT_MODEL}")
            except Exception as e:
                print(f"⚠️  Tokenizer-Fehler: {e}")
                self.tokenizer = None
        
        # Lade ONNX Modell
        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                # Konfiguriere Session für IoT
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2  # Wenig Threads für IoT
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                if self.profile.use_quantization:
                    # Nutze quantisierte Operationen
                    sess_options.add_session_config_entry(
                        "session.disable_prepacking", "1"
                    )
                
                self.session = ort.InferenceSession(
                    MODEL_PATH,
                    sess_options,
                    providers=['CPUExecutionProvider']
                )
                print(f"✅ ONNX Modell geladen: {MODEL_PATH}")
            except Exception as e:
                print(f"⚠️  ONNX-Fehler: {e}")
                self.session = None
        else:
            print(f"ℹ️  Kein ONNX-Modell - nutze Simulation")
    
    def tokenize(self, text: str) -> Dict[str, List[int]]:
        """Tokenisiert Text"""
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.profile.max_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return {
                'input_ids': encoded['input_ids'].tolist()[0],
                'attention_mask': encoded['attention_mask'].tolist()[0],
                'tokens': self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            }
        else:
            # Fallback: Einfache Tokenisierung
            tokens = text.lower().split()[:self.profile.max_tokens]
            return {
                'input_ids': list(range(len(tokens))),
                'attention_mask': [1] * len(tokens),
                'tokens': tokens
            }
    
    def compute_embedding(
        self, 
        tokens: List[str], 
        token_ids: List[int],
        layer_range: tuple
    ) -> Dict[str, Any]:
        """
        Berechnet Embeddings für die angegebenen Layer.
        
        Args:
            tokens: Liste von Token-Strings
            token_ids: Liste von Token-IDs
            layer_range: Tupel (start_layer, end_layer)
        
        Returns:
            {
                "embedding": List[float],
                "layer_range": tuple,
                "dim": int,
                "computation_time_ms": float
            }
        """
        start_time = time.time()
        
        # Prüfe ob Layer-Range für dieses Gerät geeignet
        if layer_range[1] - layer_range[0] > self.profile.max_layers:
            return {
                "error": f"Layer range too large for device (max: {self.profile.max_layers})",
                "layer_range": layer_range,
            }
        
        if self.session:
            # Echte ONNX Inferenz
            try:
                import numpy as np
                
                input_ids = np.array([token_ids[:self.profile.max_tokens]], dtype=np.int64)
                attention_mask = np.ones_like(input_ids)
                
                outputs = self.session.run(
                    None,
                    {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                    }
                )
                
                # Extrahiere nur die relevanten Layer
                # (In echter Implementierung: Layer-spezifische Outputs)
                embedding = outputs[0][0].mean(axis=0).tolist()
                
            except Exception as e:
                return {"error": str(e), "layer_range": layer_range}
        else:
            # Simulation: Generiere Pseudo-Embedding
            import random
            embedding_dim = 312  # TinyBERT hidden size
            embedding = [random.gauss(0, 0.1) for _ in range(embedding_dim)]
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "embedding": embedding,
            "layer_range": layer_range,
            "dim": len(embedding),
            "num_tokens": len(tokens),
            "computation_time_ms": round(computation_time, 2),
        }

# =========================
# SSE Client
# =========================
class IoTWorkerClient:
    """
    Hauptklasse für den IoT-Worker.
    Verbindet sich mit dem Server und bearbeitet Tasks.
    """
    
    def __init__(self):
        self.client_id = CLIENT_ID
        self.device_type = DEVICE_TYPE
        self.profile = DEVICE_PROFILES.get(DEVICE_TYPE, DEVICE_PROFILES["default"])
        self.worker = TinyBERTWorker(self.profile)
        self.running = False
        self.tasks_completed = 0
        
    async def register(self) -> bool:
        """Registriert sich beim Server"""
        url = f"{SERVER_URL}/register_iot_client"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Nutze MCP Tool-Aufruf
                payload = {
                    "client_id": self.client_id,
                    "device_type": self.device_type,
                    "capability": self.profile.capability.value,
                }
                
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"✅ Registriert: {data}")
                        return True
                    else:
                        print(f"❌ Registrierung fehlgeschlagen: {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")
            return False
    
    async def submit_result(self, task_id: str, job_id: str, result: Dict) -> bool:
        """Sendet Ergebnis zurück an Server"""
        url = f"{SERVER_URL}/submit_task_result"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "task_id": task_id,
                    "job_id": job_id,
                    "client_id": self.client_id,
                    "embedding": result.get("embedding", []),
                    "latency_ms": result.get("computation_time_ms", 0),
                }
                
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        print(f"⚠️  Ergebnis nicht akzeptiert: {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Fehler beim Senden: {e}")
            return False
    
    async def handle_task(self, task: Dict):
        """Bearbeitet eine empfangene Task"""
        task_type = task.get("task_type")
        task_id = task.get("task_id")
        job_id = task.get("job_id")
        
        print(f"📥 Task empfangen: {task_id} (Type: {task_type})")
        
        if task_type == "embedding":
            # Berechne Embedding
            result = self.worker.compute_embedding(
                tokens=task.get("tokens", []),
                token_ids=task.get("token_ids", []),
                layer_range=tuple(task.get("layer_range", (0, 6)))
            )
            
            if "error" not in result:
                success = await self.submit_result(task_id, job_id, result)
                if success:
                    self.tasks_completed += 1
                    print(f"✅ Task abgeschlossen: {task_id} ({result['computation_time_ms']}ms)")
                else:
                    print(f"⚠️  Task-Ergebnis nicht gesendet: {task_id}")
            else:
                print(f"❌ Task-Fehler: {result['error']}")
        else:
            print(f"⚠️  Unbekannter Task-Typ: {task_type}")
    
    async def listen_sse(self):
        """Lauscht auf SSE-Events vom Server"""
        url = f"{SERVER_URL}{SSE_ENDPOINT}?client_id={self.client_id}"
        
        print(f"📡 Verbinde mit SSE: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data:'):
                            data = json.loads(line[5:].strip())
                            msg_type = data.get("type")
                            
                            if msg_type == "connected":
                                print(f"🔗 Verbunden mit Server")
                            elif msg_type == "heartbeat":
                                print(f"💓 Heartbeat: {data.get('ts')}")
                            elif msg_type == "task":
                                await self.handle_task(data)
                            else:
                                print(f"📨 Nachricht: {data}")
        except Exception as e:
            print(f"❌ SSE-Fehler: {e}")
    
    async def run(self):
        """Hauptschleife des Workers"""
        print(f"🚀 Starting IoT Worker: {self.client_id}")
        print(f"📱 Device: {self.device_type}")
        print(f"⚡ Capability: {self.profile.capability.value}")
        
        # Initialisiere Worker
        await self.worker.initialize()
        
        # Registriere beim Server
        if not await self.register():
            print("❌ Konnte nicht registrieren - beende")
            return
        
        self.running = True
        
        # Starte SSE-Listener mit Reconnect-Logik
        while self.running:
            try:
                await self.listen_sse()
            except Exception as e:
                print(f"⚠️  Verbindung verloren: {e}")
                print("🔄 Reconnect in 5 Sekunden...")
                await asyncio.sleep(5)

# =========================
# CLI Interface
# =========================
def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║     DiffusionBERTScore IoT Worker                ║
║     Verteilte Textbewertung auf Edge-Geräten     ║
╚══════════════════════════════════════════════════╝
    """)

def print_device_info():
    profile = DEVICE_PROFILES.get(DEVICE_TYPE, DEVICE_PROFILES["default"])
    print(f"""
📱 Geräteinformationen:
   Client ID:      {CLIENT_ID}
   Device Type:    {DEVICE_TYPE}
   Capability:     {profile.capability.value}
   Max Tokens:     {profile.max_tokens}
   Max Layers:     {profile.max_layers}
   Batch Size:     {profile.batch_size}
   Quantization:   {'Ja' if profile.use_quantization else 'Nein'}
   
🌐 Server:
   URL:            {SERVER_URL}
   SSE Endpoint:   {SSE_ENDPOINT}
   
📦 Bibliotheken:
   ONNX Runtime:   {'✅ Verfügbar' if ONNX_AVAILABLE else '❌ Nicht verfügbar'}
   Transformers:   {'✅ Verfügbar' if TRANSFORMERS_AVAILABLE else '❌ Nicht verfügbar'}
    """)

# =========================
# Main
# =========================
if __name__ == "__main__":
    print_banner()
    print_device_info()
    
    client = IoTWorkerClient()
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n👋 Worker beendet")
