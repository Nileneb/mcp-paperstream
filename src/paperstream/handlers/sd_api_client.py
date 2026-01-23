"""
Stable Diffusion API Client für AUTOMATIC1111 WebUI

Kommuniziert mit dem SD WebUI REST API für Bildgenerierung.
Unterstützt txt2img und img2img Endpoints.
"""
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List
from pathlib import Path

import httpx
from PIL import Image
import yaml


def load_config() -> dict:
    """Lädt Konfiguration aus config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class StableDiffusionClient:
    """Async Client für SD WebUI API"""
    
    def __init__(self, api_url: Optional[str] = None, timeout: Optional[float] = None):
        config = load_config()
        sd_config = config.get("stable_diffusion", {})
        
        self.api_url = (api_url or sd_config.get("api_url", "http://127.0.0.1:7860")).rstrip("/")
        self.timeout = timeout or sd_config.get("timeout", 120.0)
    
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
        batch_size: int = 1,
        n_iter: int = 1,
    ) -> Dict[str, Any]:
        """
        Generiert Bild aus Text-Prompt.
        
        Args:
            prompt: Positiver Prompt für Bildgenerierung
            negative_prompt: Was NICHT im Bild sein soll
            steps: Anzahl Diffusion-Schritte (20-50 empfohlen)
            cfg_scale: Classifier-Free Guidance Scale (7-12 empfohlen)
            width: Bildbreite in Pixeln (512/768/1024)
            height: Bildhöhe in Pixeln (512/768/1024)
            sampler: Sampling-Methode
            seed: Random Seed (-1 = zufällig)
            batch_size: Bilder pro Batch
            n_iter: Anzahl Batches
        
        Returns:
            {
                "images": List[PIL.Image],
                "seed": int,
                "info": dict,
                "parameters": dict
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
            "batch_size": batch_size,
            "n_iter": n_iter,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_url}/sdapi/v1/txt2img",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        # Bilder dekodieren
        images = []
        for img_b64 in result.get("images", []):
            img_data = base64.b64decode(img_b64)
            image = Image.open(BytesIO(img_data))
            images.append(image)
        
        return {
            "images": images,
            "seed": result.get("seed", seed),
            "info": result.get("info", ""),
            "parameters": result.get("parameters", {}),
        }
    
    async def img2img(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        denoising_strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler: str = "DPM++ 2M Karras",
        seed: int = -1,
    ) -> Dict[str, Any]:
        """
        Generiert Bild basierend auf Eingabebild.
        
        Args:
            init_image: Eingabebild als PIL.Image
            prompt: Positiver Prompt
            negative_prompt: Negativer Prompt
            denoising_strength: Stärke der Änderung (0-1, höher = mehr Änderung)
            steps: Anzahl Diffusion-Schritte
            cfg_scale: CFG Scale
            sampler: Sampling-Methode
            seed: Random Seed
        
        Returns:
            {"images": List[PIL.Image], "seed": int, "info": dict}
        """
        # Bild zu Base64 konvertieren
        buffered = BytesIO()
        init_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        payload = {
            "init_images": [img_b64],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler,
            "seed": seed,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_url}/sdapi/v1/img2img",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        images = []
        for img_b64 in result.get("images", []):
            img_data = base64.b64decode(img_b64)
            image = Image.open(BytesIO(img_data))
            images.append(image)
        
        return {
            "images": images,
            "seed": result.get("seed", seed),
            "info": result.get("info", ""),
        }
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Gibt Liste verfügbarer SD-Modelle zurück"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.api_url}/sdapi/v1/sd-models")
            response.raise_for_status()
            return response.json()
    
    async def get_samplers(self) -> List[Dict[str, Any]]:
        """Gibt Liste verfügbarer Sampler zurück"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.api_url}/sdapi/v1/samplers")
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> bool:
        """Prüft ob SD WebUI erreichbar ist"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_url}/sdapi/v1/options")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_progress(self) -> Dict[str, Any]:
        """Gibt aktuellen Generierungsfortschritt zurück"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.api_url}/sdapi/v1/progress")
            response.raise_for_status()
            return response.json()


# Singleton-Instanz
_client: Optional[StableDiffusionClient] = None


def get_client() -> StableDiffusionClient:
    """Gibt Singleton-Instanz des Clients zurück"""
    global _client
    if _client is None:
        _client = StableDiffusionClient()
    return _client
