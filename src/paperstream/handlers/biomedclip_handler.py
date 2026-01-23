"""
BiomedCLIP Handler für Text-Bild-Vergleiche

Nutzt microsoft/BiomedCLIP für semantische Ähnlichkeit zwischen
wissenschaftlichen Texten und Bildern.
"""
from typing import List, Union, Optional
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
import yaml

# Optional imports - nur wenn verfügbar
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


def load_config() -> dict:
    """Lädt Konfiguration aus config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class BiomedCLIPHandler:
    """Handler für BiomedCLIP Modell"""
    
    def __init__(self, model_path: Optional[str] = None):
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip nicht installiert. "
                "Installiere mit: pip install open_clip_torch"
            )
        
        config = load_config()
        clip_config = config.get("models", {}).get("biomedclip", {})
        
        self.model_path = model_path or clip_config.get("path", "./models/biomedclip")
        self.model_name = clip_config.get(
            "model_name", 
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Lazy-load des Modells"""
        if self._model is None:
            # Versuche lokales Modell, sonst HuggingFace Hub
            try:
                self._model, self._preprocess = open_clip.create_model_from_pretrained(
                    self.model_path
                )
            except Exception:
                self._model, self._preprocess = open_clip.create_model_from_pretrained(
                    self.model_name
                )
            
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
    
    @property
    def model(self):
        self._load_model()
        return self._model
    
    @property
    def preprocess(self):
        self._load_model()
        return self._preprocess
    
    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer
    
    def encode_text(self, text: str) -> List[float]:
        """
        Berechnet Text-Embedding.
        
        Args:
            text: Eingabetext (z.B. wissenschaftliche Beschreibung)
            
        Returns:
            Normalisierter Embedding-Vektor
        """
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self._device)
            text_features = self.model.encode_text(tokens)
            # L2-Normalisierung für Cosine Similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze().cpu().tolist()
    
    def encode_image(self, image: Union[str, Image.Image, bytes]) -> List[float]:
        """
        Berechnet Bild-Embedding.
        
        Args:
            image: Bild als Pfad (str), PIL.Image, oder Bytes
            
        Returns:
            Normalisierter Embedding-Vektor
        """
        # Bild laden falls nötig
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        
        # Sicherstellen, dass RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self._device)
            image_features = self.model.encode_image(image_tensor)
            # L2-Normalisierung
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.squeeze().cpu().tolist()
    
    def similarity(
        self, 
        text: str, 
        image: Union[str, Image.Image, bytes]
    ) -> float:
        """
        Berechnet Text-Bild-Ähnlichkeit (Cosine Similarity).
        
        Args:
            text: Beschreibungstext
            image: Zu vergleichendes Bild
            
        Returns:
            Similarity-Score zwischen 0 und 1
        """
        text_emb = self.encode_text(text)
        image_emb = self.encode_image(image)
        
        # Cosine Similarity (schon normalisiert)
        text_tensor = torch.tensor(text_emb)
        image_tensor = torch.tensor(image_emb)
        
        similarity = torch.dot(text_tensor, image_tensor).item()
        
        # Clamp auf 0-1 Range
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def rank_images(
        self, 
        text: str, 
        images: List[Union[str, Image.Image]]
    ) -> List[tuple]:
        """
        Rankt mehrere Bilder nach Ähnlichkeit zum Text.
        
        Args:
            text: Beschreibungstext
            images: Liste von Bildern
            
        Returns:
            Liste von (index, similarity) sortiert nach Ähnlichkeit (absteigend)
        """
        scores = []
        for i, image in enumerate(images):
            score = self.similarity(text, image)
            scores.append((i, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)


# Singleton-Instanz
_handler: Optional[BiomedCLIPHandler] = None


def get_handler() -> BiomedCLIPHandler:
    """Gibt Singleton-Instanz des Handlers zurück"""
    global _handler
    if _handler is None:
        _handler = BiomedCLIPHandler()
    return _handler
