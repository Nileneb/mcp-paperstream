"""
BioBERT Handler für Tokenisierung und Embedding-Berechnung

Verwendet distil-biobert für effiziente Verarbeitung auf Server-Seite.
TinyBERT-kompatible Ausgabe für IoT-Worker-Verteilung.
"""
import os
from typing import List, Tuple, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
import yaml


def load_config() -> dict:
    """Lädt Konfiguration aus config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class BioBERTHandler:
    """Handler für distil-biobert Modell"""
    
    def __init__(self, model_path: Optional[str] = None):
        config = load_config()
        self.model_path = model_path or config.get("models", {}).get("biobert", {}).get(
            "path", "./models/biobert/distil-biobert"
        )
        self.model_name = config.get("models", {}).get("biobert", {}).get(
            "model_name", "nlpie/distil-biobert"
        )
        self._tokenizer = None
        self._model = None
    
    @property
    def tokenizer(self):
        """Lazy-load Tokenizer"""
        if self._tokenizer is None:
            # Versuche lokales Modell, sonst HuggingFace
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    @property
    def model(self):
        """Lazy-load Modell"""
        if self._model is None:
            try:
                self._model = AutoModel.from_pretrained(self.model_path)
            except Exception:
                self._model = AutoModel.from_pretrained(self.model_name)
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
        encoded = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=False
        )
        token_ids = encoded["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens, token_ids
    
    def embed(
        self, 
        text: str, 
        layer_range: Optional[Tuple[int, int]] = None,
        use_cache: bool = True,
        cache_id: Optional[str] = None
    ) -> List[float]:
        """
        Berechnet Embeddings für Text.
        
        Args:
            text: Eingabetext
            layer_range: Optional (start, end) für partielle Layer-Ausgabe
            use_cache: Check Qdrant cache first (default True)
            cache_id: Optional ID for caching (default: text hash)
            
        Returns:
            Embedding-Vektor als Liste von Floats
        """
        # Try Qdrant cache first
        if use_cache:
            try:
                from ..db.vector_store import get_paper_embedding
                lookup_id = cache_id or f"text_{self._get_text_hash(text)}"
                cached = get_paper_embedding(lookup_id)
                if cached:
                    return cached
            except Exception:
                pass  # Qdrant not available, compute normally
        
        with torch.no_grad():
            encoded = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            outputs = self.model(**encoded, output_hidden_states=True)
            
            if layer_range:
                # Nur bestimmte Layer ausgeben (für IoT-Verteilung)
                start, end = layer_range
                hidden_states = outputs.hidden_states[start:end]
                # Mean über Layer und Tokens
                embedding = torch.stack(hidden_states).mean(dim=[0, 2])
            else:
                # Letzter Layer, Mean über Tokens (Standard BERTScore)
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            return embedding.squeeze().tolist()
    
    def embed_tokens(
        self,
        token_ids: List[int],
        layer_range: Optional[Tuple[int, int]] = None
    ) -> List[List[float]]:
        """
        Berechnet Token-Level Embeddings (für BERTScore).
        
        Args:
            token_ids: Liste von Token-IDs
            layer_range: Optional (start, end) für partielle Layer-Ausgabe
            
        Returns:
            Liste von Embedding-Vektoren (ein Vektor pro Token)
        """
        with torch.no_grad():
            input_ids = torch.tensor([token_ids])
            attention_mask = torch.ones_like(input_ids)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if layer_range:
                start, end = layer_range
                hidden_states = outputs.hidden_states[start:end]
                # Mean über Layer, behalte Token-Dimension
                embeddings = torch.stack(hidden_states).mean(dim=0)
            else:
                embeddings = outputs.last_hidden_state
            
            return embeddings.squeeze(0).tolist()


# Singleton für einfachen Import
_handler: Optional[BioBERTHandler] = None


def get_handler() -> BioBERTHandler:
    """Gibt Singleton-Instanz des Handlers zurück"""
    global _handler
    if _handler is None:
        _handler = BioBERTHandler()
    return _handler
