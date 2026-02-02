"""
Handler-Module für mcp-paperstream

- biobert_handler: BioBERT/DistilBioBERT Tokenisierung & Embeddings (768-dim)
- biomedclip_handler: Text-Bild-Ähnlichkeit mit BiomedCLIP (optional)
- download_model: Modell-Download Utility
"""

from .biobert_handler import BioBERTHandler, get_handler as get_biobert_handler

# BiomedCLIP ist optional (benötigt open_clip)
try:
    from .biomedclip_handler import BiomedCLIPHandler, get_handler as get_biomedclip_handler
    BIOMEDCLIP_AVAILABLE = True
except ImportError:
    BIOMEDCLIP_AVAILABLE = False
    BiomedCLIPHandler = None  # type: ignore
    get_biomedclip_handler = None  # type: ignore

__all__ = [
    "BioBERTHandler",
    "get_biobert_handler",
    "BiomedCLIPHandler",
    "get_biomedclip_handler",
    "BIOMEDCLIP_AVAILABLE",
]
