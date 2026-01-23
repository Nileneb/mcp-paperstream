"""
Handler-Module für mcp-paperstream

- biobert_handler: BioBERT/TinyBERT Tokenisierung & Embeddings
- biomedclip_handler: Text-Bild-Ähnlichkeit mit BiomedCLIP
- sd_api_client: Stable Diffusion WebUI API Client
- download_model: Modell-Download Utility
"""

from .biobert_handler import BioBERTHandler, get_handler as get_biobert_handler
from .sd_api_client import StableDiffusionClient, get_client as get_sd_client

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
    "StableDiffusionClient", 
    "get_sd_client",
    "BiomedCLIPHandler",
    "get_biomedclip_handler",
    "BIOMEDCLIP_AVAILABLE",
]
