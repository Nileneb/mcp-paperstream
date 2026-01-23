"""
Model Download Utility

Lädt ML-Modelle von HuggingFace herunter und speichert sie lokal.
Nutzt Pfade aus config.yaml für konsistente Speicherorte.
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from transformers import AutoTokenizer, AutoModel


def load_config() -> dict:
    """Lädt Konfiguration aus config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def download_biobert(
    save_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Lädt distil-biobert herunter und speichert lokal.
    
    Args:
        save_path: Zielverzeichnis (default: aus config.yaml)
        model_name: HuggingFace Modell-Name (default: aus config.yaml)
    
    Returns:
        Pfad zum gespeicherten Modell
    """
    config = load_config()
    biobert_config = config.get("models", {}).get("biobert", {})
    
    model_name = model_name or biobert_config.get("model_name", "nlpie/distil-biobert")
    save_path = save_path or biobert_config.get("path", "./models/biobert/distil-biobert")
    
    # Relativen Pfad zu absolutem Pfad konvertieren
    if not os.path.isabs(save_path):
        project_root = Path(__file__).parent.parent.parent.parent
        save_path = str(project_root / save_path)
    
    print(f"📥 Lade {model_name}...")
    
    # Download von HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Speichern
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print(f"✅ Gespeichert in {save_path}")
    return save_path


def download_biomedclip(
    save_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Lädt BiomedCLIP herunter und speichert lokal.
    
    Args:
        save_path: Zielverzeichnis
        model_name: HuggingFace Modell-Name
    
    Returns:
        Pfad zum gespeicherten Modell
    """
    try:
        import open_clip
    except ImportError:
        print("❌ open_clip nicht installiert. Installiere mit: pip install open_clip_torch")
        raise
    
    config = load_config()
    clip_config = config.get("models", {}).get("biomedclip", {})
    
    model_name = model_name or clip_config.get(
        "model_name", 
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    save_path = save_path or clip_config.get("path", "./models/biomedclip")
    
    if not os.path.isabs(save_path):
        project_root = Path(__file__).parent.parent.parent.parent
        save_path = str(project_root / save_path)
    
    print(f"📥 Lade {model_name}...")
    
    # Download und speichern (open_clip cached automatisch)
    model, preprocess = open_clip.create_model_from_pretrained(model_name)
    
    os.makedirs(save_path, exist_ok=True)
    # Hinweis: open_clip Modelle werden anders gespeichert
    print(f"✅ Modell geladen (gecached in ~/.cache/huggingface/)")
    print(f"   Für lokale Kopie: manuelles Kopieren aus Cache notwendig")
    
    return save_path


def download_all():
    """Lädt alle konfigurierten Modelle herunter"""
    print("🚀 Starte Download aller Modelle...\n")
    
    try:
        download_biobert()
    except Exception as e:
        print(f"❌ BioBERT Download fehlgeschlagen: {e}")
    
    print()
    
    try:
        download_biomedclip()
    except Exception as e:
        print(f"❌ BiomedCLIP Download fehlgeschlagen: {e}")
    
    print("\n✨ Download abgeschlossen!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model = sys.argv[1].lower()
        if model == "biobert":
            download_biobert()
        elif model == "biomedclip":
            download_biomedclip()
        elif model == "all":
            download_all()
        else:
            print(f"Unbekanntes Modell: {model}")
            print("Verfügbar: biobert, biomedclip, all")
    else:
        download_all()
