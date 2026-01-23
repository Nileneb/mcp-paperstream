"""
Prompt-Module für mcp-paperstream

- scientific_templates: SD Prompt-Templates für wissenschaftliche Visualisierungen
- term_mappings: Wissenschaftliche Begriffe → visuelle Beschreibungen
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .scientific_templates import (
    get_template,
    list_templates,
    get_template_info,
    TEMPLATES,
    DEFAULT_VALUES,
)


def load_term_mappings() -> Dict[str, Any]:
    """Lädt term_mappings.json"""
    mappings_path = Path(__file__).parent / "term_mappings.json"
    if mappings_path.exists():
        with open(mappings_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_visual_terms(
    term: str,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sucht visuelle Beschreibungen für einen wissenschaftlichen Begriff.
    
    Args:
        term: Wissenschaftlicher Begriff (z.B. "neuron", "DNA")
        category: Optionale Kategorie (z.B. "cell_types", "molecules")
    
    Returns:
        Dict mit synonyms, visual_descriptors, etc. oder leeres Dict
    """
    mappings = load_term_mappings()
    
    # Wenn Kategorie angegeben, nur dort suchen
    if category and category in mappings:
        return mappings[category].get(term, {})
    
    # Sonst alle Kategorien durchsuchen
    for cat_name, cat_data in mappings.items():
        if cat_name.startswith("_"):  # Meta-Felder überspringen
            continue
        if isinstance(cat_data, dict) and term in cat_data:
            return cat_data[term]
    
    return {}


def expand_scientific_term(term: str) -> List[str]:
    """
    Erweitert einen Begriff um Synonyme und visuelle Beschreibungen.
    
    Args:
        term: Wissenschaftlicher Begriff
    
    Returns:
        Liste von verwandten Begriffen für Prompt-Erweiterung
    """
    info = get_visual_terms(term)
    if not info:
        return [term]
    
    expanded = [term]
    expanded.extend(info.get("synonyms", []))
    expanded.extend(info.get("visual_descriptors", []))
    
    return expanded


__all__ = [
    "get_template",
    "list_templates", 
    "get_template_info",
    "TEMPLATES",
    "DEFAULT_VALUES",
    "load_term_mappings",
    "get_visual_terms",
    "expand_scientific_term",
]
