"""
Scientific Prompt Templates für Stable Diffusion

Vordefinierte Templates für wissenschaftliche Visualisierungen.
Optimiert für biomedizinische und naturwissenschaftliche Darstellungen.
"""
from typing import Dict, Any, List, Optional


TEMPLATES: Dict[str, Dict[str, str]] = {
    "cell_diagram": {
        "base": (
            "scientific diagram of {cell_type} cell, labeled anatomical parts, "
            "textbook illustration style, white background, high detail, "
            "educational medical illustration, cross-section view"
        ),
        "negative": (
            "photo, realistic, blurry, text overlay, watermark, "
            "low quality, jpeg artifacts, cartoon style"
        ),
    },
    "molecular_structure": {
        "base": (
            "3D molecular structure of {molecule}, ball-and-stick model, "
            "scientific visualization, clean background, atoms colored by element, "
            "chemical bonds visible, publication quality render"
        ),
        "negative": (
            "cartoon, sketch, blurry, 2D flat, text, watermark, "
            "low resolution, artistic interpretation"
        ),
    },
    "anatomical": {
        "base": (
            "medical illustration of {organ}, anatomical cross-section, "
            "labeled diagram style, textbook quality, detailed anatomy, "
            "professional medical illustration, neutral background"
        ),
        "negative": (
            "photo, x-ray, blurry, gore, blood, realistic skin, "
            "low quality, cartoon, artistic"
        ),
    },
    "process_flow": {
        "base": (
            "scientific flowchart showing {process}, clear arrows between steps, "
            "labeled boxes, infographic style, clean design, "
            "educational diagram, professional presentation quality"
        ),
        "negative": (
            "photo, 3D render, complex background, artistic, "
            "hand-drawn, messy, cluttered"
        ),
    },
    "microscopy": {
        "base": (
            "microscopy image of {specimen}, {staining} staining, "
            "high magnification, scientific imaging, clear cellular structures, "
            "research quality, {microscope_type} microscopy style"
        ),
        "negative": (
            "blurry, out of focus, artistic, illustration, "
            "low resolution, noise, artifacts"
        ),
    },
    "protein_structure": {
        "base": (
            "protein structure visualization of {protein}, {representation} representation, "
            "colored by {coloring}, scientific 3D render, "
            "publication quality, PyMOL style visualization"
        ),
        "negative": (
            "cartoon character, low poly, blurry, artistic interpretation, "
            "2D flat, sketch"
        ),
    },
    "pathway_diagram": {
        "base": (
            "biological pathway diagram showing {pathway}, "
            "interconnected nodes, reaction arrows, enzyme labels, "
            "KEGG style visualization, scientific accuracy, clean layout"
        ),
        "negative": (
            "photo, 3D, artistic, hand-drawn, messy, "
            "complex background, decorative"
        ),
    },
    "tissue_section": {
        "base": (
            "histological section of {tissue}, {staining} stain, "
            "microscopy view, clear cellular organization, "
            "pathology textbook quality, educational annotation"
        ),
        "negative": (
            "blurry, artistic, illustration style, cartoon, "
            "low magnification, dark, overexposed"
        ),
    },
}

# Standard-Werte für optionale Template-Variablen
DEFAULT_VALUES: Dict[str, Dict[str, str]] = {
    "microscopy": {
        "staining": "H&E",
        "microscope_type": "light",
    },
    "protein_structure": {
        "representation": "ribbon",
        "coloring": "secondary structure",
    },
    "tissue_section": {
        "staining": "hematoxylin and eosin",
    },
}


def get_template(
    name: str, 
    variables: Dict[str, Any],
    custom_suffix: Optional[str] = None
) -> Dict[str, str]:
    """
    Füllt Template mit Variablen.
    
    Args:
        name: Template-Name (z.B. "cell_diagram")
        variables: Dict mit Platzhaltern (z.B. {"cell_type": "neuron"})
        custom_suffix: Optionaler Zusatz zum Base-Prompt
    
    Returns:
        {"prompt": str, "negative_prompt": str}
    
    Raises:
        ValueError: Wenn Template nicht existiert
    
    Example:
        >>> get_template("cell_diagram", {"cell_type": "neuron"})
        {"prompt": "scientific diagram of neuron cell...", "negative_prompt": "..."}
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template: {name}. Available: {available}")
    
    template = TEMPLATES[name]
    
    # Merge mit Default-Werten
    merged_vars = {**DEFAULT_VALUES.get(name, {}), **variables}
    
    # Template ausfüllen
    try:
        base_prompt = template["base"].format(**merged_vars)
    except KeyError as e:
        raise ValueError(f"Missing variable {e} for template '{name}'")
    
    # Custom suffix hinzufügen
    if custom_suffix:
        base_prompt = f"{base_prompt}, {custom_suffix}"
    
    return {
        "prompt": base_prompt,
        "negative_prompt": template.get("negative", ""),
    }


def list_templates() -> List[str]:
    """Gibt Liste aller verfügbaren Template-Namen zurück"""
    return list(TEMPLATES.keys())


def get_template_info(name: str) -> Dict[str, Any]:
    """
    Gibt Informationen über ein Template zurück.
    
    Returns:
        {
            "name": str,
            "base_prompt": str,
            "negative_prompt": str,
            "required_variables": List[str],
            "default_values": Dict[str, str]
        }
    """
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}")
    
    template = TEMPLATES[name]
    base = template["base"]
    
    # Extrahiere Variablen aus Format-String
    import re
    variables = re.findall(r'\{(\w+)\}', base)
    
    return {
        "name": name,
        "base_prompt": base,
        "negative_prompt": template.get("negative", ""),
        "required_variables": variables,
        "default_values": DEFAULT_VALUES.get(name, {}),
    }
