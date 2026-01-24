"""
PDF Voxelizer - KISS Version
============================

Converts PDF pages to 3D voxel grids for Unity mesh generation.
Uses text density as height map.

Grid: 8x8x12 (X, Y, Z) = 768 voxels
- Matches BioBERT embedding dimension (768)
- X: horizontal position (8 columns)
- Y: height (8 layers based on text density)
- Z: vertical position (12 rows, top to bottom)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default grid dimensions (8x8x12 = 768 voxels = BioBERT dim)
GRID_X = 8   # Width (columns)
GRID_Y = 8   # Height (layers)
GRID_Z = 12  # Depth (rows)

# Voxel types based on content
VOXEL_EMPTY = 0
VOXEL_TEXT = 1
VOXEL_DENSE = 2  # Dense text area


def pdf_to_voxel_grid(
    pdf_path: Union[str, Path],
    page: int = 0,
    grid_size: tuple = (GRID_X, GRID_Y, GRID_Z)
) -> Dict[str, Any]:
    """
    Convert a PDF page to a voxel grid.
    
    Uses grayscale intensity (inverted) as height map:
    - White areas → empty
    - Text/dark areas → voxels with height proportional to darkness
    
    Args:
        pdf_path: Path to PDF file
        page: Page number (0-indexed)
        grid_size: Tuple of (X, Y, Z) dimensions
    
    Returns:
        {
            "grid_size": [X, Y, Z],
            "voxels": [[x, y, z, density], ...],
            "stats": {"total": int, "density_avg": float}
        }
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available")
        return _empty_grid(grid_size)
    
    if not PIL_AVAILABLE:
        logger.error("PIL/numpy not available")
        return _empty_grid(grid_size)
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return _empty_grid(grid_size)
    
    gx, gy, gz = grid_size
    
    try:
        doc = fitz.open(str(pdf_path))
        
        # Clamp page number
        page_num = min(page, len(doc) - 1)
        page_obj = doc[page_num]
        
        # Render page to pixmap (2x scale for better detail)
        mat = fitz.Matrix(2, 2)
        pix = page_obj.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to grayscale and resize to grid dimensions
        img_gray = img.convert("L")
        img_resized = img_gray.resize((gx, gz), Image.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(img_resized)
        
        # Invert: dark (text) becomes high values
        inverted = 255 - pixels
        
        # Normalize to [0, 1]
        normalized = inverted / 255.0
        
        # Calculate heights (max height = gy)
        heights = (normalized * gy).astype(int)
        
        # Build voxel list
        voxels = []
        density_sum = 0.0
        
        for x in range(gx):
            for z in range(gz):
                h = int(heights[z, x])
                density = float(normalized[z, x])
                
                # Create voxels from y=0 to y=h
                for y in range(h):
                    # Density decreases with height
                    voxel_density = density * (1.0 - y / gy * 0.3)
                    voxels.append([x, y, z, round(float(voxel_density), 2)])
                    density_sum += voxel_density
        
        doc.close()
        
        avg_density = density_sum / len(voxels) if voxels else 0.0
        
        return {
            "grid_size": list(grid_size),
            "page": page_num,
            "voxels": voxels,
            "stats": {
                "total": len(voxels),
                "density_avg": round(avg_density, 3),
                "fill_ratio": round(len(voxels) / (gx * gy * gz), 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to voxelize PDF: {e}")
        return _empty_grid(grid_size)


def pdf_to_voxel_grid_colored(
    pdf_path: Union[str, Path],
    page: int = 0,
    grid_size: tuple = (GRID_X, GRID_Y, GRID_Z)
) -> Dict[str, Any]:
    """
    Convert PDF page to colored voxel grid.
    
    Includes RGB color information for each voxel.
    
    Returns:
        {
            "grid_size": [X, Y, Z],
            "voxels": [[x, y, z, density, r, g, b], ...],
            "stats": {...}
        }
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        return _empty_grid(grid_size)
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return _empty_grid(grid_size)
    
    gx, gy, gz = grid_size
    
    try:
        doc = fitz.open(str(pdf_path))
        page_num = min(page, len(doc) - 1)
        page_obj = doc[page_num]
        
        mat = fitz.Matrix(2, 2)
        pix = page_obj.get_pixmap(matrix=mat, alpha=False)
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize color image
        img_resized = img.resize((gx, gz), Image.LANCZOS)
        pixels_rgb = np.array(img_resized)
        
        # Grayscale for height calculation
        img_gray = img_resized.convert("L")
        pixels_gray = np.array(img_gray)
        
        inverted = 255 - pixels_gray
        normalized = inverted / 255.0
        heights = (normalized * gy).astype(int)
        
        voxels = []
        
        for x in range(gx):
            for z in range(gz):
                h = int(heights[z, x])
                density = float(normalized[z, x])
                r, g, b = pixels_rgb[z, x]
                
                for y in range(h):
                    voxel_density = density * (1.0 - y / gy * 0.3)
                    voxels.append([
                        x, y, z, 
                        round(float(voxel_density), 2),
                        int(r), int(g), int(b)
                    ])
        
        doc.close()
        
        return {
            "grid_size": list(grid_size),
            "page": page_num,
            "voxels": voxels,
            "colored": True,
            "stats": {
                "total": len(voxels),
                "fill_ratio": round(len(voxels) / (gx * gy * gz), 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to voxelize PDF (colored): {e}")
        return _empty_grid(grid_size)


def _empty_grid(grid_size: tuple) -> Dict[str, Any]:
    """Return empty voxel grid"""
    return {
        "grid_size": list(grid_size),
        "voxels": [],
        "stats": {"total": 0, "density_avg": 0.0, "fill_ratio": 0.0}
    }


def section_to_voxel(
    section_text: str,
    section_type: str = "body",
    grid_size: tuple = (GRID_X, GRID_Y, GRID_Z)
) -> Dict[str, Any]:
    """
    Convert a text section to voxel representation.
    
    Simple text-based voxelization:
    - Word count determines density
    - Section type affects height pattern
    
    Args:
        section_text: The section text content
        section_type: Type of section (title, abstract, methods, etc.)
        grid_size: Grid dimensions
    
    Returns:
        Voxel grid dict
    """
    if not section_text:
        return _empty_grid(grid_size)
    
    gx, gy, gz = grid_size
    words = section_text.split()
    word_count = len(words)
    
    # Density based on word count (normalize to 0-1)
    base_density = min(1.0, word_count / 500)  # 500 words = max density
    
    # Height multiplier based on section type
    height_mult = {
        "title": 0.8,
        "abstract": 0.9,
        "introduction": 0.7,
        "methods": 0.85,
        "results": 0.9,
        "discussion": 0.75,
        "conclusion": 0.7,
        "references": 0.5,
    }.get(section_type.lower(), 0.7)
    
    # Generate voxels with gradual falloff
    voxels = []
    max_height = int(gy * height_mult)
    
    for x in range(gx):
        for z in range(gz):
            # Add some variation based on position
            variation = 0.8 + 0.4 * np.sin(x * 0.5) * np.cos(z * 0.5)
            local_density = base_density * variation
            local_height = int(max_height * local_density)
            
            for y in range(local_height):
                d = local_density * (1.0 - y / gy * 0.3)
                voxels.append([x, y, z, round(d, 2)])
    
    return {
        "grid_size": list(grid_size),
        "section_type": section_type,
        "word_count": word_count,
        "voxels": voxels,
        "stats": {
            "total": len(voxels),
            "base_density": round(base_density, 3),
            "fill_ratio": round(len(voxels) / (gx * gy * gz), 3)
        }
    }
