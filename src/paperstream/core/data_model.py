"""
Unified Data Model for PaperStream
==================================
Version: 0.2.0 (DATAMODEL.md Contract)

This module defines the canonical data structures that ensure Papers and Rules
are processed identically between Python backend and Unity frontend.

HIERARCHY:
----------
Text (Paper/Rule) → Embedding (768-dim BioBERT) → Chunk (container) → Voxels (visible)

CRITICAL INVARIANT:
-------------------
embedding_to_voxel_grid() MUST produce identical results in Python and C#!
The Unity implementation is in: ValidationGame/Assets/Scripts/Voxel/EmbeddingToVoxel.cs

VOXEL MAPPING:
--------------
embedding[i] → voxel[x,y,z] where:
    i = x + y*8 + z*64
    x ∈ [0,7], y ∈ [0,7], z ∈ [0,11]
    
To iterate correctly:
    for z in range(12):      # Outer loop
        for y in range(8):   # Middle loop
            for x in range(8): # Inner loop
                i = x + y*8 + z*64
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import base64
import json


# ============================================================================
# CONSTANTS (CONTRACT VALUES - DO NOT CHANGE)
# ============================================================================

VOXEL_GRID_X = 8   # Width
VOXEL_GRID_Y = 8   # Height (layers)
VOXEL_GRID_Z = 12  # Depth
VOXEL_TOTAL = VOXEL_GRID_X * VOXEL_GRID_Y * VOXEL_GRID_Z  # 768

DEFAULT_VOXEL_THRESHOLD = 0.3

# Section colors (RGB 0-1) - matched with Unity ChunkColor
SECTION_COLORS = {
    "abstract":     (0.2, 0.6, 0.9),   # Blue - #3399E6
    "introduction": (0.3, 0.8, 0.3),   # Green - #4DCC4D
    "methods":      (0.9, 0.7, 0.2),   # Yellow/Orange - #E6B233
    "results":      (0.8, 0.3, 0.3),   # Red - #CC4D4D
    "discussion":   (0.7, 0.4, 0.9),   # Purple - #B266E6
    "conclusion":   (0.5, 0.5, 0.5),   # Gray
    "references":   (0.4, 0.4, 0.4),   # Dark gray
    "positive":     (0.2, 0.9, 0.3),   # Bright green - #33E64D
    "negative":     (0.9, 0.2, 0.2),   # Bright red - #E63333
    "other":        (0.5, 0.5, 0.5),   # Default gray
}

LAYOUT_CHAIN = "chain"
LAYOUT_DIPOLE = "dipole"
CONNECTION_SEQUENTIAL = "sequential"
CONNECTION_POLAR = "polar"


# ============================================================================
# CORE TRANSFORMATION FUNCTIONS (MUST MATCH C# EXACTLY)
# ============================================================================

def enhance_visual_contrast(embedding: np.ndarray) -> np.ndarray:
    """
    Verstärkt Unterschiede für bessere visuelle Erkennbarkeit.
    
    MUST MATCH C# EmbeddingToVoxel.EnhanceVisualContrast() EXACTLY!
    
    Algorithm:
    1. centered = emb - mean(emb)
    2. amplified = tanh(centered * 2.0)
    3. result = (amplified + 1.0) / 2.0
    
    Args:
        embedding: Input array (typically 768-dim BioBERT)
    
    Returns:
        Enhanced values in [0, 1] range
    """
    emb = np.array(embedding, dtype=np.float32).flatten()
    
    if len(emb) == 0:
        return np.zeros(VOXEL_TOTAL, dtype=np.float32)
    
    # 1. Center around mean
    mean = emb.mean()
    centered = emb - mean
    
    # 2. Amplify with tanh
    amplified = np.tanh(centered * 2.0)
    
    # 3. Rescale to [0, 1]
    result = (amplified + 1.0) / 2.0
    
    # Pad to 768 if needed
    if len(result) < VOXEL_TOTAL:
        padded = np.full(VOXEL_TOTAL, 0.5, dtype=np.float32)
        padded[:len(result)] = result
        return padded
    
    return result[:VOXEL_TOTAL]


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalizes array to [0, 1] range.
    
    MUST MATCH C# EmbeddingToVoxel.Normalize() EXACTLY!
    """
    emb = np.array(embedding, dtype=np.float32).flatten()
    
    if len(emb) == 0:
        return np.zeros(VOXEL_TOTAL, dtype=np.float32)
    
    emb_min = emb.min()
    emb_max = emb.max()
    
    denom = emb_max - emb_min
    if denom < 1e-8:
        denom = 1e-8
    
    normalized = (emb - emb_min) / denom
    return normalized


def embedding_to_voxel_grid(
    embedding: np.ndarray,
    threshold: float = DEFAULT_VOXEL_THRESHOLD,
    enhance_contrast: bool = True
) -> List['Voxel']:
    """
    Convert 768-dim embedding to voxel positions.
    
    MUST MATCH C# EmbeddingToVoxel.ConvertToPositions() EXACTLY!
    
    Mapping: i = x + y*8 + z*64
    
    Iteration order (important for consistency):
        for z in range(12):
            for y in range(8):
                for x in range(8):
    
    Args:
        embedding: 768-dim BioBERT embedding
        threshold: Minimum value to create voxel (default 0.3)
        enhance_contrast: Apply visual enhancement (default True)
    
    Returns:
        List of Voxel objects
    """
    emb = np.array(embedding, dtype=np.float32).flatten()
    
    # Pad or truncate to exactly 768
    if len(emb) != VOXEL_TOTAL:
        if len(emb) > VOXEL_TOTAL:
            emb = emb[:VOXEL_TOTAL]
        else:
            padded = np.zeros(VOXEL_TOTAL, dtype=np.float32)
            padded[:len(emb)] = emb
            emb = padded
    
    # Apply transformation
    if enhance_contrast:
        processed = enhance_visual_contrast(emb)
    else:
        processed = emb
    
    # Normalize to [0, 1]
    normalized = normalize_embedding(processed)
    
    # Create voxels - EXACT SAME ITERATION ORDER AS C#
    voxels = []
    for z in range(VOXEL_GRID_Z):
        for y in range(VOXEL_GRID_Y):
            for x in range(VOXEL_GRID_X):
                i = x + y * VOXEL_GRID_X + z * (VOXEL_GRID_X * VOXEL_GRID_Y)
                value = float(normalized[i])
                
                if value >= threshold:
                    voxels.append(Voxel(x=x, y=y, z=z, value=value))
    
    return voxels


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Voxel:
    """Single voxel in the 8x8x12 grid"""
    x: int
    y: int
    z: int
    value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z, "v": round(self.value, 4)}
    
    def to_list(self) -> List:
        """[x, y, z, value] format for JSON transport"""
        return [self.x, self.y, self.z, round(self.value, 4)]


@dataclass
class VoxelGrid:
    """8x8x12 voxel grid representing a 768-dim embedding"""
    voxels: List[Voxel]
    grid_size: Tuple[int, int, int] = (VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z)
    
    @classmethod
    def from_embedding(
        cls,
        embedding: np.ndarray,
        threshold: float = DEFAULT_VOXEL_THRESHOLD,
        enhance_contrast: bool = True
    ) -> "VoxelGrid":
        """Create grid from embedding using canonical transformation"""
        voxels = embedding_to_voxel_grid(embedding, threshold, enhance_contrast)
        return cls(voxels=voxels)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export matching DATAMODEL.md contract"""
        return {
            "grid_size": list(self.grid_size),
            "voxels": [v.to_list() for v in self.voxels],
            "voxel_count": len(self.voxels),
            "fill_ratio": round(len(self.voxels) / VOXEL_TOTAL, 3)
        }


@dataclass
class Chunk:
    """Container for embedding + voxels"""
    chunk_id: int
    chunk_type: str
    text_preview: str
    embedding_b64: str
    voxels: VoxelGrid
    color: Dict[str, float]
    position: Dict[str, float]
    connects_to: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text_preview": self.text_preview,
            "embedding_b64": self.embedding_b64,
            "voxels": self.voxels.to_dict(),
            "color": self.color,
            "position": self.position,
            "connects_to": self.connects_to
        }


@dataclass
class Molecule:
    """Collection of connected chunks (Paper or Rule)"""
    molecule_id: str
    molecule_type: str
    title: str
    chunks: List[Chunk]
    layout: str
    connection_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "molecule_type": self.molecule_type,
            "title": self.title,
            "chunks": [c.to_dict() for c in self.chunks],
            "chunks_count": len(self.chunks),
            "molecule_config": {
                "embedding_dim": VOXEL_TOTAL,
                "voxel_grid": [VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z],
                "layout": self.layout,
                "connection_type": self.connection_type,
                "scale": 1.0
            }
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def embedding_to_base64(embedding: np.ndarray) -> str:
    """Convert numpy embedding to base64 (float32 bytes)"""
    return base64.b64encode(
        np.array(embedding, dtype=np.float32).tobytes()
    ).decode('utf-8')


def base64_to_embedding(b64_str: str) -> np.ndarray:
    """Convert base64 back to numpy embedding"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)


def create_chunk(
    chunk_id: int,
    chunk_type: str,
    text: str,
    embedding: np.ndarray,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    connects_to: Optional[List[int]] = None,
    threshold: float = DEFAULT_VOXEL_THRESHOLD
) -> Chunk:
    """Create a Chunk from text + embedding"""
    color_rgb = SECTION_COLORS.get(chunk_type, SECTION_COLORS["other"])
    
    return Chunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        text_preview=text[:500] if text else "",
        embedding_b64=embedding_to_base64(embedding),
        voxels=VoxelGrid.from_embedding(embedding, threshold=threshold),
        color={"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
        position={"x": position[0], "y": position[1], "z": position[2]},
        connects_to=connects_to or []
    )


def create_paper_molecule(
    paper_id: str,
    title: str,
    sections: List[Dict[str, Any]],
    threshold: float = DEFAULT_VOXEL_THRESHOLD
) -> Molecule:
    """Create Paper Molecule (chain of section chunks)"""
    chunks = []
    for idx, section in enumerate(sections):
        chunk = create_chunk(
            chunk_id=idx,
            chunk_type=section["name"],
            text=section["text"],
            embedding=section["embedding"],
            position=(idx * 2.0, 0.0, 0.0),
            connects_to=[idx + 1] if idx < len(sections) - 1 else [],
            threshold=threshold
        )
        chunks.append(chunk)
    
    return Molecule(
        molecule_id=paper_id,
        molecule_type="paper",
        title=title,
        chunks=chunks,
        layout=LAYOUT_CHAIN,
        connection_type=CONNECTION_SEQUENTIAL
    )


def create_rule_molecule(
    rule_id: str,
    question: str,
    pos_embedding: np.ndarray,
    neg_embedding: Optional[np.ndarray] = None,
    pos_text: str = "",
    neg_text: str = "",
    threshold: float = DEFAULT_VOXEL_THRESHOLD
) -> Molecule:
    """Create Rule Molecule (dipole of pos/neg chunks)"""
    chunks = []
    
    pos_chunk = create_chunk(
        chunk_id=0,
        chunk_type="positive",
        text=pos_text,
        embedding=pos_embedding,
        position=(0.0, 0.0, 0.0),
        connects_to=[1] if neg_embedding is not None else [],
        threshold=threshold
    )
    chunks.append(pos_chunk)
    
    if neg_embedding is not None:
        neg_chunk = create_chunk(
            chunk_id=1,
            chunk_type="negative",
            text=neg_text,
            embedding=neg_embedding,
            position=(3.0, 0.0, 0.0),
            connects_to=[],
            threshold=threshold
        )
        chunks.append(neg_chunk)
    
    return Molecule(
        molecule_id=rule_id,
        molecule_type="rule",
        title=question,
        chunks=chunks,
        layout=LAYOUT_DIPOLE,
        connection_type=CONNECTION_POLAR
    )


# ============================================================================
# TEST / VERIFICATION
# ============================================================================

def verify_determinism(embedding: np.ndarray, runs: int = 3) -> bool:
    """
    Verify that embedding_to_voxel_grid is deterministic.
    
    Returns True if all runs produce identical output.
    """
    results = []
    for _ in range(runs):
        voxels = embedding_to_voxel_grid(embedding)
        result = [(v.x, v.y, v.z, round(v.value, 6)) for v in voxels]
        results.append(tuple(result))
    
    return len(set(results)) == 1


if __name__ == "__main__":
    # Quick sanity check
    test_emb = np.random.randn(768).astype(np.float32)
    
    print("Testing determinism...")
    assert verify_determinism(test_emb), "FAILED: Non-deterministic output!"
    print("✓ Determinism verified")
    
    print("\nGenerating test voxels...")
    voxels = embedding_to_voxel_grid(test_emb, threshold=0.3)
    print(f"✓ Generated {len(voxels)} voxels (threshold=0.3)")
    print(f"  Fill ratio: {len(voxels)/768:.1%}")
    
    print("\nCreating test molecule...")
    mol = create_rule_molecule(
        rule_id="test_rule",
        question="Is this a test?",
        pos_embedding=test_emb,
        neg_embedding=np.random.randn(768).astype(np.float32),
        pos_text="positive phrases",
        neg_text="negative phrases"
    )
    print(f"✓ Created molecule with {len(mol.chunks)} chunks")
    
    print("\nAll tests passed!")
