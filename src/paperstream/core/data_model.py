"""
Unified Data Model for PaperStream
==================================

This module defines the canonical data structures that ensure Papers and Rules
are processed identically:

HIERARCHY:
----------
Text (Paper/Rule) → Embedding (768-dim BioBERT) → Chunk (container) → Voxels (visible)

DEFINITIONS:
------------
1. TEXT: Source content
   - Paper: PDF sections (abstract, methods, results, etc.)
   - Rule: Positive/negative phrase lists

2. EMBEDDING: 768-dimensional BioBERT vector
   - Semantic representation of text
   - Base64 encoded for transport: `embedding_b64`

3. CHUNK: Container for embeddings
   - Unity: Invisible cube at (0,0,0) local space
   - Contains: embedding + voxels + metadata
   - Paper has N chunks (one per section)
   - Rule has 2 chunks (positive + negative)

4. VOXELS: 8x8x12 grid (768 values = embedding reshaped)
   - Unity: Visible cubes for interaction
   - Each voxel = one embedding dimension visualized
   - Grid dimensions match BioBERT embedding size

WIRE/CONNECTIONS:
-----------------
- Chunks are connected with "wires/lanes" showing relationships
- Paper: abstract → intro → methods → results → discussion (sequential)
- Rule: positive ↔ negative (polar/dipole)

UNITY RENDERING:
----------------
1. Create invisible container (Chunk) at world position
2. For each voxel in chunk.voxels: spawn visible cube at local position
3. Draw wires between connected chunks using chunk.connects_to
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import base64
import json


# ============================================================================
# CONSTANTS
# ============================================================================

# Voxel grid dimensions (8 * 8 * 12 = 768 = BioBERT embedding size)
VOXEL_GRID_X = 8   # Width
VOXEL_GRID_Y = 8   # Height (layers)
VOXEL_GRID_Z = 12  # Depth

# Section colors (RGB 0-1)
SECTION_COLORS = {
    "abstract":     (0.2, 0.6, 0.9),   # Blue
    "introduction": (0.3, 0.8, 0.3),   # Green
    "methods":      (0.9, 0.7, 0.2),   # Yellow/Orange
    "results":      (0.8, 0.3, 0.3),   # Red
    "discussion":   (0.7, 0.4, 0.9),   # Purple
    "conclusion":   (0.5, 0.5, 0.5),   # Gray
    "references":   (0.4, 0.4, 0.4),   # Dark gray
    "positive":     (0.2, 0.9, 0.3),   # Bright green (rules)
    "negative":     (0.9, 0.2, 0.2),   # Bright red (rules)
    "other":        (0.5, 0.5, 0.5),   # Default gray
}

# Layout types
LAYOUT_CHAIN = "chain"      # Sequential (papers)
LAYOUT_DIPOLE = "dipole"    # Two poles (rules)

# Connection types
CONNECTION_SEQUENTIAL = "sequential"  # One after another
CONNECTION_POLAR = "polar"            # Opposing forces


# ============================================================================
# CORE DATA CLASSES
# ============================================================================

@dataclass
class Voxel:
    """Single voxel in the 8x8x12 grid"""
    x: int
    y: int
    z: int
    value: float  # 0-1 normalized embedding value
    
    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z, "v": round(self.value, 4)}
    
    def to_list(self) -> List:
        return [self.x, self.y, self.z, round(self.value, 4)]


@dataclass
class VoxelGrid:
    """
    8x8x12 voxel grid representing a 768-dim embedding.
    
    Unity renders this as visible cubes within a Chunk container.
    """
    voxels: List[Voxel]
    grid_size: tuple = (VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z)
    
    @classmethod
    def from_embedding(cls, embedding: np.ndarray, threshold: float = 0.1) -> "VoxelGrid":
        """
        Convert 768-dim embedding to voxel grid.
        
        Args:
            embedding: 768-dim BioBERT embedding
            threshold: Minimum value to create a voxel (0-1)
        
        Returns:
            VoxelGrid with active voxels
        """
        # Ensure correct size
        emb = np.array(embedding, dtype=np.float32).flatten()
        if len(emb) != 768:
            # Interpolate to 768 if needed
            indices = np.linspace(0, len(emb) - 1, 768).astype(int)
            emb = emb[indices]
        
        # Normalize to 0-1
        emb_min, emb_max = emb.min(), emb.max()
        if emb_max > emb_min:
            normalized = (emb - emb_min) / (emb_max - emb_min)
        else:
            normalized = np.zeros(768)
        
        # Reshape to 8x8x12
        grid_3d = normalized.reshape((VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z))
        
        # Create voxels above threshold
        voxels = []
        for x in range(VOXEL_GRID_X):
            for y in range(VOXEL_GRID_Y):
                for z in range(VOXEL_GRID_Z):
                    value = float(grid_3d[x, y, z])
                    if value >= threshold:
                        voxels.append(Voxel(x=x, y=y, z=z, value=value))
        
        return cls(voxels=voxels)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_size": list(self.grid_size),
            "voxels": [v.to_list() for v in self.voxels],
            "voxel_count": len(self.voxels),
            "fill_ratio": round(len(self.voxels) / (8*8*12), 3)
        }
    
    def to_3d_array(self) -> List[List[List[float]]]:
        """Return as nested 3D array for Unity"""
        grid = np.zeros((VOXEL_GRID_X, VOXEL_GRID_Y, VOXEL_GRID_Z))
        for v in self.voxels:
            grid[v.x, v.y, v.z] = v.value
        return grid.tolist()


@dataclass
class Chunk:
    """
    Container for an embedding - the building block of molecules.
    
    In Unity:
    - Chunk = invisible cube at position, defines local origin (0,0,0)
    - Voxels inside = visible cubes relative to chunk position
    - Wires connect chunks via connects_to
    """
    chunk_id: int
    chunk_type: str  # "abstract", "methods", "positive", "negative", etc.
    text_preview: str  # First ~500 chars of source text
    embedding_b64: str  # 768-dim as base64 float32
    voxels: VoxelGrid  # 8x8x12 voxel representation
    color: Dict[str, float]  # {"r": 0-1, "g": 0-1, "b": 0-1}
    position: Dict[str, float]  # {"x": float, "y": float, "z": float}
    connects_to: List[int]  # IDs of connected chunks
    
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
    """
    A molecule is a collection of connected chunks.
    
    - Paper Molecule: Chain of section chunks (abstract → intro → methods → ...)
    - Rule Molecule: Dipole with positive/negative chunks
    """
    molecule_id: str
    molecule_type: str  # "paper" or "rule"
    title: str
    chunks: List[Chunk]
    layout: str  # "chain" or "dipole"
    connection_type: str  # "sequential" or "polar"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "molecule_type": self.molecule_type,
            "title": self.title,
            "chunks": [c.to_dict() for c in self.chunks],
            "chunks_count": len(self.chunks),
            "molecule_config": {
                "embedding_dim": 768,
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
    """Convert numpy embedding to base64 string"""
    return base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode('utf-8')


def base64_to_embedding(b64_str: str) -> np.ndarray:
    """Convert base64 string back to numpy embedding"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)


def create_chunk_from_embedding(
    chunk_id: int,
    chunk_type: str,
    text: str,
    embedding: np.ndarray,
    position_x: float = 0.0,
    connects_to: Optional[List[int]] = None,
    voxel_threshold: float = 0.1
) -> Chunk:
    """
    Create a standardized Chunk from text + embedding.
    
    This is the CANONICAL way to create chunks for both Papers and Rules.
    """
    color_rgb = SECTION_COLORS.get(chunk_type, SECTION_COLORS["other"])
    
    return Chunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        text_preview=text[:500] if text else "",
        embedding_b64=embedding_to_base64(embedding),
        voxels=VoxelGrid.from_embedding(embedding, threshold=voxel_threshold),
        color={"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
        position={"x": position_x, "y": 0.0, "z": 0.0},
        connects_to=connects_to or []
    )


def create_paper_molecule(
    paper_id: str,
    title: str,
    sections: List[Dict[str, Any]]
) -> Molecule:
    """
    Create Paper Molecule from sections.
    
    Args:
        paper_id: Paper identifier
        title: Paper title
        sections: List of {"name": str, "text": str, "embedding": np.ndarray}
    
    Returns:
        Molecule with chain layout
    """
    chunks = []
    for idx, section in enumerate(sections):
        chunk = create_chunk_from_embedding(
            chunk_id=idx,
            chunk_type=section["name"],
            text=section["text"],
            embedding=section["embedding"],
            position_x=idx * 2.0,  # Space chunks along X
            connects_to=[idx + 1] if idx < len(sections) - 1 else []
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
    pos_phrases: List[str],
    neg_phrases: List[str],
    pos_embedding: np.ndarray,
    neg_embedding: Optional[np.ndarray] = None
) -> Molecule:
    """
    Create Rule Molecule with positive/negative dipole.
    
    Args:
        rule_id: Rule identifier
        question: The validation question
        pos_phrases: Positive indicator phrases
        neg_phrases: Negative indicator phrases
        pos_embedding: 768-dim embedding for positive
        neg_embedding: 768-dim embedding for negative (optional)
    
    Returns:
        Molecule with dipole layout
    """
    chunks = []
    
    # Chunk 0: Positive (always present)
    pos_text = ", ".join(pos_phrases[:5]) + ("..." if len(pos_phrases) > 5 else "")
    pos_chunk = create_chunk_from_embedding(
        chunk_id=0,
        chunk_type="positive",
        text=pos_text,
        embedding=pos_embedding,
        position_x=0.0,
        connects_to=[1] if neg_embedding is not None else []
    )
    chunks.append(pos_chunk)
    
    # Chunk 1: Negative (if exists)
    if neg_embedding is not None:
        neg_text = ", ".join(neg_phrases[:5]) + ("..." if len(neg_phrases) > 5 else "")
        neg_chunk = create_chunk_from_embedding(
            chunk_id=1,
            chunk_type="negative",
            text=neg_text,
            embedding=neg_embedding,
            position_x=2.0,
            connects_to=[]
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
