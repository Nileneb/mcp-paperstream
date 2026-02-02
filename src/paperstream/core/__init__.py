"""
Core module - shared data models and utilities
"""

from .data_model import (
    # Constants
    VOXEL_GRID_X,
    VOXEL_GRID_Y,
    VOXEL_GRID_Z,
    SECTION_COLORS,
    LAYOUT_CHAIN,
    LAYOUT_DIPOLE,
    CONNECTION_SEQUENTIAL,
    CONNECTION_POLAR,
    
    # Data classes
    Voxel,
    VoxelGrid,
    Chunk,
    Molecule,
    
    # Factory functions
    embedding_to_base64,
    base64_to_embedding,
    create_chunk_from_embedding,
    create_paper_molecule,
    create_rule_molecule,
)

__all__ = [
    # Constants
    "VOXEL_GRID_X",
    "VOXEL_GRID_Y", 
    "VOXEL_GRID_Z",
    "SECTION_COLORS",
    "LAYOUT_CHAIN",
    "LAYOUT_DIPOLE",
    "CONNECTION_SEQUENTIAL",
    "CONNECTION_POLAR",
    
    # Data classes
    "Voxel",
    "VoxelGrid",
    "Chunk",
    "Molecule",
    
    # Factory functions
    "embedding_to_base64",
    "base64_to_embedding",
    "create_chunk_from_embedding",
    "create_paper_molecule",
    "create_rule_molecule",
]
