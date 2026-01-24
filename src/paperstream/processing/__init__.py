"""
PaperStream Processing Module
=============================

Contains PDF processing utilities including thumbnail generation and voxelization.
"""

from .thumbnail import generate_thumbnail, ThumbnailGenerator
from .pdf_voxelizer import pdf_to_voxel_grid, pdf_to_voxel_grid_colored, section_to_voxel

__all__ = [
    "generate_thumbnail", 
    "ThumbnailGenerator",
    "pdf_to_voxel_grid",
    "pdf_to_voxel_grid_colored",
    "section_to_voxel"
]
