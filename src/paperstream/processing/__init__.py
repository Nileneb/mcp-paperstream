"""
PaperStream Processing Module
=============================

Contains PDF processing utilities including thumbnail generation.
"""

from .thumbnail import generate_thumbnail, ThumbnailGenerator

__all__ = ["generate_thumbnail", "ThumbnailGenerator"]
