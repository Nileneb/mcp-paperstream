"""
PDF Thumbnail Generator
=======================

Generates thumbnail images from PDF first pages for Unity display.
Uses PyMuPDF (fitz) for PDF rendering.
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """
    Generates thumbnail images from PDF files.
    
    Supports:
    - PNG output (default for Unity compatibility)
    - Configurable size
    - Base64 encoding for API responses
    - Disk caching
    """
    
    DEFAULT_WIDTH = 200
    DEFAULT_HEIGHT = 280  # A4 aspect ratio approximately
    DEFAULT_DPI = 72
    
    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize thumbnail generator.
        
        Args:
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels  
            cache_dir: Directory to cache generated thumbnails
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for thumbnail generation. Install with: pip install PyMuPDF")
        
        self.width = width
        self.height = height
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_pdf(
        self,
        pdf_path: Union[str, Path],
        page_num: int = 0,
        output_format: str = "png"
    ) -> Optional[bytes]:
        """
        Generate thumbnail from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to render (0-indexed)
            output_format: Output format ("png" or "jpeg")
        
        Returns:
            Image bytes or None if failed
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None
        
        try:
            doc = fitz.open(str(pdf_path))
            
            if page_num >= len(doc):
                logger.warning(f"Page {page_num} not found in {pdf_path}, using page 0")
                page_num = 0
            
            page = doc[page_num]
            
            # Calculate zoom factor to fit within dimensions
            rect = page.rect
            zoom_x = self.width / rect.width
            zoom_y = self.height / rect.height
            zoom = min(zoom_x, zoom_y)  # Fit within bounds
            
            # Create transformation matrix
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to image bytes
            if output_format.lower() == "jpeg":
                img_bytes = pix.tobytes("jpeg")
            else:
                img_bytes = pix.tobytes("png")
            
            doc.close()
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {pdf_path}: {e}")
            return None
    
    def generate_to_base64(
        self,
        pdf_path: Union[str, Path],
        page_num: int = 0,
        output_format: str = "png"
    ) -> Optional[str]:
        """
        Generate thumbnail and return as base64 string.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to render
            output_format: Output format
        
        Returns:
            Base64 encoded image string or None
        """
        img_bytes = self.generate_from_pdf(pdf_path, page_num, output_format)
        
        if img_bytes:
            return base64.b64encode(img_bytes).decode("utf-8")
        return None
    
    def generate_and_save(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        page_num: int = 0,
        output_format: str = "png"
    ) -> Optional[Path]:
        """
        Generate thumbnail and save to disk.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output path (auto-generated if None)
            page_num: Page number to render
            output_format: Output format
        
        Returns:
            Path to saved thumbnail or None
        """
        pdf_path = Path(pdf_path)
        
        # Auto-generate output path if not provided
        if output_path is None:
            if self.cache_dir:
                output_path = self.cache_dir / f"{pdf_path.stem}_thumb.{output_format}"
            else:
                output_path = pdf_path.with_suffix(f".thumb.{output_format}")
        else:
            output_path = Path(output_path)
        
        # Check cache
        if output_path.exists():
            logger.debug(f"Using cached thumbnail: {output_path}")
            return output_path
        
        # Generate thumbnail
        img_bytes = self.generate_from_pdf(pdf_path, page_num, output_format)
        
        if img_bytes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(img_bytes)
            logger.info(f"Saved thumbnail: {output_path}")
            return output_path
        
        return None
    
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get PDF metadata and page count.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with PDF info or None
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return None
        
        try:
            doc = fitz.open(str(pdf_path))
            
            info = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
            }
            
            # Get first page dimensions
            if len(doc) > 0:
                rect = doc[0].rect
                info["page_width"] = rect.width
                info["page_height"] = rect.height
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info for {pdf_path}: {e}")
            return None


# Module-level convenience function
_generator: Optional[ThumbnailGenerator] = None


def generate_thumbnail(
    pdf_path: Union[str, Path],
    width: int = ThumbnailGenerator.DEFAULT_WIDTH,
    height: int = ThumbnailGenerator.DEFAULT_HEIGHT,
    as_base64: bool = False,
    output_format: str = "png"
) -> Optional[Union[bytes, str]]:
    """
    Convenience function to generate a thumbnail.
    
    Args:
        pdf_path: Path to PDF file
        width: Thumbnail width
        height: Thumbnail height
        as_base64: Return as base64 string
        output_format: Image format
    
    Returns:
        Image bytes, base64 string, or None
    """
    global _generator
    
    if _generator is None or _generator.width != width or _generator.height != height:
        _generator = ThumbnailGenerator(width=width, height=height)
    
    if as_base64:
        return _generator.generate_to_base64(pdf_path, output_format=output_format)
    else:
        return _generator.generate_from_pdf(pdf_path, output_format=output_format)
