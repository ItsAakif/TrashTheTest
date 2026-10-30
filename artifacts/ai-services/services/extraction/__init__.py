"""
Text Extraction Service Package

Handles document text extraction with PP-OCRv6 (CPU ONNX) and PyMuPDF.
"""

from .routes import router

__all__ = ["router"]