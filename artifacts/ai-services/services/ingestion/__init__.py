"""
Ingestion Service Package

Handles document upload, storage, and initial processing.
"""

from .routes import router

__all__ = ["router"]