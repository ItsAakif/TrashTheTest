"""
Website Generation Module

Transforms structured content from the Transformation Engine into
standalone HTML study websites.
"""

from .types import (
    WebsiteContentType,
    WebsiteGenerationRequest,
    WebsiteGenerationResult,
    WebsiteMetadata,
    WebsiteContent,
)

from .service import WebsiteGenerationService, website_gen_service

from .routes import router


__all__ = [
    # Types
    "WebsiteContentType",
    "WebsiteGenerationRequest",
    "WebsiteGenerationResult",
    "WebsiteMetadata",
    "WebsiteContent",
    # Service
    "WebsiteGenerationService",
    "website_gen_service",
    # Routes
    "router",
]