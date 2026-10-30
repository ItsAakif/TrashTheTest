"""
Text Extraction Types and Models

Pydantic models for extraction results and progress.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PageExtractionResult(BaseModel):
    """Result for a single page extraction"""
    page_number: int
    text: str
    confidence: float = 1.0
    method: str  # "native", "layout", "vision:<provider>", "docx", "plain-text", "ocr_unavailable"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    structural_elements: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentExtractionResult(BaseModel):
    """Complete result for document extraction"""
    document_id: str
    total_pages: int
    pages: List[PageExtractionResult]
    full_text: str
    method: str  # Primary extraction method used
    processing_time: float
    topics: List[str] = Field(default_factory=list)  # Extracted topics for research
    is_ocr_required: bool = False
    is_complete: bool = True
    
    @property
    def page_count(self) -> int:
        return len(self.pages)


class ExtractionProgress(BaseModel):
    """Progress tracking for extraction"""
    document_id: str
    current_page: int
    total_pages: int
    status: str  # "extracting", "processing", "completed", "failed"
    progress_percent: float
    message: Optional[str] = None
    last_processed_page: Optional[int] = None


class ExtractionJobInput(BaseModel):
    """Input for extraction job"""
    document_id: str
    project_id: str
    file_path: str
    file_type: str
    start_page: Optional[int] = None  # For resume
    force_ocr: bool = False


__all__ = [
    "PageExtractionResult",
    "DocumentExtractionResult", 
    "ExtractionProgress",
    "ExtractionJobInput",
]