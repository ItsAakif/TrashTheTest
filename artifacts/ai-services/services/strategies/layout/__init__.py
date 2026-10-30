# Layout parsers (T204). Docling/MinerU adapters are NOT shipped: their
# licenses are unconfirmed (OQ-1) and must be verified from the upstream
# LICENSE file before either becomes selectable. pdfplumber is the active
# permissive parser.
from .base import LayoutParser
from .pdfplumber_parser import PdfplumberParser

__all__ = ["LayoutParser", "PdfplumberParser"]
