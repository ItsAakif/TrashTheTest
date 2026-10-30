"""Layout tier interface (T204, license-gated per OQ-1).

Docling/MinerU are NOT defaults (license unconfirmed; MinerU AGPL-gated in
older releases). The active parser is the permissive pdfplumber geometry
recovery; Docling/PyMuPDF4LLM adapters may be registered only after their
repository LICENSE files are confirmed permissive, recorded in evidence.
"""

from typing import Any, Dict, Protocol


class LayoutParser(Protocol):
    def parse_page(self, file_path: str, page_number: int) -> Dict[str, Any]:
        """Return {"markdown": str, "tables_markdown": str, "latex": [...], "note": str}."""
        ...


__all__ = ["LayoutParser"]
