"""Pdfplumber layout parser (T204 permissive last-resort, pure geometry).

No AGPL component. Returns per-page Markdown plus a structured table
representation; equations are preserved as original glyphs (pdfplumber has
no LaTeX recognizer - stated honestly in the note).
"""

from typing import Any, Dict, List


class PdfplumberParser:
    name = "pdfplumber"

    def parse_page(self, file_path: str, page_number: int) -> Dict[str, Any]:
        try:
            import pdfplumber
        except ImportError:
            return {"markdown": "", "tables_markdown": "", "latex": [],
                    "note": "pdfplumber not installed; tier-0 output kept"}
        tables_md: List[str] = []
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                page = pdf.pages[page_number - 1]
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                for table in page.extract_tables() or []:
                    rows = [" | ".join((cell or "").strip() for cell in row) for row in table if row]
                    if rows:
                        tables_md.append("\n".join(rows))
        except Exception as exc:
            return {"markdown": text, "tables_markdown": "", "latex": [],
                    "note": f"pdfplumber error: {exc}; tier-0 output kept"}
        return {"markdown": text,
                "tables_markdown": "\n\n".join(tables_md),
                "latex": [],
                "note": "equations preserved as glyphs only (no LaTeX recognizer)"}


__all__ = ["PdfplumberParser"]
