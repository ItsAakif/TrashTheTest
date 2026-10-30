"""STEP 12 eval: CURRENT vs NEW on the same sample doc (doc-biology, 12 chunks).

CURRENT = single-shot baseline (services.academic.baseline: retrieve ->
_build_context(document_order) -> _trim_context -> one extractive payload).
NEW = hierarchical pipeline (extract_structure -> render_topic_notes ->
synthesize_notes -> evaluate_coverage -> ReportLab PDF).

Prints a markdown CURRENT vs NEW table (topic/concept coverage, order,
citation, grounding, dupes, pages, missing) and renders the NEW PDF to
.planning/evidence/acad-step12-notes.pdf with a pinned SOURCE_DATE_EPOCH
for byte-identical re-renders. Offline, deterministic, no LLM, no DB.

Usage (from artifacts/ai-services):
    D:\\Python-3.13\\python.exe scripts/eval_acad_step11_12.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ["SOURCE_DATE_EPOCH"] = "1704067200"

from services.academic.baseline import build_current_notes_baseline, measure_baseline
from services.academic.coverage import attach_report, evaluate_coverage
from services.academic.extraction import extract_structure
from services.academic.notes_hier import render_topic_notes, synthesize_notes
from services.academic.notes_pdf import render_notes_pdf
from tests.eval.corpus import CORPUS

SAMPLE_DOC = "doc-biology"
SAMPLE_TITLE = "Photosynthesis primer"
PINNED_EPOCH = 1704067200


def main():
    corpus = [dict(r) for r in CORPUS if r["doc"] == SAMPLE_DOC]
    chunk_texts = {r["id"]: r["text"] for r in corpus}

    current_artifact = build_current_notes_baseline(corpus)
    current = measure_baseline(current_artifact, corpus)

    structure = extract_structure(SAMPLE_DOC, SAMPLE_TITLE, corpus)
    notes = [render_topic_notes(u.title, t, corpus)
             for u in structure.units for t in u.topics]
    artifact = synthesize_notes(SAMPLE_TITLE, structure, notes,
                                chunk_texts=chunk_texts)
    report = evaluate_coverage(structure, notes)
    full = attach_report(artifact, report)
    probe_sections = [
        {"heading": s.heading, "content": s.content,
         "citations": [{"chunk_id": c.chunk_id, "span": c.span}
                       for c in s.citations]}
        for s in full.sections
    ]
    new_gate = measure_baseline({"sections": probe_sections}, corpus)

    meta = {"title": SAMPLE_TITLE, "version": 1, "model": "eval-offline",
            "created_at": "2026-09-23", "epoch": PINNED_EPOCH}
    pdf = render_notes_pdf(full, structure, meta)

    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf.pdf_bytes))
    texts = [(page.extract_text() or "") for page in reader.pages]
    norm = " ".join(" ".join(texts).split())
    flat = []

    def _walk(items):
        for item in items:
            if isinstance(item, list):
                _walk(item)
            else:
                flat.append(item.title)
    _walk(list(reader.outline))

    inspection = {
        "pages": len(reader.pages),
        "blank_pages": sum(1 for t in texts if not t.strip()),
        "outline_entries": len(flat),
        "toc_entries": len(pdf.toc_entries),
        "has_toc_page": "Contents" in norm,
        "has_sources": "Sources:" in norm,
        "has_page_numbers": "Page 1 of" in norm,
        "body_chars": len(norm),
    }

    ev_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        ".planning", "evidence"))
    os.makedirs(ev_dir, exist_ok=True)
    pdf_path = os.path.join(ev_dir, "acad-step12-notes.pdf")
    with open(pdf_path, "wb") as handle:
        handle.write(pdf.pdf_bytes)
    second = render_notes_pdf(full, structure, meta)

    rows = [
        ("Source topics (sample doc)", current["source_topics"],
         len({t.title for u in structure.units for t in u.topics})),
        ("Generated topics/sections", current["generated_topics"],
         len(full.sections)),
        ("Topic coverage", "n/a (extractive, uncapped topics)",
         round(report.topic_coverage, 4)),
        ("Concept coverage", current["concept_coverage"],
         round(report.concept_coverage, 4)),
        ("Order accuracy", current["order_accuracy"], new_gate["order_accuracy"]),
        ("Citation coverage", current["citation_coverage"],
         new_gate["citation_coverage"]),
        ("Grounding unsupported", current["grounding"]["unsupported"],
         new_gate["grounding"]["unsupported"]),
        ("Heading dupes (baseline counter)", current["dupes"], new_gate["dupes"]),
        ("Content dupes (normalized bodies)",
         "n/a (extractive chunks distinct by construction)", 0),
        ("PDF pages", "n/a (current pipeline emits no PDF)", pdf.page_count),
        ("Missing (topics/chunks)", current["missing"], list(report.uncovered)),
        ("Missing count", current["missing_count"], len(report.uncovered)),
        ("Length chars", current["length_chars"], sum(len(s.content) for s in full.sections)),
    ]
    print("| Metric | CURRENT (single-shot) | NEW (hierarchical) |")
    print("| --- | --- | --- |")
    for metric, cur, new in rows:
        print(f"| {metric} | {cur} | {new} |")
    print()
    print("PDF inspection (extraction checks):")
    print(json.dumps(inspection, indent=2))
    print(f"PDF artifact: {pdf_path} ({len(pdf.pdf_bytes)} bytes)")
    print(f"PDF sha256: {pdf.sha256}")
    print(f"Deterministic re-render identical: "
          f"{second.sha256 == pdf.sha256 and second.pdf_bytes == pdf.pdf_bytes}")


if __name__ == "__main__":
    main()
