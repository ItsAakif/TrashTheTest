"""STEP 8 study-notes PDF renderer (ReportLab, the ONE PDF library).

Library choice (ADR-007 D8): ReportLab, not WeasyPrint and not an
HTML-to-PDF chain.

- Docker: ReportLab is a pure-Python wheel (no apt layer, no system
  libraries). WeasyPrint needs Pango/Cairo/GDK-PixBuf in the image, which
  violates the "no added apt layer on python:3.13-slim" constraint.
- Fonts: built-in Type1 base fonts (Helvetica/Courier) render without
  shipping font files. Non-Latin titles degrade gracefully (see
  _safe_text); a full CJK CID bundle is deferred per OQ-5, documented here
  rather than silently half-working.
- Tables: platypus Table with repeatable header rows and GRID styling
  keeps wide tables inside the frame (wrapped Paragraph cells, computed
  column widths) instead of clipping.
- Formulas: source text in monospace code blocks (no matplotlib, per the
  spec non-goal); formulas stay extractable text, never images.
- Bookmarks/nav: canvas.bookmarkPage + addOutlineEntry per unit/topic,
  plus a TableOfContents flowable whose page numbers are resolved by
  multiBuild and verified by reading the PDF back.
- Numbering: a NumberedCanvas replays saved pages so the footer reads
  "Page X of Y" with the artifact version beside it.
- Deterministic: no datetime.now() in the body. The visible date comes
  from meta/created_at (or SOURCE_DATE_EPOCH), and _determinize_pdf pins
  CreationDate/ModDate plus the trailer ID before hashing, so two renders
  with a pinned SOURCE_DATE_EPOCH are byte-identical.
- No browser: pure Python, works offline in CI and in Docker.

Pipeline position: structured NotesArtifact -> render_notes_pdf ->
determinized bytes -> storage backend (opaque key on the GeneratedArtifact
`path`) -> GET /api/academic/notes/{project_id}/pdf -> the EXISTING
DocumentViewer (react-pdf). No new viewer, no new retrieval.
"""

import hashlib
import io
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

BRAND = "TrashTheTest"
BRAND_SUBTITLE = "Study Notes"

# Visible date default when meta carries none: the Unix epoch rendered as
# an ISO date. Never datetime.now(): the body must be deterministic.
DEFAULT_EPOCH = 0

# Inline code/formula detection on plain synthesis text (the shape extras
# from TopicNotes do not survive synthesis, so the renderer classifies
# sentences textually; explicit topic_extras override when supplied).
_FORMULA_RE = re.compile(r"[A-Za-z0-9_\)\]]\s*=\s*.+|[0-9]\s*[\+\-\*/\^]\s*[0-9A-Za-z]")
_EXAMPLE_RE = re.compile(r"\b(for example|e\.g\.|such as|example:|for instance)\b", re.I)
_EXAM_RE = re.compile(r"\b(exam|marks?\b|important|key point|remember that|crucial|frequently asked)\b", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_TOC_TITLE = "Contents"

# Honest pointer for a spine topic whose prose was deduped into a shared
# section during synthesis (identical normalized paragraphs appear once,
# first wins, citations merged). The heading still renders so the PDF
# covers every spine topic; the pointer names the mechanism truthfully
# instead of inventing body text.
_DEDUPED_POINTER = (
    "This topic shares source prose with a section above "
    "(identical material deduped during synthesis); see Sources."
)


def _escape(text: str) -> str:
    """XML-escape plain text for ReportLab Paragraph markup."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _safe_text(text: str) -> str:
    """Keep base-font (WinAnsi) renderable text; degrade the rest honestly.

    Helvetica cannot draw CJK/complex scripts without a CID bundle (OQ-5,
    deferred: no font files ship in the slim image). Rather than raising
    or emitting tofu mid-sentence, unrenderable characters become "?" and
    the title page notes the substitution count. Latin-1 passes through
    untouched.
    """
    out: List[str] = []
    dropped = 0
    for char in text or "":
        try:
            char.encode("cp1252")
            out.append(char)
        except UnicodeEncodeError:
            out.append("?")
            dropped += 1
    return "".join(out), dropped


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def _split_long_bullet(text: str, max_words: int = 25) -> List[str]:
    """Split an over-long bullet into <= ``max_words``-word bullets.

    Splits on word boundaries (never truncates mid-word); short bullets
    pass through unchanged.
    """
    words = (text or "").split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _pdf_date(epoch: int) -> str:
    moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    return moment.strftime("D:%Y%m%d%H%M%S+00'00'")


def _visible_date(meta: Dict[str, Any], epoch: int) -> str:
    raw = (meta or {}).get("created_at") or (meta or {}).get("date")
    if raw:
        return str(raw)[:10]
    moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    return moment.strftime("%Y-%m-%d")


def _determinize_pdf(data: bytes, epoch: int) -> bytes:
    """Pin volatile PDF trailer fields so renders are byte-identical.

    ReportLab stamps the build clock into /CreationDate + /ModDate and a
    random-ish trailer /ID per save. Both are transport metadata, not
    content: replacing them with epoch-derived/fixed tokens is the same
    reproducible-build practice as SOURCE_DATE_EPOCH elsewhere.
    """
    fixed_date = _pdf_date(epoch).encode("ascii")
    out = re.sub(rb"/CreationDate\s*\([^)]*\)", b"/CreationDate (" + fixed_date + b")", data)
    out = re.sub(rb"/ModDate\s*\([^)]*\)", b"/ModDate (" + fixed_date + b")", out)
    out = re.sub(
        rb"/ID\s*\[[^\]]*\]",
        b"/ID [<00000000000000000000000000000000> <00000000000000000000000000000000>]",
        out,
    )
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def notes_pdf_storage_hint(project_id: str, version: int) -> str:
    """Storage key hint for one notes-PDF version (server names the file)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(project_id)) or "project"
    return f"study-notes-{safe}-v{int(version)}.pdf"


def _section_text(section: Any) -> str:
    if isinstance(section, dict):
        return str(section.get("content") or "")
    return str(getattr(section, "content", "") or "")


def _section_heading(section: Any) -> str:
    if isinstance(section, dict):
        return str(section.get("heading") or section.get("topic_title") or "Notes")
    return str(getattr(section, "heading", None) or getattr(section, "topic_title", "Notes"))


def _section_cites(section: Any) -> List[Dict[str, str]]:
    raw = section.get("citations", []) if isinstance(section, dict) else (getattr(section, "citations", []) or [])
    out: List[Dict[str, str]] = []
    for cite in raw or []:
        if isinstance(cite, dict):
            out.append({"chunk_id": str(cite.get("chunk_id") or ""), "span": str(cite.get("span") or "")})
        else:
            out.append({
                "chunk_id": str(getattr(cite, "chunk_id", "") or ""),
                "span": str(getattr(cite, "span", "") or ""),
            })
    return [c for c in out if c["chunk_id"]]


def _cite_label(chunk_id: str, span: str, page: Optional[int], index: int) -> str:
    short = (chunk_id or "")[:8] or f"chunk-{index + 1}"
    bits = [f"[{index + 1}] {short}"]
    if page is not None:
        bits.append(f"p.{page}")
    if span:
        clipped = " ".join(span.split())
        if len(clipped) > 90:
            clipped = clipped[:87] + "..."
        bits.append(f'"{clipped}"')
    return " ".join(bits)


@dataclass
class NotesPdfReport:
    """Render outcome: determinized bytes plus structural evidence."""

    pdf_bytes: bytes = b""
    page_count: int = 0
    toc_entries: List[Dict[str, Any]] = field(default_factory=list)
    bookmark_count: int = 0
    sha256: str = ""
    topic_count: int = 0
    section_count: int = 0


def render_notes_pdf(
    artifact: Any,
    structure: Any = None,
    meta: Optional[Dict[str, Any]] = None,
    topic_extras: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    chunk_texts: Optional[Dict[str, str]] = None,
    allowed_chunk_ids: Optional[set[str]] = None,
) -> NotesPdfReport:
    """Render a NotesArtifact (+ optional spine) to a university notes PDF.

    artifact: NotesArtifact (or its model_dump dict) with sections in
      spine order. structure: DocumentStructure (or None; units/topics
      are then grouped from the sections themselves). meta: title,
      version, model, created_at/date, coverage override, chunk_pages
      ({chunk_id: page} for page-aware cites), epoch override.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        KeepTogether,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    meta = dict(meta or {})
    extras = topic_extras or {}
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", str(meta.get("epoch", DEFAULT_EPOCH))))
    version = str(meta.get("version", 1))
    model_name = str(meta.get("model", "academic-notes"))
    title = str(meta.get("title") or _artifact_title(artifact) or "Study Notes")
    coverage = _artifact_coverage(artifact, meta)
    chunk_pages = dict(meta.get("chunk_pages") or {})

    sections = _artifact_sections(artifact)
    if not sections:
        raise ValueError("render_notes_pdf found no sections to render")
    # Do not turn headings/boilerplate into a successful PDF.  The render
    # guard receives the spine and source scope so a long title-only section
    # cannot pass merely because it is over the character minimum.
    from services.academic.validation import assert_pdf_has_body
    source_texts = chunk_texts or meta.get("chunk_texts")
    source_ids = allowed_chunk_ids
    if source_ids is None and isinstance(source_texts, dict):
        source_ids = set(source_texts)
    assert_pdf_has_body(
        artifact,
        structure=structure,
        allowed_chunk_ids=source_ids,
        chunk_texts=source_texts,
        min_chars=40,
    )
    units = _plan_units(sections, structure)
    topic_count = sum(len(topics) for _, topics in units)

    # -- styles (Helvetica family only: base fonts, Docker-safe) ------------
    sheet = getSampleStyleSheet()
    brand_style = ParagraphStyle("Brand", parent=sheet["Normal"], fontName="Helvetica",
                                 fontSize=10, leading=13, textColor=colors.HexColor("#4f46e5"),
                                 spaceAfter=4)
    title_style = ParagraphStyle("TitlePage", parent=sheet["Title"], fontName="Helvetica-Bold",
                                 fontSize=28, leading=32, spaceAfter=8)
    subtitle_style = ParagraphStyle("Subtitle", parent=sheet["Normal"], fontName="Helvetica",
                                    fontSize=12, leading=15, textColor=colors.HexColor("#525252"),
                                    spaceAfter=12)
    meta_cell = ParagraphStyle("MetaCell", parent=sheet["Normal"], fontName="Helvetica",
                               fontSize=9, leading=12)
    meta_head = ParagraphStyle("MetaHead", parent=meta_cell, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("UnitH1", parent=sheet["Heading1"], fontName="Helvetica-Bold",
                        fontSize=18, leading=22, spaceBefore=14, spaceAfter=8,
                        keepWithNext=1, textColor=colors.HexColor("#111111"))
    h2 = ParagraphStyle("TopicH2", parent=sheet["Heading2"], fontName="Helvetica-Bold",
                        fontSize=14, leading=18, spaceBefore=12, spaceAfter=6,
                        keepWithNext=1, textColor=colors.HexColor("#1f2937"))
    h3 = ParagraphStyle("SubH3", parent=sheet["Heading3"], fontName="Helvetica-Bold",
                        fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
                        keepWithNext=1, textColor=colors.HexColor("#374151"))
    # Body: left-aligned (never justified), leading >= 1.5 * fontSize, and
    # widow/orphan control so a line never strands alone.
    body = ParagraphStyle("Body", parent=sheet["Normal"], fontName="Helvetica",
                          fontSize=10, leading=15.5, spaceAfter=5, alignment=0,
                          allowWidows=0, allowOrphans=0)
    code = ParagraphStyle("Formula", parent=sheet["Code"], fontName="Courier",
                          fontSize=9, leading=12.5, spaceAfter=5,
                          backColor=colors.HexColor("#f5f5f4"), borderPadding=6)
    callout = ParagraphStyle("Callout", parent=sheet["Normal"], fontName="Helvetica",
                             fontSize=9.5, leading=13.5,
                             allowWidows=0, allowOrphans=0)
    callout_head = ParagraphStyle("CalloutHead", parent=callout, fontName="Helvetica-Bold")
    cite_style = ParagraphStyle("Cite", parent=sheet["Normal"], fontName="Helvetica-Oblique",
                                fontSize=8.5, leading=11.5, textColor=colors.HexColor("#57534e"),
                                spaceAfter=2)
    toc_h1 = ParagraphStyle("TOCH1", parent=sheet["Normal"], fontName="Helvetica-Bold",
                            fontSize=11, leading=15)
    toc_h2 = ParagraphStyle("TOCH2", parent=sheet["Normal"], fontName="Helvetica",
                            fontSize=10, leading=14, leftIndent=12)

    page_w, page_h = A4
    margin = 20 * mm
    available_w = page_w - 2 * margin
    # ~66-character measure: measure the body font once and cap the frame.
    sample = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;:"
    body_char_w = stringWidth(sample, "Helvetica", 10) / max(1, len(sample))
    frame_w = min(available_w, body_char_w * 66)
    frame_h = page_h - 2 * margin

    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer, pagesize=A4,
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
        title=_safe_ascii(title), author=BRAND,
        subject="Study notes generated from course material",
        keywords="study notes",
    )
    frame = Frame(margin, margin, frame_w, frame_h, id="body")

    toc_pages: Dict[Tuple[int, str], int] = {}
    bookmarks: List[Tuple[int, str]] = []
    counter = {"n": 0}

    def _after_flowable(flowable: Any) -> None:
        if not hasattr(flowable, "style"):
            return
        name = getattr(flowable.style, "name", "")
        level = {"UnitH1": 0, "TopicH2": 1, "SubH3": 2}.get(name)
        if level is None or level > 1:
            return  # TOC + outline cover units and topics only.
        text = ""
        try:
            text = flowable.getPlainText()
        except Exception:
            text = ""
        text = (text or "").strip()
        if not text:
            return
        counter["n"] += 1
        key = f"notes-h{counter['n']}"
        canv = doc.canv
        canv.bookmarkPage(key)
        canv.addOutlineEntry(text, key, level, closed=False)
        toc_pages[(level, text)] = doc.page
        bookmarks.append((level, text))
        try:
            doc.notify("TOCEntry", (level, text, doc.page))
        except Exception:
            pass

    def _header_footer(canv: Any, _doc: Any) -> None:
        canv.saveState()
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#a8a29e"))
        canv.drawString(margin, page_h - 14 * mm, BRAND)
        canv.drawRightString(page_w - margin, page_h - 14 * mm, f"v{version}")
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#a8a29e"))
        canv.drawCentredString(page_w / 2, 12 * mm, f"v{version}")
        canv.restoreState()

    template = PageTemplate(id="notes", frames=[frame], onPage=_header_footer)
    doc.addPageTemplates([template])
    doc.afterFlowable = _after_flowable  # type: ignore[attr-defined]

    story: List[Any] = []
    safe_title, dropped_chars = _safe_text(title)

    # -- title page ---------------------------------------------------------
    story.append(Paragraph(_escape(BRAND.upper() + "  -  " + BRAND_SUBTITLE.upper()), brand_style))
    story.append(Paragraph(_escape(safe_title), title_style))
    story.append(Paragraph(
        _escape(f"{topic_count} topics - {len(sections)} sections - generated from your course material"),
        subtitle_style,
    ))
    topic_cov = coverage.get("topic_coverage")
    concept_cov = coverage.get("concept_coverage")
    meta_rows = [
        [Paragraph("<b>Date</b>", meta_head), Paragraph(_escape(_visible_date(meta, epoch)), meta_cell)],
        [Paragraph("<b>Model</b>", meta_head), Paragraph(_escape(model_name), meta_cell)],
        [Paragraph("<b>Version</b>", meta_head), Paragraph(_escape(f"v{version}"), meta_cell)],
        [Paragraph("<b>Topics</b>", meta_head), Paragraph(_escape(str(topic_count)), meta_cell)],
        [Paragraph("<b>Coverage</b>", meta_head), Paragraph(_escape(_coverage_line(coverage)), meta_cell)],
        [Paragraph("<b>Sources</b>", meta_head), Paragraph(_escape(str(meta.get("source_count", "course documents"))), meta_cell)],
    ]
    meta_table = Table(meta_rows, colWidths=[32 * mm, frame_w - 32 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f5f4")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6d3d1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e7e5e4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    if dropped_chars:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(
            _escape(f"Note: {dropped_chars} character(s) outside the base font set were simplified (?)."),
            cite_style,
        ))
    story.append(PageBreak())

    # -- table of contents --------------------------------------------------
    story.append(Paragraph(_escape(_TOC_TITLE), h1))
    toc = TableOfContents()
    toc.levelStyles = [toc_h1, toc_h2]
    story.append(toc)
    story.append(PageBreak())

    # -- units -> topics (spine order; every spine topic gets a heading) --
    for unit_pos, (unit_title, topics) in enumerate(units):
        unit_heading, _ = _safe_text(f"Unit {unit_pos + 1} -- {unit_title}")
        story.append(Paragraph(_escape(unit_heading), h1))
        for topic_pos, (topic_title, subtopics) in enumerate(topics):
            if topic_pos > 0:
                story.append(PageBreak())  # top-level topic boundary
            topic_heading, _ = _safe_text(f"{unit_pos + 1}.{topic_pos + 1}  {topic_title}")
            heading_para = Paragraph(_escape(topic_heading), h2)
            topic_flow: List[Any] = []
            if subtopics and any(paras for _, paras in subtopics):
                for sub_title, paras in subtopics:
                    if not paras:
                        continue
                    if sub_title and sub_title.strip().lower() != topic_title.strip().lower():
                        safe_sub, _ = _safe_text(sub_title)
                        topic_flow.append(Paragraph(_escape(safe_sub), h3))
                    topic_flow.extend(_render_paras(
                        paras, unit_title, topic_title, chunk_pages, extras,
                        body, code, callout, callout_head, cite_style, frame_w,
                    ))
            if not topic_flow:
                topic_flow = [Paragraph(_escape(_DEDUPED_POINTER), body)]
            # Keep the heading with its first paragraph so a heading never
            # strands at the bottom of a page.
            story.append(KeepTogether([heading_para, topic_flow[0]]))
            story.extend(topic_flow[1:])
        if unit_pos < len(units) - 1:
            story.append(PageBreak())

    # -- coverage appendix (honest, never synthesized) ----------------------
    story.append(PageBreak())
    story.append(Paragraph(_escape("Coverage"), h1))
    story.append(Paragraph(_escape(_coverage_line(coverage, verbose=True)), body))
    uncovered = list(coverage.get("uncovered") or _artifact_uncovered(artifact))
    if uncovered:
        story.append(Paragraph(_escape("Uncovered topics (reported, not invented):"), callout_head))
        for name in uncovered:
            safe_name, _ = _safe_text(str(name))
            story.append(Paragraph(_escape(f"- {safe_name}"), body))
    else:
        story.append(Paragraph(_escape("Every topic in the spine is covered by at least one cited section."), body))
    for note in _artifact_cross_notes(artifact):
        story.append(Paragraph(_escape(str(note)), cite_style))

    doc.multiBuild(story, canvasmaker=_NumberedCanvas)

    raw = buffer.getvalue()
    final = _determinize_pdf(raw, epoch)
    digest = _sha256(final)

    # afterFlowable fires on every multiBuild pass; recorded page numbers
    # come from the final pass (dict overwrite), and the entry list is
    # deduped to one row per heading (first occurrence wins).
    seen: set = set()
    toc_entries: List[Dict[str, Any]] = []
    for level, text in bookmarks:
        if (level, text) in seen or level > 1:
            continue
        seen.add((level, text))
        toc_entries.append(
            {"level": level, "title": text, "page": toc_pages.get((level, text), 0)}
        )
    report = NotesPdfReport(
        pdf_bytes=final,
        page_count=_count_pages(final),
        toc_entries=toc_entries,
        bookmark_count=len(toc_entries),
        sha256=digest,
        topic_count=topic_count,
        section_count=len(sections),
    )
    return report


# -- story builders ----------------------------------------------------------

def _artifact_title(artifact: Any) -> str:
    if isinstance(artifact, dict):
        return str(artifact.get("title") or "")
    return str(getattr(artifact, "title", "") or "")


def _artifact_sections(artifact: Any) -> List[Any]:
    if isinstance(artifact, dict):
        raw = artifact.get("sections") or []
        return list(raw)
    return list(getattr(artifact, "sections", []) or [])


def _artifact_coverage(artifact: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    if meta.get("coverage"):
        return dict(meta["coverage"])
    raw = artifact.get("coverage") if isinstance(artifact, dict) else getattr(artifact, "coverage", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return dict(raw.model_dump())
    except Exception:
        return {}


def _artifact_uncovered(artifact: Any) -> List[str]:
    if isinstance(artifact, dict):
        return [str(u) for u in (artifact.get("uncovered") or [])]
    return [str(u) for u in (getattr(artifact, "uncovered", []) or [])]


def _artifact_cross_notes(artifact: Any) -> List[str]:
    if isinstance(artifact, dict):
        return [str(n) for n in (artifact.get("cross_topic_notes") or [])]
    return [str(n) for n in (getattr(artifact, "cross_topic_notes", []) or [])]


def _coverage_line(coverage: Dict[str, Any], verbose: bool = False) -> str:
    if not coverage:
        return "coverage not measured" if verbose else "n/a"
    parts: List[str] = []
    if coverage.get("topic_coverage") is not None:
        parts.append(f"topics {float(coverage['topic_coverage']):.0%}")
    if coverage.get("concept_coverage") is not None:
        parts.append(f"concepts {float(coverage['concept_coverage']):.0%}")
    if coverage.get("passes") is not None:
        parts.append("passes" if coverage["passes"] else "below threshold")
    line = ", ".join(parts) or "measured"
    if verbose:
        line += f" (threshold {coverage.get('threshold', 'n/a')})"
    return line


def _safe_ascii(text: str) -> str:
    return "".join(c if ord(c) < 128 else "?" for c in (text or ""))


def _count_pages(data: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(data)).pages)


def _plan_units(
    sections: List[Any],
    structure: Any,
) -> List[Tuple[str, List[Tuple[str, List[Tuple[str, List[Dict[str, Any]]]]]]]]:
    """Order the render plan from the spine when available.

    Sections already arrive in spine order, but synthesis dedupe can merge
    identical prose across topics, leaving a spine topic with no surviving
    section. When the structure is supplied, every spine topic is planned
    (empty ones render their heading plus the honest dedupe pointer);
    without a structure the plan falls back to the sections as grouped.
    """
    grouped = _group_units(sections, structure)
    if structure is None:
        return grouped
    by_unit: Dict[str, List[Any]] = {}
    for unit_title, topics in grouped:
        by_unit.setdefault(unit_title, []).extend(topics)
    plan: List[Tuple[str, List[Any]]] = []
    for unit in _struct_units(structure):
        unit_title = _struct_field(unit, "title")
        planned_topics: List[Any] = []
        available = {t[0]: t for t in by_unit.get(unit_title, [])}
        for topic in _struct_topics(unit):
            topic_title = _struct_field(topic, "title")
            match = available.get(topic_title)
            if match is None:
                planned_topics.append((topic_title, []))
            else:
                planned_topics.append(match)
        # Sections for units the spine walk missed (defensive): keep them,
        # never drop rendered content.
        planned_topics.extend(
            t for name, t in available.items()
            if all(name != p[0] for p in planned_topics)
        )
        plan.append((unit_title, planned_topics))
    if not plan:
        return grouped
    return plan


def _struct_field(item: Any, name: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _struct_units(structure: Any) -> List[Any]:
    return list(_struct_field(structure, "units", []) or [])


def _struct_topics(unit: Any) -> List[Any]:
    return list(_struct_field(unit, "topics", []) or [])


def _group_units(
    sections: List[Any],
    structure: Any,
) -> List[Tuple[str, List[Tuple[str, List[Tuple[str, List[Dict[str, Any]]]]]]]]:
    """Nest sections into [(unit, [(topic, [(subtopic, [paras])])])] in order.

    Spine order is authoritative: units/topics follow first-seen section
    order (the synthesis already emits spine order). Paragraphs carry
    their citations for the Sources line.
    """
    units: List[Tuple[str, List[Any]]] = []
    unit_index: Dict[str, int] = {}
    for section in sections:
        if isinstance(section, dict):
            unit = str(section.get("unit_title") or "Notes")
            topic = str(section.get("topic_title") or _section_heading(section))
            sub = str(section.get("subtopic_title") or "")
        else:
            unit = str(getattr(section, "unit_title", "") or "Notes")
            topic = str(getattr(section, "topic_title", "") or _section_heading(section))
            sub = str(getattr(section, "subtopic_title", "") or "")
        if unit not in unit_index:
            unit_index[unit] = len(units)
            units.append((unit, []))
        topics = units[unit_index[unit]][1]
        match = next((t for t in topics if t[0] == topic), None)
        if match is None:
            match = (topic, [])
            topics.append(match)
        subs = match[1]
        target = next((s for s in subs if s[0] == sub), None)
        if target is None:
            target = (sub, [])
            subs.append(target)
        target[1].append({
            "text": _section_text(section),
            "cites": _section_cites(section),
        })
    void_structure = structure  # hierarchy comes from ordered sections; kept for signature parity.
    return units


def _render_paras(
    paras: List[Dict[str, Any]],
    unit_title: str,
    topic_title: str,
    chunk_pages: Dict[str, int],
    extras: Dict[Tuple[str, str], Dict[str, Any]],
    body: Any,
    code: Any,
    callout: Any,
    callout_head: Any,
    cite_style: Any,
    frame_w: float,
) -> List[Any]:
    """Render one subtopic's paragraphs: prose, formulas, callouts, cites."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm

    story: List[Any] = []
    extra = extras.get((unit_title, topic_title), {})
    key_terms = [str(t) for t in (extra.get("key_terms") or [])]
    extra_formulas = [str(f) for f in (extra.get("formulas") or [])]
    extra_steps = [str(s) for s in (extra.get("steps") or [])]
    extra_table = extra.get("table")

    if key_terms:
        story.append(Paragraph(_escape("Key terms"), callout_head))
        story.append(_terms_table(key_terms, callout, frame_w))
        story.append(Spacer(1, 2 * mm))

    for formula in extra_formulas:
        story.append(Paragraph(_escape(formula), code))

    if extra_steps:
        story.append(Paragraph(_escape("Steps"), callout_head))
        for pos, step in enumerate(extra_steps[:12], 1):
            for piece in _split_long_bullet(str(step)):
                story.append(Paragraph(_escape(f"{pos}. {piece}"), callout))

    if isinstance(extra_table, list) and extra_table:
        story.append(Spacer(1, 2 * mm))
        story.append(_data_table(extra_table, callout, frame_w))

    examples: List[str] = []
    exam_notes: List[str] = []
    cite_no = 0
    for para in paras:
        for sentence in _sentences(para["text"]):
            if _FORMULA_RE.search(sentence):
                story.append(Paragraph(_escape(sentence), code))
            elif _EXAMPLE_RE.search(sentence):
                examples.append(sentence)
                story.append(Paragraph(_escape(sentence), body))
            else:
                if _EXAM_RE.search(sentence):
                    exam_notes.append(sentence)
                story.append(Paragraph(_escape(sentence), body))
        for cite in para["cites"]:
            cite_no += 1
            page = chunk_pages.get(cite["chunk_id"])
            if page is not None and not isinstance(page, int):
                page = None
            story.append(Paragraph(
                _escape("Sources: " + _cite_label(cite["chunk_id"], cite["span"], page, cite_no - 1)),
                cite_style,
            ))

    if examples:
        story.append(Spacer(1, 2 * mm))
        story.append(_callout_box("Example", examples[:3], callout, callout_head,
                                 colors.HexColor("#eff6ff"), colors.HexColor("#bfdbfe"), frame_w))
    if exam_notes:
        story.append(Spacer(1, 2 * mm))
        story.append(_callout_box("Exam focus", exam_notes[:4], callout, callout_head,
                                 colors.HexColor("#fefce8"), colors.HexColor("#fde68a"), frame_w))
    return story


def _terms_table(terms: List[str], style: Any, frame_w: float) -> Any:
    """Key terms as a real Table (grid, wrapped cells, never clipped)."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    cols = 3
    rows: List[List[Any]] = []
    for pos in range(0, len(terms), cols):
        chunk = terms[pos:pos + cols]
        while len(chunk) < cols:
            chunk.append("")
        rows.append([Paragraph(_escape(t), style) for t in chunk])
    table = Table(rows, colWidths=[frame_w / cols] * cols)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafaf9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6d3d1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e7e5e4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _data_table(rows: List[Any], style: Any, frame_w: float) -> Any:
    """Explicit tabular body blocks survive as tables (first row = header)."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    cleaned = [[str(cell) for cell in (row if isinstance(row, list) else [row])]
               for row in rows if row]
    if not cleaned:
        return Paragraph("", style)
    width = max(len(r) for r in cleaned)
    grid = [r + [""] * (width - len(r)) for r in cleaned]
    body_rows = [[Paragraph(_escape(c), style) for c in r] for r in grid]
    table = Table(body_rows, colWidths=[frame_w / width] * width, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5b4fc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d2fe")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _callout_box(title: str, lines: List[str], style: Any, head_style: Any,
                 bg: Any, border: Any, frame_w: float) -> Any:
    from reportlab.platypus import Paragraph, Table, TableStyle

    inner = [Paragraph(f"<b>{_escape(title)}</b>", head_style)]
    for line in lines:
        safe, _ = _safe_text(line)
        for piece in _split_long_bullet(safe):
            inner.append(Paragraph(_escape(f"- {piece}"), style))
    box = Table([[inner]], colWidths=[frame_w - 4])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.75, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return box


def _make_numbered_canvas_class() -> Any:
    """Build the numbered-pages Canvas subclass (module-level shape)."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas as _Base

    class _Numbered(_Base):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self._saved_pages: List[Dict[str, Any]] = []

        def showPage(self) -> None:
            # Collect the page state only; the base showPage() would emit
            # the page here AND again during save() (12 content pages ->
            # 24).  Resetting the page state with _startPage() while
            # collecting is the canonical ReportLab NumberedCanvas idiom:
            # exactly one emission per page happens in save().
            self._saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = max(1, len(self._saved_pages))
            for state in self._saved_pages:
                self.__dict__.update(state)
                self._draw_page_number(total)
                super().showPage()
            super().save()

        def _draw_page_number(self, total: int) -> None:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#78716c"))
            current = getattr(self, "_pageNumber", 1)
            self.drawRightString(
                self._pagesize[0] - 20 * mm,
                12 * mm,
                f"Page {current} of {total}",
            )
            self.restoreState()

    return _Numbered


_NumberedCanvas = _make_numbered_canvas_class()


__all__ = [
    "BRAND",
    "BRAND_SUBTITLE",
    "NotesPdfReport",
    "render_notes_pdf",
    "notes_pdf_storage_hint",
]
