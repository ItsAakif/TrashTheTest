"""
Semantic Chunking Implementation

Core chunking engine that transforms extracted text into semantically meaningful chunks
with structure awareness, memory-safe processing, and metadata preservation.
"""

import re
import uuid
import hashlib
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

from services.text_utils import (
    collapse_letter_spacing,
    collapse_spaces,
    is_page_number_line,
    strip_control_chars,
)

from .types import (
    ChunkMetadata,
    ChunkResult,
    DocumentChunkingInput,
    ChunkingProgress,
    ChunkingResult,
    ChunkingConfig,
)


# Structural fallbacks that must NEVER be produced by the detector
# (ADR-008).  Stored lower-case; matched case-insensitively.
BANNED_HEADINGS = {"introduction", "conclusion", "full document"}

# Minor words permitted un-capitalised inside a title-case heading line.
_TITLE_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "at", "from", "as", "is", "are", "vs", "via",
}


class SemanticChunker:
    """
    Semantic chunking engine that splits document text into meaningful chunks.
    
    Key features:
    - Structure-aware splitting (headings, paragraphs, sections)
    - Memory-safe batch processing for large documents
    - Token counting for downstream processing
    - Content hashing for embedding caching
    - Resume capability for interrupted processing
    
    This module does NOT generate embeddings or call LLMs.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving paragraph breaks.

        The heading detector (_detect_structure) runs on this output, so
        intra-line whitespace is collapsed but paragraph breaks (\\n\\n)
        are preserved.  End-of-line hyphenation is dehyphenated; control
        chars are stripped and letter spacing collapsed only when it
        dominates a line (both via the shared services.text_utils helpers).
        """
        if not text:
            return ""
        # 1. Dehyphenate end-of-line breaks: "exam-\\nple" -> "example".
        text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
        # 2. Strip control chars (bullet glyphs become "• ").
        text = strip_control_chars(text)
        # 3. Collapse letter spacing on lines that are mostly space-
        #    separated single letters (corrupted text layer; >=60% of
        #    tokens are single chars).
        text = collapse_letter_spacing(text)
        # 4. Collapse intra-line whitespace; preserve paragraph breaks.
        return collapse_spaces(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple approximation.
        
        This uses a rough heuristic: ~4 characters per token on average.
        More accurate counting can be done with tiktoken if needed.
        """
        return len(text) // 4
    
    def _generate_content_hash(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Generate content hash for caching embeddings.

        Hash is content-addressed: normalized chunk text only (not position),
        so reprocessing is idempotent and (document_id, content_hash) is a
        stable upsert key. 64-hex sha256.
        """
        import unicodedata

        normalized = unicodedata.normalize("NFKC", content or "").strip()
        normalized = re.sub(r"\s+", " ", normalized).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    
    # -- structure detection (ADR-008) --------------------------------
    #
    # The detector NEVER invents a structural fallback name.  Documents
    # without detectable headings become page-bounded sections titled
    # "Pages <a>-<b>" with a low confidence; the banned literals are only
    # ever rejected, never produced.

    def _is_valid_topic_title(self, title: str) -> bool:
        """Reject literal fallbacks, sentence fragments, and bad shapes.

        Used by both the chunker and the academic spine to decide whether
        a section name is safe to surface to the UI.
        """
        if not title:
            return False
        t = title.strip()
        if not t or len(t) > 120 or len(t) < 2:
            return False
        words = t.split()
        if not words or len(words) > 8:
            return False
        if t[-1:] in {".", "!", "?", ";", ","}:
            return False
        if t.lower() in BANNED_HEADINGS:
            return False
        if t.lower().startswith("untitled"):
            return False
        return True

    def _is_title_case_line(self, line: str) -> bool:
        """Rule 6 shape: short, capitalized words (or minor stopwords)."""
        if not line or len(line) > 80:
            return False
        if line[-1:] in {".", "!", "?", ";", ",", ":"}:
            return False
        words = line.split()
        if not words or len(words) > 8:
            return False
        capitalized = 0
        for word in words:
            if word[0].isupper():
                capitalized += 1
            elif word.lower() in _TITLE_STOPWORDS:
                continue
            else:
                return False
        return capitalized >= 1

    def _heading_match(self, line: str) -> Optional[tuple]:
        """Rules 2..8 (first match wins).  Returns (title, method, conf)."""
        if not line:
            return None
        # 2. unit/chapter keyword
        if re.match(r"^(unit|chapter|part|module|section)\s+\S+", line,
                    re.IGNORECASE):
            return (line, "unit_keyword", 0.9)
        # 3. numbered heading (1., 1.1, 1.1.1)
        if re.match(r"^(?:\d+\.){1,3}\s+\S", line):
            return (line, "numbered", 0.85)
        # 4. markdown heading
        md = re.match(r"^#{1,6}\s+(\S.*)$", line)
        if md:
            return (md.group(1).strip(), "markdown", 0.9)
        # 5. ALL CAPS line, 4-80 chars, no terminal punctuation
        if (4 <= len(line) <= 80
                and line[-1:] not in {".", "!", "?", ";", ",", ":"}
                and line == line.upper()
                and any(ch.isalpha() for ch in line)):
            return (line, "all_caps", 0.8)
        # 6. title-case short line
        if self._is_title_case_line(line):
            return (line, "title_case", 0.7)
        # 7. colon-terminated short line
        if line.endswith(":") and len(line) <= 80:
            return (line, "colon", 0.75)
        # 8. numbered list label (1) Foo / 1. Foo)
        if 3 <= len(line) <= 80 and re.match(r"^\d+[\.)]\s+\S", line):
            return (line, "list_label", 0.6)
        return None

    def _is_boilerplate_line(self, line: str, boilerplate: set) -> bool:
        """True for running-header lines and page/bare-number lines.

        Such lines are dropped before heading candidacy and are never
        emitted as section headings.
        """
        if not line:
            return False
        norm = collapse_spaces(collapse_letter_spacing(line))
        if norm and norm in boilerplate:
            return True
        return is_page_number_line(line) or bool(re.fullmatch(r"\d+", line.strip()))

    def _detect_structure(
        self,
        text: str,
        page_spans: Optional[List[Dict[str, Any]]] = None,
        boilerplate: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Return ordered sections with method + confidence.

        Each section dict: {"heading": str|None, "start_offset": int,
        "end_offset": int, "method": str, "confidence": float}.  No banned
        literal fallbacks; a heading-less document returns page-bounded
        sections.

        Detection rules (first match per line wins):
          1. font/position hints from page_spans (span heading confirmed
             against the line itself)
          2. unit/chapter keyword: ^(unit|chapter|part|module|section) ...
          3. numbered: ^(?:\\d+\\.){1,3}\\s+\\S
          4. markdown: ^#{1,6}\\s+\\S
          5. ALL CAPS line, 4-80 chars, no terminal punctuation
          6. title-case short line, <=8 words, <=80 chars, no terminal
          7. colon-terminated: ends with :, <=80 chars
          8. numbered list label: ^\\d+[\\.\\)]\\s+\\S (3-80 chars)

        If fewer than 2 sections are detected AND the detected headings
        cover less than 2% of the text, the list is cleared and
        page-bounded sections are emitted instead.
        """
        if not text:
            return []
        boilerplate = {str(b) for b in (boilerplate or set())}
        total_len = len(text)

        def _int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # b. normalised, sorted page spans for the offset -> page lookup.
        spans: List[Dict[str, Any]] = []
        for raw in page_spans or []:
            if not isinstance(raw, dict):
                continue
            start = _int(raw.get("start"))
            end = _int(raw.get("end"))
            if end > start:
                spans.append({**raw, "start": start, "end": end})
        spans.sort(key=lambda item: item["start"])

        def _dominant_span(p_start: int, p_end: int) -> Dict[str, Any]:
            """Most-overlapping page span for [p_start, p_end)."""
            best: Dict[str, Any] = {}
            best_overlap = 0
            for span in spans:
                overlap = (min(span["end"], p_end)
                           - max(span["start"], p_start))
                if overlap > best_overlap:
                    best, best_overlap = span, overlap
            if best:
                return best
            # Nearest preceding span, else the first span (documents whose
            # offsets drift after cleaning still resolve an honest page).
            preceding = [s for s in spans if s["start"] <= p_start]
            if preceding:
                return preceding[-1]
            return spans[0] if spans else {}

        def _page_label(start_offset: int, end_offset: int) -> str:
            """"Pages {a}-{b}" from the dominant spans of the range."""
            head = _dominant_span(start_offset,
                                  max(start_offset + 1, end_offset))
            tail = _dominant_span(max(start_offset, end_offset - 1),
                                  max(start_offset + 1, end_offset))
            first = _int(head.get("page"), 0) or None
            last = _int(tail.get("page"), 0) or None
            if first is None:
                first = last
            if last is None:
                last = first
            if first is None or last is None:
                return "Pages 1-1"
            lo, hi = (first, last) if first <= last else (last, first)
            return f"Pages {lo}-{hi}"

        # a. split on paragraph breaks, preserving character offsets.
        paragraphs: List[tuple] = []
        offset = 0
        for index, part in enumerate(re.split(r"(\n\n+)", text)):
            if index % 2 == 0 and part.strip():
                lead = len(part) - len(part.lstrip())
                end = len(part.rstrip())
                paragraphs.append((offset + lead, part[lead:end]))
            offset += len(part)

        # c..e. heading candidates, first match per paragraph wins.
        detected: List[Dict[str, Any]] = []
        for p_start, para in paragraphs:
            first_line = para.split("\n", 1)[0].strip()
            if not first_line or len(first_line) > 120:
                continue
            # Running headers / page numbers are never heading candidates.
            if self._is_boilerplate_line(first_line, boilerplate):
                continue
            heading: Optional[str] = None
            method: Optional[str] = None
            confidence: Optional[float] = None
            # 1. font hint: only when the dominant span heading is
            #    confirmed against the line itself (span offsets drift on
            #    some extractions, so an unconfirmed hint is ignored).
            span = _dominant_span(p_start, p_start + len(para))
            span_heading = span.get("heading")
            if (isinstance(span_heading, str)
                    and span_heading.strip() == first_line):
                heading = span_heading.strip()
                method = "font_hint"
                confidence = _float(span.get("confidence"))
                if confidence is None:
                    confidence = 0.9
                confidence = min(0.95, max(0.5, confidence))
            else:
                # 2..8.
                match = self._heading_match(first_line)
                if match is None and len(para) <= 120:
                    whole = para.strip()
                    if self._is_title_case_line(whole):
                        match = (whole, "title_case", 0.7)
                if match is not None:
                    heading, method, confidence = match
            if heading is None or method is None:
                continue
            detected.append({
                "heading": heading,
                "start_offset": p_start,
                "end_offset": min(total_len, p_start + len(para)),
                "method": method,
                "confidence": float(confidence if confidence is not None
                                    else 0.5),
            })

        # Sections partition the text: a heading starts a section, the
        # section ends where the next heading starts.
        sections: List[Dict[str, Any]] = []
        for index, candidate in enumerate(detected):
            start = candidate["start_offset"] if index else 0
            end = (detected[index + 1]["start_offset"]
                   if index + 1 < len(detected) else total_len)
            if end > start:
                sections.append({**candidate,
                                 "start_offset": start,
                                 "end_offset": end})

        heading_chars = sum(len(s["heading"]) for s in sections
                            if s["heading"])
        if len(sections) < 2 and (
                heading_chars / max(1, total_len)) < 0.02:
            # f. no usable structure -> page-bounded sections.
            sections = self._page_bounded_sections(total_len, spans)

        # g. title post-processing: heading-less sections get a page
        #    label; shape-broken titles are renamed (a genuine literal
        #    heading found by the detector is kept as the section name).
        for section in sections:
            if section["heading"] is None:
                section["heading"] = _page_label(section["start_offset"],
                                                 section["end_offset"])
                section["method"] = "page_bounded"
                section["confidence"] = 0.3
            else:
                title = section["heading"].strip()
                banned = (
                    title.rstrip(":").strip().lower() in BANNED_HEADINGS
                    or self._is_boilerplate_line(title, boilerplate)
                )
                if banned or not self._is_valid_topic_title(title):
                    # Banned literals and shape-broken titles are never
                    # surfaced as topic names: keep the source heading for
                    # provenance and fall back to a page-bounded label so
                    # validation treats the structure as degraded instead of
                    # failing the notes run with a hard topic_mismatch.
                    section["original_heading"] = section["heading"]
                    section["heading"] = _page_label(section["start_offset"],
                                                     section["end_offset"])
                    section["method"] = "page_bounded"
                    section["confidence"] = 0.3

        # 4. ordered, first starts at 0, last ends at the text end.
        sections.sort(key=lambda item: item["start_offset"])
        if sections:
            sections[0]["start_offset"] = 0
            sections[-1]["end_offset"] = total_len
            for index in range(1, len(sections)):
                prev_end = sections[index - 1]["end_offset"]
                if sections[index]["start_offset"] < prev_end:
                    sections[index]["start_offset"] = prev_end
                if sections[index]["end_offset"] < sections[index]["start_offset"]:
                    sections[index]["end_offset"] = sections[index]["start_offset"]
        return sections

    def _page_bounded_sections(
        self,
        text_len: int,
        spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Page-granular fallback sections (no invented topic names).

        Emits one section per page span (gaps included) so the sections
        still partition the text; when no page spans exist a single
        low-confidence section covering the document is emitted.
        """
        if text_len <= 0:
            return []
        if not spans:
            return [{"heading": None, "start_offset": 0,
                     "end_offset": text_len, "method": "page_bounded",
                     "confidence": 0.1}]

        built: List[Dict[str, Any]] = []

        def _add(start: int, end: int, confidence: float) -> None:
            a = max(0, min(start, text_len))
            b = max(a, min(end, text_len))
            if b > a:
                built.append({"heading": None, "start_offset": a,
                              "end_offset": b, "method": "page_bounded",
                              "confidence": confidence})

        cursor = 0
        for span in spans:
            if span["start"] > cursor:
                _add(cursor, span["start"], 0.3)
            start = max(cursor, span["start"])
            end = max(start, span["end"])
            _add(start, end, 0.3)
            cursor = max(cursor, min(end, text_len))
        _add(cursor, text_len, 0.3)
        if not built:
            built.append({"heading": None, "start_offset": 0,
                          "end_offset": text_len, "method": "page_bounded",
                          "confidence": 0.1})
        return built

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraph boundaries.
        
        Uses double newlines as primary separator.
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\n+', text)
        
        # Filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        Split a long paragraph by sentences when it exceeds max size.
        
        Uses sentence-ending punctuation as split points.
        """
        if len(text) <= self.config.max_chunk_size:
            return [text]
        
        # Split by sentence endings
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.config.max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk, but try to fit if sentence is too long
                if len(sentence) > self.config.max_chunk_size:
                    # Sentence itself is too long, split by clauses
                    clauses = re.split(r'[,:;]\s+', sentence)
                    for clause in clauses:
                        if len(current_chunk) + len(clause) <= self.config.max_chunk_size:
                            current_chunk += " " + clause if current_chunk else clause
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = clause
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:self.config.max_chunk_size]]
    
    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small.
        
        Combines adjacent chunks that fall below min_chunk_size.
        """
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            if len(current) < self.config.min_chunk_size:
                # Merge with next chunk
                current = current + " " + next_chunk
            else:
                merged.append(current)
                current = next_chunk
        
        # Don't forget the last chunk
        if current:
            if len(current) < self.config.min_chunk_size and merged:
                # Merge with previous if possible
                merged[-1] = merged[-1] + " " + current
            else:
                merged.append(current)
        
        return merged
    
    def _merge_index_groups(self, sizes: List[int]) -> List[List[int]]:
        """Mirror `_merge_short_chunks`, returning the original chunk
        indexes behind each merged chunk.

        Used to re-bind section metadata after a merge so merged chunks
        keep the provenance of their first constituent chunk.
        """
        if not sizes:
            return []
        groups: List[List[int]] = []
        current_group = [0]
        current_len = sizes[0]
        for index in range(1, len(sizes)):
            if current_len < self.config.min_chunk_size:
                current_group.append(index)
                current_len = current_len + 1 + sizes[index]
            else:
                groups.append(current_group)
                current_group = [index]
                current_len = sizes[index]
        if current_group:
            if current_len < self.config.min_chunk_size and groups:
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)
        return groups
    
    def _create_chunks_from_text(
        self,
        text: str,
        document_id: str,
        start_page: int = 1,
        start_position: int = 0,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        page_spans: Optional[List[Dict[str, Any]]] = None,
        boilerplate_lines: Optional[set] = None,
    ) -> List[ChunkResult]:
        """
        Create chunks from cleaned text with structure awareness.
        Provenance (page_start/page_end/method/confidence) is derived from
        page_spans (document_pages offsets) when supplied; otherwise the
        chunk keeps start_page and section/heading from structure parsing.
        """
        chunks = []
        position = start_position
        spans = page_spans or []

        def _span_for(offset: int) -> Dict[str, Any]:
            for span in spans:
                if span.get("start", 0) <= offset < span.get("end", 0):
                    return span
            return {}

        def _make_metadata(section_heading: str, offset: int = 0,
                           *,
                           section: Optional[Dict[str, Any]] = None
                           ) -> ChunkMetadata:
            span = _span_for(offset)
            page_no = int(span.get("page", start_page) or start_page)
            method = (section or {}).get("method")
            if method is None:
                method = span.get("method")
            confidence = (section or {}).get("confidence")
            if confidence is None:
                confidence = span.get("confidence")
            if confidence is not None:
                try:
                    confidence = min(1.0, max(0.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = None
            return ChunkMetadata(
                page_number=page_no,
                section=section_heading,
                heading=section_heading if self._is_valid_topic_title(section_heading) else None,
                position=position,
                document_id=document_id,
                project_id=project_id,
                user_id=user_id,
                page_start=page_no,
                page_end=page_no,
                section_path=[section_heading] if section_heading else [],
                source=source,
                method=span.get("method"),
                confidence=span.get("confidence"),
                structure_method=method,
                structure_confidence=confidence,
            )
        
        # ADR-008: the detector runs on the cleaned text; sections
        # partition it, so slicing cleaned[start:end] never loses text.
        cleaned = self._clean_text(text)
        structure = self._detect_structure(
            cleaned, spans, boilerplate=boilerplate_lines,
        )
        # position -> section (T003): the merge pass re-binds metadata.
        position_to_section: Dict[int, Dict[str, Any]] = {}
        position_to_offset: Dict[int, int] = {}

        for section in structure:
            section_heading = section["heading"]
            section_start = section["start_offset"]
            first_position = position
            section_content = cleaned[section["start_offset"]:section["end_offset"]]
            # Then split each section by paragraphs
            paragraphs = self._split_by_paragraphs(section_content)
            
            current_chunk_content = ""
            
            for para in paragraphs:
                # If paragraph is too long, split it
                if len(para) > self.config.max_chunk_size:
                    # First save current accumulated content
                    if current_chunk_content:
                        sub_chunks = self._split_long_paragraph(current_chunk_content)
                        for sc in sub_chunks:
                            metadata = _make_metadata(section_heading, section_start, section=section)
                            chunks.append(ChunkResult(
                                content=sc,
                                metadata=metadata,
                                token_count=self._estimate_tokens(sc),
                                content_hash=self._generate_content_hash(sc, metadata.model_dump())
                            ))
                            position += 1
                        current_chunk_content = ""
                    
                    # Now split the long paragraph
                    para_chunks = self._split_long_paragraph(para)
                    for pc in para_chunks:
                        metadata = _make_metadata(section_heading, section_start, section=section)
                        chunks.append(ChunkResult(
                            content=pc,
                            metadata=metadata,
                            token_count=self._estimate_tokens(pc),
                            content_hash=self._generate_content_hash(pc, metadata.model_dump())
                        ))
                        position += 1
                else:
                    # Check if adding this paragraph would exceed max size
                    if len(current_chunk_content) + len(para) > self.config.max_chunk_size:
                        # Save current chunk
                        if current_chunk_content:
                            sub_chunks = self._split_long_paragraph(current_chunk_content)
                            for sc in sub_chunks:
                                metadata = _make_metadata(section_heading, section_start, section=section)
                                chunks.append(ChunkResult(
                                    content=sc,
                                    metadata=metadata,
                                    token_count=self._estimate_tokens(sc),
                                    content_hash=self._generate_content_hash(sc, metadata.model_dump())
                                ))
                                position += 1
                        current_chunk_content = para
                    else:
                        # Add to current chunk
                        current_chunk_content = (
                            current_chunk_content + "\n\n" + para
                            if current_chunk_content
                            else para
                        )
            
            # Handle remaining content in current chunk
            if current_chunk_content:
                sub_chunks = self._split_long_paragraph(current_chunk_content)
                for sc in sub_chunks:
                    metadata = _make_metadata(section_heading, section_start, section=section)
                    chunks.append(ChunkResult(
                        content=sc,
                        metadata=metadata,
                        token_count=self._estimate_tokens(sc),
                        content_hash=self._generate_content_hash(sc, metadata.model_dump())
                    ))
                    position += 1

            # T003: every chunk created inside this section binds back to
            # it, so the merge pass can re-attach method/confidence.
            for bound_position in range(first_position, position):
                position_to_section[bound_position] = section
                position_to_offset[bound_position] = section_start
        
        # Merge short chunks
        if len(chunks) > 1:
            chunk_texts = [c.content for c in chunks]
            merged_texts = self._merge_short_chunks(chunk_texts)
            
            # Rebuild chunks with new merged content
            if len(merged_texts) != len(chunks):
                groups = self._merge_index_groups(
                    [len(c.content) for c in chunks])
                if len(groups) != len(merged_texts):
                    groups = [[min(i, len(chunks) - 1)]
                              for i in range(len(merged_texts))]
                new_chunks = []
                pos = start_position
                for i, text in enumerate(merged_texts):
                    original = chunks[min(groups[i][0], len(chunks) - 1)]
                    section = position_to_section.get(
                        original.metadata.position)
                    offset = position_to_offset.get(
                        original.metadata.position, 0)
                    _m = _make_metadata(original.metadata.section, offset,
                                        section=section)
                    structure_method = (_m.structure_method
                                        or original.metadata.structure_method)
                    structure_confidence = (
                        _m.structure_confidence
                        if _m.structure_confidence is not None
                        else original.metadata.structure_confidence)
                    metadata = ChunkMetadata(
                        page_number=_m.page_number, section=_m.section, heading=_m.heading,
                        position=pos, document_id=document_id, project_id=_m.project_id,
                        user_id=_m.user_id, page_start=_m.page_start, page_end=_m.page_end,
                        section_path=_m.section_path, source=_m.source,
                        method=_m.method, confidence=_m.confidence,
                        structure_method=structure_method,
                        structure_confidence=structure_confidence)
                    new_chunks.append(ChunkResult(
                        content=text,
                        metadata=metadata,
                        token_count=self._estimate_tokens(text),
                        content_hash=self._generate_content_hash(text, metadata.model_dump())
                    ))
                    pos += 1
                return new_chunks
        
        return chunks
    
    def chunk_document(
        self,
        input_data: DocumentChunkingInput,
        start_position: int = 0
    ) -> ChunkingResult:
        """
        Chunk a complete document.
        
        For smaller documents that fit in memory.
        For large documents, use chunk_document_streaming().
        """
        start_time = time.time()
        
        # Create chunks from text
        chunks = self._create_chunks_from_text(
            text=input_data.full_text,
            document_id=input_data.document_id,
            start_page=1,
            start_position=start_position,
            project_id=getattr(input_data, 'project_id', None),
            user_id=getattr(input_data, 'user_id', None),
            source=getattr(input_data, 'source', None),
            page_spans=getattr(input_data, 'page_spans', None),
            boilerplate_lines=getattr(input_data, 'boilerplate_lines', None),
        )
        
        # Calculate total tokens
        total_tokens = sum(c.token_count for c in chunks)
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            document_id=input_data.document_id,
            project_id=input_data.project_id,
            total_chunks=len(chunks),
            chunks=chunks,
            total_tokens=total_tokens,
            processing_time=processing_time,
            is_complete=True
        )
    
    async def chunk_document_streaming(
        self,
        input_data: DocumentChunkingInput,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[tuple[List[ChunkResult], ChunkingProgress], None]:
        """
        Stream chunks in batches for large documents.
        
        Processes document in batches to maintain memory safety.
        Yields (chunks, progress) tuples for each batch.
        """
        total_text_length = len(input_data.full_text)
        batch_size_chars = self.config.batch_size * 2000  # Approx chars per page
        
        # Calculate total batches
        total_batches = (total_text_length + batch_size_chars - 1) // batch_size_chars
        
        current_position = 0
        chunks_created = 0
        
        for batch_num in range(total_batches):
            start_char = batch_num * batch_size_chars
            end_char = min((batch_num + 1) * batch_size_chars, total_text_length)
            
            # Get batch text (with overlap)
            if batch_num > 0:
                # Include overlap from previous batch
                start_char = start_char - self.config.overlap_size
            
            batch_text = input_data.full_text[start_char:end_char]
            
            # Process batch
            batch_chunks = self._create_chunks_from_text(
                text=batch_text,
                document_id=input_data.document_id,
                start_page=batch_num + 1,
                start_position=current_position,
                boilerplate_lines=getattr(input_data, 'boilerplate_lines', None),
            )
            
            # Update position counter
            current_position += len(batch_chunks)
            chunks_created += len(batch_chunks)
            
            # Create progress
            progress = ChunkingProgress(
                document_id=input_data.document_id,
                current_batch=batch_num + 1,
                total_batches=total_batches,
                chunks_created=chunks_created,
                status="chunking",
                progress_percent=((batch_num + 1) / total_batches) * 100,
                message=f"Processing batch {batch_num + 1} of {total_batches}",
                last_processed_position=current_position
            )
            
            # Yield chunks and progress
            yield batch_chunks, progress
            
            # Call progress callback if provided
            if progress_callback:
                await progress_callback(progress)
        
        # Final progress update
        final_progress = ChunkingProgress(
            document_id=input_data.document_id,
            current_batch=total_batches,
            total_batches=total_batches,
            chunks_created=chunks_created,
            status="completed",
            progress_percent=100.0,
            message="Chunking completed",
            last_processed_position=current_position
        )
        
        if progress_callback:
            await progress_callback(final_progress)


# Default chunker instance
semantic_chunker = SemanticChunker()