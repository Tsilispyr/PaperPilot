"""Two chunking strategies: fixed-size (v1) and section-aware (v2).

Both share the same Chunk schema so downstream code is version-agnostic.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Iterable

import tiktoken

from paperpilot.config import settings
from paperpilot.ingest.schema import Chunk, PaperMetadata, SectionType

logger = logging.getLogger(__name__)


        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
_ENC = tiktoken.get_encoding("cl100k_base")


def _tok_count(text: str) -> int:
    return len(_ENC.encode(text))


def _det_uuid(*parts: str) -> str:
    """Deterministic UUID5 — Qdrant accepts UUID strings as point ids."""
    base = "::".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
SECTION_PATTERNS: list[tuple[re.Pattern, SectionType]] = [
    (re.compile(r"^\s*abstract\b", re.IGNORECASE), "abstract"),
    (re.compile(r"^\s*(introduction|motivation)\b", re.IGNORECASE), "introduction"),
    (re.compile(r"^\s*(background|related work|preliminaries)\b", re.IGNORECASE), "background"),
    (re.compile(r"^\s*(method|methods|approach|methodology|model|architecture)\b", re.IGNORECASE), "method"),
    (re.compile(r"^\s*(experiments?|experimental setup|setup|implementation)\b", re.IGNORECASE), "experiments"),
    (re.compile(r"^\s*(results?|evaluation|findings)\b", re.IGNORECASE), "results"),
    (re.compile(r"^\s*(discussion|analysis|ablation)\b", re.IGNORECASE), "discussion"),
    (re.compile(r"^\s*conclusion(s)?\b", re.IGNORECASE), "conclusion"),
    (re.compile(r"^\s*references\b", re.IGNORECASE), "references"),
    (re.compile(r"^\s*appendix\b", re.IGNORECASE), "appendix"),
]

HEADER_RX = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


_STRIP_MD = re.compile(r"[*_`]")
_STRIP_NUM = re.compile(r"^\d+(\.\d+)*\.?\s+")


def _classify_section(title: str) -> SectionType:
    # pymupdf4llm renders headers as "**1.2 Introduction**" — strip bold markers and
    # leading numbering so patterns like r"^\s*introduction\b" can match.
    clean = _STRIP_MD.sub("", title).strip()
    clean = _STRIP_NUM.sub("", clean).strip()
    for rx, kind in SECTION_PATTERNS:
        if rx.search(clean):
            return kind
    return "other"


def _arxiv_abs_url(paper_id: str) -> str:
    # paper_id looks like "2401.01234v2" → strip version suffix for canonical url
    base = re.sub(r"v\d+$", "", paper_id)
    return f"https://arxiv.org/abs/{base}"



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
class FixedSizeChunker:
    """Sliding window over the full markdown text, ignoring structure."""

    def __init__(self, size_tokens: int | None = None, overlap_tokens: int | None = None):
        self.size = size_tokens or settings.chunk_size_tokens
        self.overlap = overlap_tokens or settings.chunk_overlap_tokens
        if self.overlap >= self.size:
            raise ValueError("overlap must be < size")

    def chunk(self, meta: PaperMetadata, text: str) -> list[Chunk]:
        tokens = _ENC.encode(text)
        out: list[Chunk] = []
        step = self.size - self.overlap
        i, idx = 0, 0
        while i < len(tokens):
            window = tokens[i : i + self.size]
            piece = _ENC.decode(window).strip()
            if piece:
                out.append(
                    Chunk(
                        id_=_det_uuid("v1", meta.paper_id, str(idx)),
                        paper_id=meta.paper_id,
                        title=meta.title,
                        authors=meta.authors,
                        year=meta.year,
                        primary_category=meta.primary_category,
                        section_type="other",
                        section_title=None,
                        chunk_index=idx,
                        text=piece,
                        pdf_url=meta.pdf_url,
                        source_url=_arxiv_abs_url(meta.paper_id),
                    )
                )
                idx += 1
            i += step
        return out



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
class SectionAwareChunker:
    """Split on markdown headers; subdivide oversized sections with the same window."""

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_tokens: int = 80,
    ):
        self.max = max_tokens or settings.chunk_size_tokens
        self.overlap = overlap_tokens or settings.chunk_overlap_tokens
        self.min = min_tokens

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """Return [(title, body), ...] split on markdown headers (#, ##, ###)."""
        positions = [(m.start(), m.end(), m.group(2).strip()) for m in HEADER_RX.finditer(text)]
        if not positions:
            return [("Body", text)]
        sections = []
        # Pre-amble (before first header) treated as 'Header'
        if positions[0][0] > 0:
            pre = text[: positions[0][0]].strip()
            if pre:
                sections.append(("Header", pre))
        for i, (start, end, title) in enumerate(positions):
            body_start = end
            body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                sections.append((title, body))
        return sections

    def _window(self, body: str) -> Iterable[str]:
        toks = _ENC.encode(body)
        if len(toks) <= self.max:
            yield body
            return
        step = self.max - self.overlap
        i = 0
        while i < len(toks):
            piece = _ENC.decode(toks[i : i + self.max]).strip()
            if _tok_count(piece) >= self.min:
                yield piece
            i += step

    _version_tag: str = "v2"  # overridden in subclass for deterministic IDs

    def chunk(self, meta: PaperMetadata, text: str) -> list[Chunk]:
        out: list[Chunk] = []
        idx = 0
        for title, body in self._split_by_headers(text):
            section_type = _classify_section(title)
            # Skip references entirely — noise for QA over methods/results.
            if section_type == "references":
                continue
            for piece in self._window(body):
                out.append(
                    Chunk(
                        id_=_det_uuid(self._version_tag, meta.paper_id, str(idx)),
                        paper_id=meta.paper_id,
                        title=meta.title,
                        authors=meta.authors,
                        year=meta.year,
                        primary_category=meta.primary_category,
                        section_type=section_type,
                        section_title=title[:160],
                        chunk_index=idx,
                        text=piece,
                        pdf_url=meta.pdf_url,
                        source_url=_arxiv_abs_url(meta.paper_id),
                    )
                )
                idx += 1
        return out



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
_TABLE_LINE_RX = re.compile(r"^\s*\|")


class TableAwareSectionChunker(SectionAwareChunker):
    """v3: section-aware chunking that never splits inside a markdown table.

    Tables (consecutive lines starting with ``|``) are extracted as atomic
    blocks before the sliding window is applied. If a single table exceeds
    max_tokens it is kept whole rather than cut mid-row.
    """

    _version_tag: str = "v3"

    def _split_tables(self, body: str) -> list[tuple[str, bool]]:
        """Return [(text_segment, is_table), …] preserving order."""
        segments: list[tuple[str, bool]] = []
        buf_normal: list[str] = []
        buf_table: list[str] = []
        in_table = False

        for line in body.splitlines(keepends=True):
            is_table_line = bool(_TABLE_LINE_RX.match(line))
            if is_table_line:
                if not in_table:
                    if buf_normal:
                        segments.append(("".join(buf_normal), False))
                        buf_normal = []
                    in_table = True
                buf_table.append(line)
            else:
                if in_table:
                    segments.append(("".join(buf_table), True))
                    buf_table = []
                    in_table = False
                buf_normal.append(line)

        if buf_table:
            segments.append(("".join(buf_table), True))
        if buf_normal:
            segments.append(("".join(buf_normal), False))
        return segments

    def _window(self, body: str) -> Iterable[str]:  # type: ignore[override]
        for segment, is_table in self._split_tables(body):
            if not segment.strip():
                continue
            if is_table:
                # Keep the table atomic; split at row boundaries only if huge.
                if _tok_count(segment) <= self.max:
                    yield segment.strip()
                else:
                    # Split at row boundaries (each row = one | line).
                    rows = [l for l in segment.splitlines(keepends=True) if l.strip()]
                    header = rows[:2]  # keep header + separator row together
                    batch: list[str] = list(header)
                    for row in rows[2:]:
                        candidate = "".join(batch + [row])
                        if _tok_count(candidate) > self.max and len(batch) > 2:
                            yield "".join(batch).strip()
                            batch = list(header) + [row]
                        else:
                            batch.append(row)
                    if batch:
                        yield "".join(batch).strip()
            else:
                yield from super()._window(segment)


def get_chunker(version: str):
    if version == "v1":
        return FixedSizeChunker()
    if version == "v3":
        return TableAwareSectionChunker()
    return SectionAwareChunker()  # v2 and default
