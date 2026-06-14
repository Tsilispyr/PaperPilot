"""Parse downloaded PDFs to markdown using pymupdf4llm.

Why pymupdf4llm: extracts a clean markdown rendering with H1/H2/H3 preserved,
which is what the section-aware chunker keys off.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf4llm

from paperpilot.config import PROCESSED_DIR, RAW_DIR
from paperpilot.ingest.schema import PaperMetadata

logger = logging.getLogger(__name__)


def _md_path(paper_id: str) -> Path:
    return PROCESSED_DIR / f"{paper_id}.md"


def _meta_path(paper_id: str) -> Path:
    return PROCESSED_DIR / f"{paper_id}.meta.json"


def parse_one(paper_id: str) -> tuple[Path, Path]:
    pdf = RAW_DIR / f"{paper_id}.pdf"
    raw_meta = json.loads((RAW_DIR / f"{paper_id}.meta.json").read_text(encoding="utf-8"))
    meta = PaperMetadata.model_validate(raw_meta)

    md_p = _md_path(paper_id)
    meta_p = _meta_path(paper_id)

    if md_p.exists() and meta_p.exists():
        logger.debug("Already parsed %s", paper_id)
        return md_p, meta_p

    md = pymupdf4llm.to_markdown(str(pdf))
    md_p.write_text(md, encoding="utf-8")
    meta_p.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return md_p, meta_p


def parse_corpus() -> int:
    paper_ids = sorted(p.stem for p in RAW_DIR.glob("*.pdf"))
    if not paper_ids:
        logger.warning("data/raw/ is empty - run `make fetch` first.")
        return 0

    n_ok = 0
    for pid in paper_ids:
        try:
            parse_one(pid)
            n_ok += 1
        except Exception as exc:
            logger.warning("Parse failed for %s: %s", pid, exc)
    logger.info("Parsed %d/%d papers into %s", n_ok, len(paper_ids), PROCESSED_DIR)
    return n_ok


def load_processed() -> list[tuple[PaperMetadata, str]]:
    """Return [(metadata, markdown_text), ...] for all parsed papers."""
    out: list[tuple[PaperMetadata, str]] = []
    for meta_file in sorted(PROCESSED_DIR.glob("*.meta.json")):
        pid = meta_file.stem.replace(".meta", "")
        md_file = _md_path(pid)
        if not md_file.exists():
            continue
        meta = PaperMetadata.model_validate_json(meta_file.read_text(encoding="utf-8"))
        text = md_file.read_text(encoding="utf-8")
        out.append((meta, text))
    return out
