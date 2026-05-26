"""Download ArXiv PDFs into data/raw/.

Uses the official `arxiv` package. Each query in `settings.arxiv_query_list` contributes
its share of the total budget, so we cover all themes evenly.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from paperpilot.config import RAW_DIR, settings

logger = logging.getLogger(__name__)

# Seconds to wait between PDF downloads.
# ArXiv rate-limits aggressive downloaders; 5s keeps us well under the limit.
_DOWNLOAD_DELAY = 5

# Minimum valid PDF size — anything smaller is a truncated/error response.
_MIN_PDF_BYTES = 10_000


@retry(
    wait=wait_exponential(multiplier=15, min=30, max=300),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_results(client: arxiv.Client, search: arxiv.Search) -> list[arxiv.Result]:
    """Fetch all results for one search query, with exponential backoff on 429/503."""
    return list(client.results(search))


@retry(
    wait=wait_exponential(multiplier=2, min=15, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _download_pdf(result: arxiv.Result, pdf_path: Path) -> None:
    """Download one PDF with retry on 429 / incomplete transfer.

    Deletes any partial file before each attempt so a failed download
    doesn't leave a truncated PDF that would be mistaken for a valid one.
    """
    if pdf_path.exists():
        pdf_path.unlink()
    result.download_pdf(dirpath=str(pdf_path.parent), filename=pdf_path.name)
    size = pdf_path.stat().st_size if pdf_path.exists() else 0
    if size < _MIN_PDF_BYTES:
        if pdf_path.exists():
            pdf_path.unlink()
        raise RuntimeError(f"PDF too small ({size} bytes) — likely truncated or error page")


def _safe_paper_id(entry_id: str) -> str:
    # entry_id looks like 'http://arxiv.org/abs/2401.01234v2'
    return entry_id.rsplit("/", 1)[-1].replace("/", "_")


def _within_dates(published: datetime, frm: str, to: str) -> bool:
    f = datetime.fromisoformat(frm)
    t = datetime.fromisoformat(to)
    p = published.replace(tzinfo=None)
    return f <= p <= t


def _write_meta(result: arxiv.Result, paper_id: str, query: str = "seed") -> Path:
    meta_path = RAW_DIR / f"{paper_id}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "primary_category": result.primary_category,
                "categories": list(result.categories),
                "entry_id": result.entry_id,
                "pdf_url": result.pdf_url,
                "query": query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta_path


def fetch_corpus(max_papers: int | None = None) -> int:
    """Fetch up to `max_papers` total.

    Phase 1: fetch explicit seed IDs (foundational papers whose titles don't match
             query terms, e.g. HyDE = "Precise Zero-Shot Dense Retrieval...").
    Phase 2: distribute remaining budget across the configured queries.

    Skips papers already on disk (idempotent).
    """
    target_total = max_papers or settings.arxiv_max_papers
    queries = settings.arxiv_query_list
    if not queries:
        raise RuntimeError("No queries configured (settings.arxiv_query_list is empty).")

    client = arxiv.Client(page_size=50, delay_seconds=10, num_retries=5)
    fetched = 0
    seen: set[str] = set()

    # --- Phase1:seedIDs ---
    seed_ids = settings.arxiv_seed_id_list
    if seed_ids:
        logger.info("Fetching %d seed papers by ID …", len(seed_ids))
        try:
            seed_search = arxiv.Search(id_list=seed_ids)
            seed_results = _fetch_results(client, seed_search)
        except Exception as exc:
            logger.warning("Seed fetch failed: %s — continuing without seeds.", exc)
            seed_results = []
        for result in seed_results:
            paper_id = _safe_paper_id(result.entry_id)
            seen.add(paper_id)
            pdf_path = RAW_DIR / f"{paper_id}.pdf"
            meta_path = RAW_DIR / f"{paper_id}.meta.json"
            if pdf_path.exists() and meta_path.exists() and pdf_path.stat().st_size >= _MIN_PDF_BYTES:
                fetched += 1
                logger.debug("Seed already on disk: %s", paper_id)
                continue
            try:
                logger.info("[seed] %s — %s", paper_id, result.title[:80])
                _download_pdf(result, pdf_path)
                _write_meta(result, paper_id, query="seed")
                fetched += 1
                time.sleep(_DOWNLOAD_DELAY)
            except Exception as exc:
                logger.warning("Seed fetch failed for %s: %s", paper_id, exc)
                if pdf_path.exists():
                    pdf_path.unlink()

    # --- Phase2:query-basedfetch ---
    remaining = target_total - fetched
    per_query = max(1, remaining // len(queries))
    logger.info("Query fetch: ~%d remaining (%d per query, %d queries)", remaining, per_query, len(queries))

    for q in queries:
        search = arxiv.Search(
            query=q,
            max_results=per_query * 4,  # over-fetch to absorb date filtering
            sort_by=arxiv.SortCriterion.Relevance,
        )
        kept_for_q = 0
        try:
            results = _fetch_results(client, search)
        except Exception as exc:
            logger.warning("Query '%s' failed after retries: %s — skipping.", q[:60], exc)
            continue

        for result in results:
            if kept_for_q >= per_query or fetched >= target_total:
                break
            paper_id = _safe_paper_id(result.entry_id)
            if paper_id in seen:
                continue
            if not _within_dates(result.published, settings.arxiv_from_date, settings.arxiv_to_date):
                continue
            seen.add(paper_id)
            pdf_path = RAW_DIR / f"{paper_id}.pdf"
            meta_path = RAW_DIR / f"{paper_id}.meta.json"
            if pdf_path.exists() and meta_path.exists() and pdf_path.stat().st_size >= _MIN_PDF_BYTES:
                fetched += 1
                kept_for_q += 1
                logger.debug("Already have %s", paper_id)
                continue
            try:
                logger.info("[%d/%d] %s — %s", fetched + 1, target_total, paper_id, result.title[:80])
                _download_pdf(result, pdf_path)
                _write_meta(result, paper_id, query=q)
                fetched += 1
                kept_for_q += 1
                time.sleep(_DOWNLOAD_DELAY)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", paper_id, exc)
                if pdf_path.exists():
                    pdf_path.unlink()
                continue

        if fetched >= target_total:
            break

    logger.info("Done. %d papers on disk in %s", fetched, RAW_DIR)
    return fetched


def list_local_papers() -> list[Path]:
    return sorted(RAW_DIR.glob("*.pdf"))
