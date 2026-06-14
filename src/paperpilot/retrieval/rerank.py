"""Cross-encoder reranker for v2.

Loaded lazily - first call downloads the HF model (~280 MB for bge-reranker-base).

Phase 3 addition: strict score threshold. Chunks whose cross-encoder score falls
below settings.reranker_score_threshold are dropped entirely, even if that means
returning fewer than top_k chunks. This keeps noise out of the agent context window.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from paperpilot.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def _model():
    import torch
    from sentence_transformers import CrossEncoder

    # Limit threads to avoid Windows native-crash under multi-threaded OTel.
    torch.set_num_threads(1)
    logger.info("Loading reranker %s …", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, max_length=512, device="cpu")


def rerank(
    query: str,
    chunks: list,
    top_k: int,
    score_threshold: float | None = None,
    diversity: bool = True,
) -> list:
    """Score (query, chunk.text) pairs, drop below threshold, return up to top_k by score.

    score_threshold defaults to settings.reranker_score_threshold (-2.0).
    diversity=True (default) enforces at most one chunk per paper_id, ensuring
    results span multiple papers rather than being dominated by one long paper.
    Fewer than top_k chunks may be returned when all remaining scores are above
    top_k but some are below the threshold - that is intentional.
    """
    if not chunks:
        return []

    threshold = score_threshold if score_threshold is not None else settings.reranker_score_threshold

    pairs = [(query, c.text) for c in chunks]
    try:
        raw = _model().predict(pairs, show_progress_bar=False, batch_size=8)
        scores: list[float] = raw.tolist() if hasattr(raw, "tolist") else [float(s) for s in raw]
    except Exception as exc:
        # Model download failed or not available yet - return dense-order results so
        # v2/v3 degrade gracefully instead of showing "No relevant papers found".
        logger.warning("Reranker unavailable (%s) - returning top-%d dense results", exc, top_k)
        return chunks[:top_k]

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    out: list = []
    dropped = 0
    seen_papers: set[str] = set()
    for c, s in ranked:
        if len(out) >= top_k:
            break
        if s < threshold:
            logger.debug(
                "Dropped chunk score=%.3f < threshold=%.3f: %s…",
                s, threshold, getattr(c, "text", "")[:60],
            )
            dropped += 1
            continue
        if diversity:
            paper_id = getattr(c, "paper_id", None)
            if paper_id and paper_id in seen_papers:
                logger.debug("Diversity: skipping duplicate paper_id=%s score=%.3f", paper_id, s)
                continue
            if paper_id:
                seen_papers.add(paper_id)
        try:
            c.rerank_score = float(s)
        except Exception:
            pass
        out.append(c)

    if dropped:
        logger.info("Reranker dropped %d/%d chunks below threshold %.2f", dropped, len(chunks), threshold)
    if diversity and seen_papers:
        logger.debug("Diversity filter: %d unique papers in top-%d results", len(seen_papers), len(out))

    return out
