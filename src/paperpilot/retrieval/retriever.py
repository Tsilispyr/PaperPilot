"""Retriever — v1 (pure dense) and v2 (dense + heuristic filter + cross-encoder rerank)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from paperpilot.config import settings
from paperpilot.ingest.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """View of a Qdrant hit, exposed to the agent and the UI."""
    paper_id: str
    title: str
    authors: list[str]
    year: int
    primary_category: str
    section_type: str
    section_title: Optional[str]
    text: str
    pdf_url: str
    source_url: str
    score: float
    rerank_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def short_citation(self) -> str:
        a = self.authors[0].split()[-1] if self.authors else "?"
        more = " et al." if len(self.authors) > 1 else ""
        return f"{a}{more} ({self.year}) — {self.title[:80]}"



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
_YEAR_RX = re.compile(r"\b(20\d{2})\b")
_CAT_HINTS: dict[str, str] = {
    "cs.cl": "cs.CL", "cs.ai": "cs.AI", "cs.lg": "cs.LG", "cs.ir": "cs.IR",
    "nlp": "cs.CL", "natural language": "cs.CL",
}


def _infer_filters(query: str) -> qm.Filter | None:
    """Cheap rule-based detection of year/category hints in the user query."""
    must: list[qm.FieldCondition] = []
    q = query.lower()
    yrs = _YEAR_RX.findall(query)
    if yrs:
        ys = sorted({int(y) for y in yrs if 2018 <= int(y) <= 2027})
        if len(ys) == 1:
            must.append(qm.FieldCondition(key="year", match=qm.MatchValue(value=ys[0])))
        elif len(ys) >= 2:
            must.append(qm.FieldCondition(key="year", range=qm.Range(gte=ys[0], lte=ys[-1])))
    for hint, cat in _CAT_HINTS.items():
        if hint in q:
            must.append(qm.FieldCondition(key="primary_category", match=qm.MatchValue(value=cat)))
            break  # one is enough
    return qm.Filter(must=must) if must else None



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
class Retriever:
    def __init__(self, version: str):
        self.version = version
        self.embedder = EmbeddingProvider()
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=300)
        self.collection = settings.collection_for(version)

    def _query_vec(self, query: str) -> list[float]:
        return self.embedder.embed([query])[0]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: Optional[qm.Filter] = None,
        auto_filter: bool = True,
    ) -> list[RetrievedChunk]:
        """v1: dense top-k. v2/v3: dense over-fetch → cross-encoder rerank → top-k.

        v3 additionally enforces per-paper diversity in the reranker.
        """
        vec = self._query_vec(query)

        if self.version == "v1":
            k = top_k or settings.top_k_v1
            result = self.client.query_points(
                collection_name=self.collection, query=vec, limit=k, query_filter=filters
            )
            return [self._to_chunk(h) for h in result.points]

        # v2 / v3 — dense over-fetch, then rerank
        k_dense = settings.top_k_v2_dense
        k_final = top_k or settings.top_k_v2_rerank
        if filters is None and auto_filter:
            filters = _infer_filters(query)
        result = self.client.query_points(
            collection_name=self.collection, query=vec, limit=k_dense, query_filter=filters
        )
        candidates = [self._to_chunk(h) for h in result.points]
        if not candidates:
            return []
        from paperpilot.retrieval.rerank import rerank
        # v3 enables diversity reranking (max 1 chunk/paper); v2 keeps original behaviour.
        return rerank(query, candidates, top_k=k_final, diversity=(self.version == "v3"))

    @staticmethod
    def _to_chunk(hit) -> RetrievedChunk:
        p = hit.payload or {}
        return RetrievedChunk(
            paper_id=p.get("paper_id", ""),
            title=p.get("title", ""),
            authors=p.get("authors", []) or [],
            year=int(p.get("year", 0)),
            primary_category=p.get("primary_category", ""),
            section_type=p.get("section_type", "other"),
            section_title=p.get("section_title"),
            text=p.get("text", ""),
            pdf_url=p.get("pdf_url", ""),
            source_url=p.get("source_url", ""),
            score=float(hit.score) if hit.score is not None else 0.0,
            metadata={"chunk_index": p.get("chunk_index")},
        )



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
_retrievers: dict[str, Retriever] = {}


def get_retriever(version: str) -> Retriever:
    if version not in _retrievers:
        _retrievers[version] = Retriever(version)
    return _retrievers[version]
