"""LangChain-style tools used by the ReAct agent and the multi-agent Researcher node.

Both tools have:
  - clear, structured docstring (used as tool description by the model)
  - Pydantic input schema (becomes the JSON schema seen by the model)
  - JSON-serializable return value

Public impl functions (rag_retrieve_impl, arxiv_search_impl) are called directly
by the Researcher node for concurrent asyncio.gather execution.
"""
from __future__ import annotations

import logging
from typing import Optional

import arxiv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from qdrant_client.http import models as qm

from paperpilot.retrieval import get_retriever

logger = logging.getLogger(__name__)



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
class RagRetrieveArgs(BaseModel):
    query: str = Field(..., description="Natural-language question or keywords to search.")
    year_from: Optional[int] = Field(None, description="Earliest publication year (inclusive).")
    year_to: Optional[int] = Field(None, description="Latest publication year (inclusive).")
    primary_category: Optional[str] = Field(
        None, description="Restrict to an arXiv primary category, e.g. 'cs.CL' or 'cs.AI'."
    )
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to return.")


def rag_retrieve_impl(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    primary_category: Optional[str] = None,
    top_k: int = 4,
    *,
    version: str = "v2",
) -> list[dict]:
    """Core retrieval logic shared by the ReAct tool wrapper and the Researcher node."""
    must: list[qm.FieldCondition] = []
    if year_from is not None and year_to is not None:
        must.append(qm.FieldCondition(key="year", range=qm.Range(gte=year_from, lte=year_to)))
    elif year_from is not None:
        must.append(qm.FieldCondition(key="year", range=qm.Range(gte=year_from)))
    elif year_to is not None:
        must.append(qm.FieldCondition(key="year", range=qm.Range(lte=year_to)))
    if primary_category:
        must.append(qm.FieldCondition(key="primary_category", match=qm.MatchValue(value=primary_category)))
    flt = qm.Filter(must=must) if must else None

    retriever = get_retriever(version)  # type: ignore[arg-type]
    chunks = retriever.search(query=query, top_k=top_k, filters=flt, auto_filter=flt is None)
    return [
        {
            "paper_id": c.paper_id,
            "title": c.title,
            "authors": c.authors,
            "year": c.year,
            "primary_category": c.primary_category,
            "section_type": c.section_type,
            "section_title": c.section_title,
            "text": c.text,
            "source_url": c.source_url,
            "score": c.score,
            "rerank_score": c.rerank_score,
        }
        for c in chunks
    ]


def make_rag_retrieve_tool(version: str = "v2") -> StructuredTool:
    """Bind a `version` (v1/v2) at build time so each variant uses the right collection."""

    def _f(
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        primary_category: Optional[str] = None,
        top_k: int = 4,
    ) -> list[dict]:
        return rag_retrieve_impl(
            query=query,
            year_from=year_from,
            year_to=year_to,
            primary_category=primary_category,
            top_k=top_k,
            version=version,
        )

    return StructuredTool.from_function(
        func=_f,
        name="rag_retrieve",
        description=(
            "Semantic search over PaperPilot's indexed corpus of NLP/LLM/RAG/Agents papers "
            "(~2023–2026). Returns top-k chunks with paper title, authors, year, section, "
            "and a relevance score. Use this FIRST for any factual question. "
            "Optional filters: year_from/year_to, primary_category (e.g. 'cs.CL')."
        ),
        args_schema=RagRetrieveArgs,
    )



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
class ArxivSearchArgs(BaseModel):
    query: str = Field(..., description="ArXiv search expression (free text or fielded).")
    max_results: int = Field(5, ge=1, le=10, description="Max papers to return.")


def arxiv_search_impl(query: str, max_results: int = 5) -> list[dict]:
    """Core ArXiv search logic shared by the ReAct tool wrapper and the Researcher node.

    Results are cached in SQLite so repeated eval runs don't trigger HTTP 429.
    """
    from paperpilot.cache import get_cache
    cache = get_cache()
    cached = cache.get_arxiv(query, max_results)
    if cached is not None:
        logger.debug("arxiv_search cache hit: %s", query[:60])
        return cached

    client = arxiv.Client(page_size=20, delay_seconds=10, num_retries=5)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    out: list[dict] = []
    for r in client.results(search):
        out.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "primary_category": r.primary_category,
            "summary": r.summary,
            "entry_id": r.entry_id,
            "pdf_url": r.pdf_url,
        })
    cache.set_arxiv(query, max_results, out)
    return out


arxiv_search_tool = StructuredTool.from_function(
    func=arxiv_search_impl,
    name="arxiv_search",
    description=(
        "Live search the ArXiv API for papers NOT in the indexed corpus (e.g. very recent or "
        "outside the curated themes). Returns titles, authors, year, primary category, abstract, "
        "and PDF URL. Use this only after `rag_retrieve` returns nothing useful."
    ),
    args_schema=ArxivSearchArgs,
)


def get_tools(version: str = "v2") -> list:
    """Return the tool list for the legacy ReAct agent (backward-compatible)."""
    return [make_rag_retrieve_tool(version=version), arxiv_search_tool]
