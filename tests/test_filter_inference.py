"""Filter inference: query string → Qdrant Filter.

The retriever auto-detects year ranges and category hints from the user query
when no explicit filter is supplied. Regression tests live here.
"""
from __future__ import annotations

from paperpilot.retrieval.retriever import _infer_filters


def _conditions(flt) -> list[dict]:
    """Convert a Qdrant Filter into a list of dicts for easy assertions."""
    if flt is None:
        return []
    out = []
    for c in (flt.must or []):
        d = {"key": c.key}
        if getattr(c, "match", None) is not None:
            d["match"] = c.match.value
        if getattr(c, "range", None) is not None:
            d["range"] = {"gte": c.range.gte, "lte": c.range.lte}
        out.append(d)
    return out


def test_no_hints_returns_none():
    assert _infer_filters("What is HyDE?") is None


def test_single_year():
    flt = _infer_filters("What did self-RAG report in 2023?")
    conds = _conditions(flt)
    assert any(c.get("match") == 2023 for c in conds)


def test_year_range():
    flt = _infer_filters("Compare RAG papers from 2022 to 2024")
    conds = _conditions(flt)
    rng = next(c for c in conds if "range" in c)
    assert rng["range"]["gte"] == 2022
    assert rng["range"]["lte"] == 2024


def test_category_hint_cs_cl():
    flt = _infer_filters("In NLP papers, what is HyDE?")
    conds = _conditions(flt)
    assert any(c["key"] == "primary_category" and c.get("match") == "cs.CL" for c in conds)


def test_implausible_years_ignored():
    # Year 1999 is below cutoff (>=2018) → ignored, not crashing
    assert _infer_filters("retrieval methods from 1999") is None


def test_category_and_year_combined():
    flt = _infer_filters("Show me cs.CL papers from 2024 about agents")
    conds = _conditions(flt)
    assert any(c["key"] == "primary_category" for c in conds)
    assert any(c["key"] == "year" for c in conds)
