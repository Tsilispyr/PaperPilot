"""Tool Call Accuracy scoring rubric - synthetic traces, no network."""
from __future__ import annotations

from paperpilot.eval.tool_call_acc import _score


def test_exact_match_in_corpus():
    assert _score(["rag_retrieve"], ["rag_retrieve"]) == 1.0


def test_extra_call_after_required_one():
    # rag_retrieve was called first (good) then arxiv_search (extra). Still acceptable
    # as a "right tools, wrong order" → 0.5? No: prefix matches expected, both tools present
    # → counts as superset → 1.0 by the rubric in tool_call_acc.py
    assert _score(["rag_retrieve"], ["rag_retrieve", "arxiv_search"]) == 1.0


def test_missing_required_tool():
    assert _score(["rag_retrieve"], []) == 0.0


def test_wrong_tool_only():
    # arxiv_search alone - never called rag_retrieve → 0.0
    assert _score(["rag_retrieve"], ["arxiv_search"]) == 0.0


def test_ooc_expected_sequence():
    expected = ["rag_retrieve", "arxiv_search"]
    assert _score(expected, ["rag_retrieve", "arxiv_search"]) == 1.0


def test_ooc_wrong_order():
    expected = ["rag_retrieve", "arxiv_search"]
    # Both tools called but arxiv first - same set, wrong order → 0.5
    assert _score(expected, ["arxiv_search", "rag_retrieve"]) == 0.5


def test_invalid_tool_name_zero():
    assert _score(["rag_retrieve"], ["rag_retrieve", "calculator"]) == 0.0


def test_dedup_preserves_order():
    # Repeated calls don't help, but don't hurt either
    assert _score(["rag_retrieve"], ["rag_retrieve", "rag_retrieve"]) == 1.0
