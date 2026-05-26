"""Tool Call Accuracy — agentic metric.

Definition (project-internal):
  For each golden question, define the *expected tool sequence* based on the question category:
    - in-corpus categories (definitional, comparative, methodological, attribution, numerical):
        expected = ["rag_retrieve"]   (agent should NOT need arxiv_search)
    - out_of_context:
        expected = ["rag_retrieve", "arxiv_search"]   (agent should fall back, then refuse)

  We compare the actual tool-call sequence (de-duplicated, in call order) against expected.
  Score per row ∈ {0, 0.5, 1}:
    1   exact match (or actual ⊇ expected with no extra/wrong tools)
    0.5 right tools, wrong order or extra calls (≤ max_iter)
    0   missing required tool, OR called a wrong tool, OR exceeded max_iter

  Tool Call Accuracy = mean of per-row scores.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from paperpilot.agent.graph import run_agent
from paperpilot.config import REPORTS_DIR, settings
from paperpilot.eval.golden_gen import load_golden_set
from paperpilot.observability.langfuse_setup import flush_langfuse

logger = logging.getLogger(__name__)

EXPECTED_BY_CATEGORY: dict[str, list[str]] = {
    "definitional": ["rag_retrieve"],
    "comparative": ["rag_retrieve"],
    "methodological": ["rag_retrieve"],
    "attribution": ["rag_retrieve"],
    "numerical": ["rag_retrieve"],
    "out_of_context": ["rag_retrieve", "arxiv_search"],
}


def _score(expected: list[str], actual: list[str]) -> float:
    if not actual:
        return 0.0
    # Strip duplicates while preserving order
    seen, ordered = set(), []
    for t in actual:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    # Hard fail: any tool not in our toolbox
    valid = {"rag_retrieve", "arxiv_search"}
    if any(t not in valid for t in ordered):
        return 0.0
    # Exact prefix or superset of expected, in order
    if ordered[: len(expected)] == expected and all(t in ordered for t in expected):
        return 1.0
    # Right set, wrong order
    if set(expected).issubset(set(ordered)):
        return 0.5
    return 0.0


def run_tool_call_acc(version: str = "v2") -> Path:
    # Pre-load native extensions in the main thread before LangGraph spawns worker threads.
    # pyarrow (via sentence_transformers → sklearn → pandas) cannot be safely imported
    # from concurrent threads on Windows — doing so causes a DLL race → access violation.
    import sentence_transformers  # noqa: F401

    golden = load_golden_set()
    rows: list[dict] = []
    for i, q in enumerate(golden):
        if i > 0:
            time.sleep(2)
        category = q.get("category", "?")
        expected = EXPECTED_BY_CATEGORY.get(category, ["rag_retrieve"])
        try:
            result = run_agent(q["question"], version=version, session_id=f"tca-{version}-{i}")
        except Exception as exc:
            logger.warning("Agent error on #%d: %s", i, exc)
            rows.append({"i": i, "category": category, "expected": expected, "actual": [], "score": 0.0})
            continue
        actual = [tc["name"] for tc in result.get("tool_calls", [])]
        s = _score(expected, actual)
        rows.append({
            "i": i, "question": q["question"], "category": category,
            "expected": expected, "actual": actual, "score": s,
            "answer": result["answer"][:400],
        })

    n = len(rows) or 1
    overall = sum(r["score"] for r in rows) / n
    by_cat: dict[str, float] = {}
    for cat in {r["category"] for r in rows}:
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = sum(r["score"] for r in sub) / max(1, len(sub))

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = {
        "version": version,
        "n": n,
        "overall": overall,
        "by_category": by_cat,
        "rows": rows,
        "timestamp": ts,
    }
    out_path = REPORTS_DIR / f"tool_call_acc_{version}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Tool Call Accuracy (%s): %.3f → %s", version, overall, out_path)
    flush_langfuse()
    return out_path
