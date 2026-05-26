"""Run RAGAS over the curated golden set against a given pipeline version.

Metrics:
  - context_precision
  - context_recall
  - faithfulness
  - answer_relevancy

Each row uses the agent itself to produce the answer + retrieved contexts (so we measure
the *system as a whole*, not just the bare retriever).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from paperpilot.agent.graph import run_agent
from paperpilot.agent.tools import rag_retrieve_impl as _rag_retrieve_impl
from paperpilot.config import REPORTS_DIR, settings
from paperpilot.eval.golden_gen import load_golden_set
from paperpilot.observability.langfuse_setup import flush_langfuse

logger = logging.getLogger(__name__)


def _gather_contexts_from_messages(messages) -> list[str]:
    """Pull the retrieved-chunk texts out of tool messages for RAGAS context_* metrics."""
    contexts: list[str] = []
    for m in messages:
        if m.__class__.__name__ != "ToolMessage":
            continue
        try:
            payload = json.loads(m.content) if isinstance(m.content, str) else m.content
        except Exception:
            continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "text" in item:
                    contexts.append(item["text"])
    return contexts


def run_ragas(version: str) -> Path:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    golden = load_golden_set()
    if not golden:
        raise RuntimeError("Golden set is empty.")

    rows: list[dict] = []
    for i, q in enumerate(golden):
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "?")
        # OOC negatives don't have a meaningful reference answer; use a sentinel.
        if category == "out_of_context":
            expected = "This is not in my corpus."

        try:
            result = run_agent(question, version=version, session_id=f"ragas-{version}-{i}")
        except Exception as exc:
            logger.warning("Agent failed on #%d: %s", i, exc)
            continue

        # Fallback context: if the agent didn't call rag_retrieve (rare), retrieve directly.
        contexts = _gather_contexts_from_messages(result["messages"])
        if not contexts:
            contexts = [r["text"] for r in _rag_retrieve_impl(question, top_k=4, version=version)]

        rows.append({
            "question": question,
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": expected,
            "reference": expected,
            "category": category,
        })

    if not rows:
        raise RuntimeError("No rows to evaluate (every agent call failed).")

    ds = Dataset.from_list(rows)

    if settings.llm_provider == "ollama":
        judge = ChatOpenAI(
            model=settings.ollama_llm_model,
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
            temperature=0,
        )
        from langchain_community.embeddings import OllamaEmbeddings
        embed = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    elif settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        judge = ChatGoogleGenerativeAI(
            model=settings.google_llm_model,
            api_key=settings.google_api_key,
            temperature=0,
        )
        embed = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key,
        )
    else:
        judge = ChatOpenAI(model=settings.openai_judge_model, temperature=0, api_key=settings.openai_api_key)
        embed = OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)
    judge_w = LangchainLLMWrapper(judge)
    embed_w = LangchainEmbeddingsWrapper(embed)

    result = evaluate(
        ds,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge_w,
        embeddings=embed_w,
    )

    df = result.to_pandas()
    df["category"] = [r["category"] for r in rows]
    out_dir = REPORTS_DIR
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"ragas_{version}_{ts}.csv"
    json_path = out_dir / f"ragas_{version}.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "version": version,
        "n": int(len(df)),
        "scores": {k: float(df[k].mean()) for k in df.columns if k.endswith(("_precision", "_recall", "faithfulness", "answer_relevancy"))},
        "by_category": {},
        "csv": str(csv_path),
        "timestamp": ts,
    }
    if "category" in df.columns:
        summary["by_category"] = {
            cat: {
                k: float(sub[k].mean())
                for k in sub.columns
                if k.endswith(("_precision", "_recall", "faithfulness", "answer_relevancy"))
            }
            for cat, sub in df.groupby("category")
        }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("RAGAS %s done → %s", version, json_path)
    flush_langfuse()
    return json_path
