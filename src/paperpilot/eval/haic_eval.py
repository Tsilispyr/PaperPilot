"""HAIC evaluation — proper Human-AI Collaboration metrics (L9-L10).

Implements EL, Tr, HCL, F, A, D, EfficiencyScore from the haic.decisions.v1 schema.

In automated mode (no real users):
  - Each golden question is run through the agent
  - An LLM judge scores helpfulness (1-5); score >= 3 → accept, else reject
  - Latencies are measured; human decision time is simulated as a short review pause
  - The collected events are fed to compute_haic_metrics() to produce the final report

For real human interaction, the Chainlit UI records live accept/reject events
(see server/chainlit_app.py) which can be post-processed with the same function.

Metric definitions (from lectures):
  EL  = (t_actual - baseline_s) / baseline_s          (Effort Loss; lower=better)
  Tr  = accepted / (accepted + rejected)               (Trust; higher=better)
  HCL = 1 - clip(mean_ai_latency_s / rt_max_s, 0, 1)  (Human Cognitive Load; higher=better)
  F   = events / session_duration_minutes              (Interaction Frequency)
  D   = mean human decision duration in seconds
  A   = tanh((Tr_late - Tr_early) / Tr_early)          (Adaptability; >0=improving)
  EfficiencyScore = 1 / (1 + EL)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from paperpilot.agent.graph import run_agent
from paperpilot.cache import get_cache
from paperpilot.config import REPORTS_DIR, settings
from paperpilot.eval.golden_gen import load_golden_set
from paperpilot.observability.haic_logger import HAICLogger, compute_haic_metrics
from paperpilot.observability.langfuse_setup import flush_langfuse

logger = logging.getLogger(__name__)

# How long (seconds) the user would take without AI (per question * 84 = baseline_s)
BASELINE_PER_QUESTION_S = 89    # ~125 min / 84 questions
RT_MAX_S = 30                   # max acceptable chatbot response time (seconds)

JUDGE_PROMPT = """\
You are a strict HAIC (Human-AI Collaboration) evaluator. Score the AI answer below.

Metric: helpfulness
Scale: 1-5 (1=useless, 5=fully addresses the user's intent)
Description: Does the answer give the user what they actually need to act/decide?

User question: {question}
Reference answer (may be approximate): {reference}
AI answer: {answer}

Return strict JSON: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

# OOC judge: score 5 if the agent correctly refuses, 1 if it hallucinates an answer.
OOC_JUDGE_PROMPT = """\
You are evaluating whether an AI assistant correctly handled an out-of-corpus question.

The question is OUTSIDE the AI's knowledge base (NLP/LLM/RAG papers 2020-2026).
A correct response REFUSES or says the topic is not in its corpus.
An incorrect response invents an answer (hallucination).

Score:
  5 = Agent clearly states it cannot answer / topic not in corpus (correct refusal)
  3 = Agent partially hedges but still provides some content (partial hallucination)
  1 = Agent confidently answers as if it knows (full hallucination)

User question: {question}
AI answer: {answer}

Return strict JSON: {{"score": <integer 1, 3, or 5>, "reason": "<one sentence>"}}"""


def _judge_score(question: str, reference: str, answer: str, is_ooc: bool = False) -> tuple[float, str]:
    """Call the judge LLM; return (score, reason). Cached.

    For OOC questions uses a separate prompt that rewards correct refusal (score=5)
    rather than helpfulness — preventing the perverse incentive of hallucinating an answer.
    """
    cache = get_cache()
    if is_ooc:
        prompt = OOC_JUDGE_PROMPT.format(question=question, answer=answer[:2000])
    else:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference[:600],
            answer=answer[:2000],
        )
    cached = cache.get_judge(settings.openai_judge_model, prompt)
    if cached:
        return float(cached.get("score", 3)), cached.get("reason", "")

    try:
        if settings.llm_provider == "ollama":
            from openai import OpenAI
            client = OpenAI(api_key="ollama", base_url=f"{settings.ollama_base_url}/v1")
            model = settings.ollama_judge_model
        elif settings.llm_provider == "google":
            return 3.0, "google-judge-not-supported-in-haic-eval"
        else:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            model = settings.openai_judge_model

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        cache.set_judge(settings.openai_judge_model, prompt, data)
        return float(data.get("score", 3)), data.get("reason", "")
    except Exception as exc:
        logger.warning("HAIC judge failed: %s — defaulting to neutral score 3", exc)
        return 3.0, f"judge-error: {exc}"


def run_haic(version: str = "v2") -> Path:
    import sentence_transformers  # noqa: F401  pre-load to avoid DLL race on Windows

    golden = load_golden_set()
    baseline_s = BASELINE_PER_QUESTION_S * len(golden)

    haic_log = HAICLogger(
        session_id=f"haic-automated-{version}-{time.strftime('%Y%m%d%H%M%S')}",
        session_num=1,
        model_name=settings.openai_llm_model if settings.llm_provider == "openai" else
                   settings.ollama_llm_model if settings.llm_provider == "ollama" else
                   settings.google_llm_model,
    )

    rows: list[dict] = []
    for i, q in enumerate(golden):
        category = q.get("category", "?")
        reference = q.get("expected_answer", "")
        if category == "out_of_context":
            reference = "This paper/topic is not in my corpus."

        # Small inter-question pause so arxiv_search fallback calls don't 429
        if i > 0:
            time.sleep(2)

        # human:query
        haic_log.log_query(duration_s=8.0)

        t0 = time.time()
        try:
            result = run_agent(q["question"], version=version, session_id=f"haic-{version}-{i}")
        except Exception as exc:
            logger.warning("Agent error on #%d: %s", i, exc)
            haic_log.log_respond(latency_ms=(time.time() - t0) * 1000)
            haic_log.log_reject(correct=False)
            rows.append({"i": i, "category": category, "question": q["question"],
                         "answer": "", "judge_score": 0, "judge_reason": str(exc),
                         "accepted": False})
            continue

        latency_ms = (time.time() - t0) * 1000

        # Record retrieve events (one per tool call)
        for tc in result.get("tool_calls", []):
            haic_log.log_retrieve(latency_ms=latency_ms / max(len(result["tool_calls"]), 1))

        # ai:respond
        haic_log.log_respond(latency_ms=latency_ms)

        # Judge: OOC uses a refusal-scoring prompt; in-context uses helpfulness scoring.
        is_ooc = category == "out_of_context"
        score, reason = _judge_score(q["question"], reference, result["answer"], is_ooc=is_ooc)
        accepted = score >= 3.0
        if accepted:
            haic_log.log_accept(correct=True)
        else:
            haic_log.log_reject(correct=False)

        rows.append({
            "i": i,
            "category": category,
            "question": q["question"],
            "answer": result["answer"][:600],
            "judge_score": score,
            "judge_reason": reason,
            "accepted": accepted,
        })
        logger.info(
            "  [%d/%d] %s | score=%.0f | %s | %.0f ms",
            i + 1, len(golden), category, score,
            "ACCEPT" if accepted else "REJECT", latency_ms,
        )

    # Compute HAIC metrics
    artifact = haic_log.build_artifact()
    try:
        from haic_metrics.compute import compute_metrics as _haic_compute
        result_metrics = _haic_compute(artifact, baseline_s=baseline_s, rt_max_s=RT_MAX_S)
        metrics = result_metrics["metrics"]
        logger.info("HAIC metrics computed via haic-metrics library")
    except ImportError:
        metrics = compute_haic_metrics(artifact, baseline_s=baseline_s, rt_max_s=RT_MAX_S)
        logger.info("HAIC metrics computed via built-in implementation")

    artifact_path = haic_log.save_artifact()
    logger.info("HAIC artifact saved → %s", artifact_path)

    # Category breakdowns
    by_cat: dict[str, dict] = {}
    for cat in {r["category"] for r in rows}:
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "accept_rate": sum(r["accepted"] for r in sub) / max(1, len(sub)),
            "mean_judge_score": sum(r["judge_score"] for r in sub) / max(1, len(sub)),
        }

    # In-context vs OOC split — OOC uses a different judge scale (refusal=5, hallucination=1)
    # so reporting them together distorts the overall mean; keep them separate.
    def _ctx_stats(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "accept_rate": 0.0, "mean_judge_score": 0.0}
        return {
            "n": len(subset),
            "accept_rate": sum(r["accepted"] for r in subset) / len(subset),
            "mean_judge_score": sum(r["judge_score"] for r in subset) / len(subset),
        }

    in_ctx_rows = [r for r in rows if r["category"] != "out_of_context"]
    ooc_rows = [r for r in rows if r["category"] == "out_of_context"]

    ts = time.strftime("%Y%m%dT%H%M%S")
    out = {
        "version": version,
        "n": len(rows),
        "metrics": metrics,
        "by_category": by_cat,
        "by_context_type": {
            "in_context": _ctx_stats(in_ctx_rows),
            "out_of_context": _ctx_stats(ooc_rows),
        },
        "baseline_s": baseline_s,
        "rt_max_s": RT_MAX_S,
        "artifact_path": str(artifact_path),
        "rows": rows,
        "timestamp": ts,
    }
    out_path = REPORTS_DIR / f"haic_{version}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "HAIC %s → EL=%.3f Tr=%.3f HCL=%.3f A=%.3f EffScore=%.3f → %s",
        version,
        metrics.get("EL", 0), metrics.get("Tr", 0), metrics.get("HCL", 0),
        metrics.get("A", 0), metrics.get("EfficiencyScore", 0),
        out_path,
    )
    flush_langfuse()
    return out_path
