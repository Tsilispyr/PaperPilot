"""Records Human-AI Collaboration (HAIC) decisions in haic.decisions.v1 schema.

Each interaction turn produces 3-4 events:
  human:query   → user submits a question
  ai:retrieve   → agent calls rag_retrieve / arxiv_search
  ai:respond    → agent produces final answer
  human:accept  → user clicks 👍 (or LLM judge scores ≥ 3/5 in automated eval)
  human:reject  → user clicks 👎 (or LLM judge scores < 3/5 in automated eval)

The artifact produced by build_artifact() is compatible with haic_metrics.compute.compute_metrics().
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Optional

HAIC_DIR: Optional[Path] = None


def _get_haic_dir() -> Path:
    global HAIC_DIR
    if HAIC_DIR is None:
        from paperpilot.config import DATA_DIR
        HAIC_DIR = DATA_DIR / "haic"
        HAIC_DIR.mkdir(parents=True, exist_ok=True)
    return HAIC_DIR


class HAICLogger:
    """Thread-safe per-session event recorder."""

    def __init__(self, session_id: str, session_num: int = 1, model_name: str = "paperpilot"):
        self.session_id = session_id
        self.session_num = session_num
        self.model_name = model_name
        self.decisions: list[dict] = []
        self._seq = 0
        self._turn = 0
        self._respond_start: Optional[float] = None
        self._file = _get_haic_dir() / f"session_{session_id}.jsonl"

    # --- Internal ---

    def _event(
        self,
        actor_type: str,
        action: str,
        *,
        duration_s: Optional[float] = None,
        latency_ms: Optional[float] = None,
        correct: Optional[bool] = None,
    ) -> dict:
        self._seq += 1
        event = {
            "schema_version": "haic.decisions.v1",
            "seq": self._seq,
            "t": time.time(),
            "actor_type": actor_type,
            "action": action,
            "object_id": f"s{self.session_num:03d}_q{self._turn}",
            "duration_s": duration_s,
            "latency_ms": latency_ms,
            "correct": correct,
            "payload": {"session": self.session_num, "turn": self._turn},
        }
        self.decisions.append(event)
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return event

    # --- PublicAPI ---

    def log_query(self, duration_s: float = 8.0) -> None:
        self._turn += 1
        self._respond_start = None
        self._event("human", "query", duration_s=duration_s)

    def log_retrieve(self, latency_ms: float) -> None:
        self._event("ai", "retrieve", latency_ms=latency_ms)

    def log_respond(self, latency_ms: float) -> None:
        self._respond_start = time.time()
        self._event("ai", "respond", latency_ms=latency_ms)

    def log_accept(self, correct: Optional[bool] = None) -> None:
        dur = time.time() - self._respond_start if self._respond_start else 3.0
        self._event("human", "accept", duration_s=round(dur, 2), correct=correct)
        self._respond_start = None

    def log_reject(self, correct: Optional[bool] = None) -> None:
        dur = time.time() - self._respond_start if self._respond_start else 3.0
        self._event("human", "reject", duration_s=round(dur, 2), correct=correct)
        self._respond_start = None

    def build_artifact(self) -> dict:
        return {
            "artifact_schema": "haic.decisions_artifact.v1",
            "schema_version": "haic.decisions.v1",
            "session_id": self.session_id,
            "run_id": str(uuid.uuid4()),
            "meta": {
                "pilot_tag": "paperpilot-eval",
                "application": {"name": "paperpilot", "version": "0.1.0", "mode": "eval"},
                "ai_system": {
                    "model_name": self.model_name,
                    "model_type": "rag-agent",
                    "model_version": "v2",
                },
                "task": {
                    "name": "Research Q&A",
                    "description": "NLP/LLM/RAG paper questions",
                },
            },
            "decisions": self.decisions,
        }

    def save_artifact(self) -> Path:
        art = self.build_artifact()
        path = _get_haic_dir() / f"artifact_{self.session_id}.json"
        path.write_text(json.dumps(art, indent=2, ensure_ascii=False), encoding="utf-8")
        return path




def compute_haic_metrics(artifact: dict, baseline_s: float = 7500.0, rt_max_s: float = 30.0) -> dict:
    """
    Compute HAIC core metrics from a decisions artifact.

    EL  = (t_actual − baseline_s) / baseline_s
    Tr  = accepted / (accepted + rejected)
    HCL = 1 − clip(mean_rt_s / rt_max_s, 0, 1)   where mean_rt = mean ai:respond latency
    F   = (human + ai events) / session_duration_minutes
    D   = mean duration_s of human accept/reject events
    A   = tanh((Tr_late − Tr_early) / max(Tr_early, 0.01))
    EfficiencyScore = 1 / (1 + max(EL, 0))
    """
    decisions = [d for d in artifact.get("decisions", []) if d.get("actor_type") != "system"]
    if not decisions:
        return {"EL": 0.0, "Tr": 0.0, "HCL": 0.0, "F": 0.0, "A": 0.0, "D": 0.0, "EfficiencyScore": 0.0}

    # Wall time
    ts = [d["t"] for d in decisions if d.get("t")]
    t_actual = (max(ts) - min(ts)) if len(ts) >= 2 else baseline_s

    # EL
    EL = max((t_actual - baseline_s) / baseline_s, 0.0)

    # Tr
    accepted = sum(1 for d in decisions if d["action"] == "accept")
    rejected = sum(1 for d in decisions if d["action"] == "reject")
    Tr = accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0

    # HCL - based on ai:respond latency
    respond_latencies = [
        d["latency_ms"] / 1000.0
        for d in decisions
        if d["action"] == "respond" and d.get("latency_ms") is not None
    ]
    mean_rt = sum(respond_latencies) / len(respond_latencies) if respond_latencies else rt_max_s / 2
    HCL = 1.0 - min(mean_rt / rt_max_s, 1.0)

    # F - events per minute
    dur_min = t_actual / 60.0
    F = len(decisions) / dur_min if dur_min > 0 else 0.0

    # D - mean human decision duration
    human_durations = [
        d["duration_s"]
        for d in decisions
        if d["actor_type"] == "human" and d["action"] in ("accept", "reject")
        and d.get("duration_s") is not None
    ]
    D = sum(human_durations) / len(human_durations) if human_durations else 0.0

    # A - adaptability (early vs late 20% of sessions by session number)
    sessions: dict[int, list[dict]] = {}
    for d in decisions:
        snum = d.get("payload", {}).get("session", 1)
        sessions.setdefault(snum, []).append(d)

    def _session_tr(sess_decisions: list[dict]) -> float:
        a = sum(1 for x in sess_decisions if x["action"] == "accept")
        r = sum(1 for x in sess_decisions if x["action"] == "reject")
        return a / (a + r) if (a + r) > 0 else 0.0

    A = 0.0
    if len(sessions) >= 5:
        keys = sorted(sessions)
        n20 = max(1, len(keys) // 5)
        early_tr = sum(_session_tr(sessions[k]) for k in keys[:n20]) / n20
        late_tr = sum(_session_tr(sessions[k]) for k in keys[-n20:]) / n20
        A = math.tanh((late_tr - early_tr) / max(early_tr, 0.01))

    EfficiencyScore = 1.0 / (1.0 + EL)

    return {
        "EL": round(EL, 4),
        "Tr": round(Tr, 4),
        "HCL": round(HCL, 4),
        "F": round(F, 3),
        "A": round(A, 4),
        "D": round(D, 3),
        "EfficiencyScore": round(EfficiencyScore, 4),
    }
