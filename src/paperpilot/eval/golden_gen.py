"""LLM-generate candidate golden questions from random chunks of indexed papers.

Output: data/golden/candidates.jsonl — review by hand, keep the best 30+,
save the curated subset as data/golden/golden_set.jsonl.

Question categories (must hit all 6):
  - definitional       (What is HyDE?)
  - comparative        (RAGAS vs RAGChecker — what's different?)
  - methodological     (How does ReAct combine reasoning and acting?)
  - attribution        (Which paper proposed Self-RAG?)
  - numerical          (What MMLU score did GPT-4 achieve?)
  - out_of_context     (negative — answer should be 'not in corpus')
"""
from __future__ import annotations

import json
import logging
import random
from typing import Literal

from openai import OpenAI

from paperpilot.config import GOLDEN_DIR, settings
from paperpilot.ingest.parse import load_processed
from paperpilot.ingest.chunk import SectionAwareChunker

logger = logging.getLogger(__name__)

CATEGORIES = ["definitional", "comparative", "methodological", "attribution", "numerical", "out_of_context"]

GEN_PROMPT = """\
You are creating a high-quality QA evaluation set for a RAG system over recent NLP/LLM/RAG/Agents papers.

Given the excerpt below, write ONE non-trivial question of category **{category}** whose answer is verifiable from the excerpt (or, for category=out_of_context, intentionally NOT answerable from this excerpt).

Return strict JSON only, no markdown, with keys:
{{
  "question": "...",
  "expected_answer": "...",     // concise reference answer
  "must_cite": ["..."],         // paper titles or arxiv ids that must be cited
  "category": "{category}",
  "rationale": "..."            // why this is a good test question
}}

Hard rules:
- For 'numerical': the answer must include a specific number (score, %, hyperparameter, dataset size).
- For 'out_of_context': the question must be plausibly NLP-related but unanswerable from this excerpt — and ideally not from the corpus at all.
- For 'attribution': the question must ask 'which paper / who proposed X' and X must be a named technique/model/metric.
- Keep questions self-contained (don't say 'in this paper' — name the technique/model directly).
- Keep expected_answer short (1–3 sentences max).

Excerpt:
\"\"\"
{excerpt}
\"\"\"
"""


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def generate_candidates(n_candidates: int = 50, seed: int = 7) -> int:
    rng = random.Random(seed)
    papers = load_processed()
    if not papers:
        raise RuntimeError("No parsed papers. Run `make parse` first.")

    chunker = SectionAwareChunker()
    pool = []
    for meta, text in papers:
        for c in chunker.chunk(meta, text):
            # Prefer method/results/abstract for QA quality
            if c.section_type in ("abstract", "method", "results", "experiments", "introduction"):
                pool.append((meta, c))
    if not pool:
        raise RuntimeError("Empty chunk pool — no usable sections found.")
    rng.shuffle(pool)

    client = _client()
    out_path = GOLDEN_DIR / "candidates.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(n_candidates):
            category = CATEGORIES[i % len(CATEGORIES)]
            meta, chunk = pool[i % len(pool)]
            prompt = GEN_PROMPT.format(category=category, excerpt=chunk.text[:3000])
            try:
                resp = client.chat.completions.create(
                    model=settings.openai_judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(resp.choices[0].message.content or "{}")
                obj.setdefault("category", category)
                obj["source_paper_id"] = meta.paper_id
                obj["source_paper_title"] = meta.title
                obj["source_section"] = chunk.section_title
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                logger.warning("Generation failed (#%d, %s): %s", i, category, exc)
                continue

    logger.info("Wrote %d candidates → %s", written, out_path)
    return written


def load_golden_set() -> list[dict]:
    """Load the human-curated golden set (after manual review).

    Falls back to the bundled `seed_questions.jsonl` (30 hand-picked Qs across all 6
    categories) so the eval pipeline runs end-to-end on a fresh install. The curated
    `golden_set.jsonl` should always be preferred once you've reviewed candidates.
    """
    curated = GOLDEN_DIR / "golden_set.jsonl"
    seeds = GOLDEN_DIR / "seed_questions.jsonl"
    p = curated if curated.exists() else seeds
    if not p.exists():
        raise FileNotFoundError(
            f"Neither {curated} nor {seeds} found. Generate candidates with `make golden` "
            f"or commit a hand-picked seed set to {seeds}."
        )
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{p} is empty.")
    return rows
