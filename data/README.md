# data/

All runtime data. Everything here is either gitignored (PDFs, Markdown, cache) or tracked (golden set, HAIC logs).

## Structure

```
data/
├── raw/           ← downloaded ArXiv PDFs + .meta.json sidecars (gitignored)
├── processed/     ← parsed Markdown + .meta.json sidecars (gitignored)
├── golden/        ← curated Q&A evaluation set (tracked)
├── haic/          ← per-session HAIC event logs (tracked)
└── cache.db       ← SQLite embedding + judge call cache (gitignored)
```

## raw/

One PDF and one meta.json per paper:
```
2309.15217v1.pdf          ← original ArXiv PDF
2309.15217v1.meta.json    ← {title, authors, year, primary_category, …}
```

Populated by `paperpilot ingest fetch`. ~100 files, ~200 MB total.

## processed/

One Markdown and one meta.json per paper:
```
2309.15217v1.md           ← pymupdf4llm output preserving section headers
2309.15217v1.meta.json    ← same metadata as raw/
```

Populated by `paperpilot ingest parse`. Section headers (`## Abstract`, `## Method`, …) enable the v2 SectionAwareChunker to split at semantic boundaries rather than arbitrary token counts.

## golden/

```
golden_set.jsonl     ← 84 curated Q&A pairs used in RAGAS + Tool Call Accuracy eval
seed_questions.jsonl ← initial human-written seed used to bootstrap golden-gen
```

Each line in `golden_set.jsonl`:
```json
{
  "category": "definitional",
  "question": "What is HyDE?",
  "expected_answer": "…",
  "must_cite": ["2212.10496"],
  "expected_tools": ["rag_retrieve"],
  "rationale": "..."
}
```

Categories: `definitional`, `comparative`, `methodological`, `attribution`, `numerical`, `out_of_context`

## haic/

Per-session event logs from the Chainlit UI (recorded when users click 👍/👎):
```
session_<uuid>.jsonl      ← raw event stream (query/retrieve/respond/accept/reject)
artifact_<uuid>.json      ← computed HAIC artifact with metric scores
```

Used by `paperpilot eval haic` to compute aggregate HAIC metrics.

## cache.db

SQLite database with two tables:
- `embeddings` — SHA-256(model, text) → float list; avoids re-calling the embed API
- `judges` — SHA-256(model, prompt) → JSON; avoids re-calling the judge LLM

Safe to delete — the pipeline will rebuild it. Before deleting or moving to a new machine, run `make export-portable` to checkpoint the WAL file.
