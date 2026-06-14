# PaperPilot

> Agentic RAG over NLP / LLM / RAG / Agents research papers.

Course project for CSIS132 - Εφαρμογές Επιστήμης Δεδομένων & Τεχνητής Νοημοσύνης.

| | |
|---|---|
| Φοιτητής | Σπυρίδων Τσιλιμπώκος (ΑΜ: 25118) |
| Διδάσκων | Γεώργιος Φατούρος |
| Ίδρυμα | Χαροκόπειο Πανεπιστήμιο / Harokopio University of Athens |
| Εξάμηνο | Εαρινό 2025-2026 |

PaperPilot is a domain-specialised retrieval-augmented agent. Ask it anything about recent NLP/LLM/RAG/agentic-systems research and it returns grounded, cited answers from a curated corpus of ~100 ArXiv papers (2020-2026). When the corpus does not cover a topic it falls back to live ArXiv search.

---

## Quick Start

```bash
bash start.sh
```

The script will ask which AI provider you want (OpenAI / Google / Ollama), prompt for your API key, update `.env`, and start all services.

After setup, services are available at:

| Service | URL |
|---|---|
| PaperPilot Chat UI | http://localhost:8000 |
| Langfuse (tracing) | http://localhost:3001 |
| Qdrant (vector DB) | http://localhost:6333 |
| MinIO (storage) | http://localhost:9091 |

PaperPilot Chat UI login:

| Role | Username | Password |
|---|---|---|
| Admin | `paperpilot` | `research2026` |
| Guest (free access) | `guest` | *(any)* |
| New user | *(choose, min 3 chars)* | *(choose, min 6 chars)* |

First login with a new username auto-registers the account. Returning users log in with the same credentials.

Langfuse credentials: `admin@paperpilot.local` / `PaperPilot2026!` (pre-configured, no setup needed).

MinIO credentials: `minio` / `miniosecret`

### First run - ingest papers

On first launch the vector collections are empty. Run in a new terminal:

```bash
docker compose exec app python -m paperpilot.ingest
```

Downloads ~100 ArXiv PDFs, parses, embeds, and indexes them. Takes 20-40 min.

### Stop

```bash
docker compose down
```

---

## Architecture

### Multi-Agent Pipeline (Planner - Researcher - Synthesizer)

```
User Question
      |
      v
+-------------+   structured plan
|   Planner   |----------------------+
|  (fast LLM) |                      |
+-------------+                      |
      | mode = quick_qa              | mode = out_of_context
      |         deep_analysis        |
      v                              |
+---------------------------------+  |
|          Researcher             |  |
|  asyncio.gather:                |  |
|   +- rag_retrieve  --> Qdrant   |  |
|   +- arxiv_search --> ArXiv API |  |
|  -> distils chunks to bullets   |  |
+---------------------------------+  |
      |                              |
      v                              |
+---------------------------------+  |
|         Synthesizer             |<-+
|  (strong LLM)                   |
|  -> cited Markdown answer       |
+---------------------------------+
```

### Ingestion Pipeline

```
ArXiv API
    |  arxiv_fetch.py - download ~100 PDFs with metadata
    v
data/raw/*.pdf + *.meta.json
    |  parse.py - pymupdf4llm -> section-aware Markdown
    v
data/processed/*.md + *.meta.json
    |
    +- FixedSizeChunker (v1) - 512 tokens, 50 overlap
    +- SectionAwareChunker (v2) - split by Abstract/Method/Eval/...
    +- TableAwareChunker (v3) - preserves Markdown tables intact
         |
         v  embeddings.py - text-embedding-3-small (SQLite-cached)
         v
    +------------+------------+------------+
    |  papers_v1 |  papers_v2 |  papers_v3 |  Qdrant collections
    +-----+------+-----+------+-----+------+
          |            |            |  rerank.py (v2/v3)
          |            |            |  BAAI/bge-reranker-base + score threshold
          +------------+------------+
                  |
                  v
          Multi-Agent Graph
                  |
          +-----------------+
          | Chainlit UI     |  <- streaming node transitions
          | Langfuse        |  <- full trace visibility
          +-----------------+
```

---

## Evaluation Results

### RAGAS (retrieval + generation quality, n=84)

Overall scores are OOC-diluted: 59/84 questions are out-of-context (agent refuses → CP≈0). In-context only (25 Qs): v3 CP=0.558, CR=0.590, F=0.647.

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Context Precision | 0.163 | **0.190** | 0.166 |
| Context Recall | 0.354 | 0.363 | **0.455** |
| Faithfulness | **0.306** | 0.292 | 0.284 |
| Answer Relevancy | **0.477** | 0.438 | 0.335 |

### Tool Call Accuracy (n=84)

v1 uses tools for OOC questions (incorrect), inflating its overall score. v2/v3 correctly refuse tools for OOC (TCA=0 for OOC), which is the right behavior.

| Version | Overall | Definitional | Numerical | Out-of-context |
|---|---|---|---|---|
| v1 | **0.619** | 1.00 | 1.00 | 0.458 |
| v2 | 0.298 | 1.00 | 1.00 | 0.000 |
| v3 | 0.298 | 1.00 | 1.00 | 0.000 |

### HAIC Evaluation

**v1** (4-axis scoring, n=45): Helpfulness=3.76, Trust=2.82, Effort Saved=3.31, Harm=1.07

**v2/v3** (LLM-as-judge, n=84) - formats are incompatible, do not compare v1 vs v2/v3 directly:

| Metric | v2 | v3 |
|---|---|---|
| Mean Judge Score (1-5) | 2.583 | **4.024** |
| Accept Rate | 0.464 | **0.786** |
| Efficiency Score | 1.0 | 1.0 |

Full results and charts: [`reports/`](reports/)

---

## v1 - v2 - v3 Improvement Story

| Component | v1 (baseline) | v2 (improved) | v3 (final) |
|---|---|---|---|
| Chunking | Fixed 512 tokens | Section-aware | Table-aware (preserves tables) |
| Retrieval | Dense top-5 | Dense top-8 + rerank -> top-4 | Same as v2 + diversity filter |
| Score gate | None | Reranker threshold | Same |
| Filtering | None | Year/category heuristic | Same |
| Agent | Multi-agent graph | Same | Same |

---

## Repository Layout

```
PaperPilot/
|
+-- README.md                    <- this file
+-- COMMANDS.md                  <- every command + troubleshooting
+-- start.sh                     <- interactive setup + launch
|
+-- docker-compose.yml           <- Qdrant + Langfuse + app stack
+-- Dockerfile                   <- Python app image
+-- Makefile                     <- task automation
+-- pyproject.toml               <- Python dependencies
|
+-- data/
|   +-- golden/                  <- 84-question evaluation set
|   +-- haic/                    <- per-session HAIC event logs
|   +-- raw/                     <- ArXiv PDFs (gitignored, ~200 MB)
|   +-- processed/               <- parsed Markdown (gitignored)
|   +-- cache.db                 <- SQLite cache (gitignored)
|
+-- reports/                     <- evaluation outputs
|   +-- ragas_v*.json / *.csv    <- RAGAS scores per version
|   +-- tool_call_acc_v*.json    <- Tool call accuracy per version
|   +-- haic_v*.json             <- HAIC scores per version
|   +-- charts/                  <- PNG charts
|   +-- report_gr.html           <- full HTML report (Greek)
|
+-- src/paperpilot/
|   +-- agent/                   <- LangGraph graph (Planner/Researcher/Synthesizer)
|   +-- ingest/                  <- ArXiv fetch, PDF parse, chunk, embed, index
|   +-- retrieval/               <- Qdrant dense search + cross-encoder rerank
|   +-- eval/                    <- RAGAS, Tool Call Accuracy, HAIC evaluation
|   +-- cache/                   <- SQLite embedding + judge cache
|   +-- observability/           <- Langfuse + HAIC event logger
|   +-- server/                  <- Chainlit chat UI
|   +-- mcp/                     <- MCP server (bonus, exposes rag_retrieve)
|   +-- config.py                <- all settings via pydantic-settings
|   +-- cli.py                   <- Typer CLI
|
+-- tests/                       <- unit tests (chunkers, filters, TCA scorer)
+-- public/                      <- Chainlit static assets (CSS, logos)
```

---

## System Requirements

| Requirement | Minimum |
|---|---|
| Docker + Compose v2 | required |
| RAM | 8 GB (16 GB recommended for eval) |
| Disk | ~6 GB (PDFs + Qdrant + Langfuse data) |
| LLM | OpenAI API key, Google Gemini key, or local Ollama |

### LLM Provider options

```bash
bash start.sh   # prompts you to choose interactively
```

Or set manually in `.env`:

```
LLM_PROVIDER=openai    # OPENAI_API_KEY required
LLM_PROVIDER=google    # GEMINI_API_KEY required
LLM_PROVIDER=ollama    # fully local, no API key
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Multi-Agent | LangGraph StateGraph | Planner/Researcher/Synthesizer nodes |
| LLM | gpt-4.1-mini (OpenAI) | Fast, cheap, configurable |
| Embeddings | text-embedding-3-small | Cost-quality balance, SQLite-cached |
| Reranker | BAAI/bge-reranker-base | Free, CPU-local |
| Vector DB | Qdrant | Three collections: papers_v1, papers_v2, papers_v3 |
| Eval | RAGAS + Tool Call Accuracy + HAIC | Per course spec |
| Tracing | Langfuse (self-hosted) | Full local control |
| UI | Chainlit | Streaming node transitions |
| Bonus | MCP server (stdio) | Exposes rag_retrieve to Claude Desktop |
| Deploy | Docker Compose | Drive-agnostic bind mounts in data/ |

---

For every command and troubleshooting see [COMMANDS.md](COMMANDS.md).
