# PaperPilot - Commands Reference

Complete command reference with examples and troubleshooting.

---

## Table of Contents

1. [Quick Start (start.sh)](#1-quick-start-startsh)
2. [Stack Lifecycle](#2-stack-lifecycle)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Agent Commands](#4-agent-commands)
5. [Evaluation](#5-evaluation)
6. [Development](#6-development)
7. [Docker Operations](#7-docker-operations)
8. [Portability](#8-portability)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start (start.sh)

The recommended way to start PaperPilot is the interactive setup script:

```bash
bash start.sh
```

This will:
- Ask which AI provider you want (OpenAI / Google / Ollama)
- Ask for your API key and update `.env` automatically
- Start all services with `docker compose up -d`
- Print all URLs and credentials immediately

After `start.sh` completes, services are at:

- PaperPilot Chat - http://localhost:8000
- Langfuse - http://localhost:3001 (admin@paperpilot.local / PaperPilot2026!)
- Qdrant - http://localhost:6333
- MinIO - http://localhost:9091 (minio / miniosecret)

On first run, the vector collections are empty. Ingest papers in a new terminal:

```bash
docker compose exec app python -m paperpilot.ingest
```

---

## 2. Stack Lifecycle

```bash
docker compose up -d       # start all services
docker compose down        # stop all services
docker compose logs -f app # tail app logs
```

### Make shortcuts

```bash
make up           # start all services
make down         # stop all services
make logs         # tail all service logs
make reset        # DESTRUCTIVE: stop + wipe data/raw, data/processed, cache.db
```

### Direct Docker Compose

```bash
docker compose up -d qdrant            # start only Qdrant
docker compose ps                      # check service health
docker compose down -v                 # stop + delete named volumes
docker compose build app               # rebuild app image after code change
docker compose up -d --build app       # rebuild + restart app
docker compose exec app bash           # shell into app container
```

---

## 3. Ingestion Pipeline

### Full pipeline (recommended)

```bash
make ingest
# or:
docker compose exec app python -m paperpilot.ingest
```

Takes 20-40 min on first run (ArXiv rate limiting).

### Step by step

```bash
make fetch             # download ~100 PDFs from ArXiv to data/raw/
make parse             # convert PDFs -> Markdown in data/processed/
make chunk-index-v1    # build papers_v1 collection (fixed-size chunks)
make chunk-index-v2    # build papers_v2 collection (section-aware + rerank)
```

### With the CLI directly

```bash
paperpilot ingest fetch              # download PDFs
paperpilot ingest parse              # parse to Markdown
paperpilot ingest index --version v1 # index v1
paperpilot ingest index --version v2 # index v2
paperpilot ops stats                 # verify collection sizes
paperpilot ops doctor                # health check all services
```

---

## 4. Agent Commands

### Ask a question

```bash
make ask Q="What is retrieval-augmented generation?"

# With CLI:
paperpilot ask "What is HyDE?" --version v2 --trace
paperpilot ask "Compare RAGAS and RAGChecker" --version v1
```

### Compare v1 vs v2 on a question

```bash
make compare Q="What is dense retrieval?"
paperpilot compare "What are the RAGAS metrics?"
```

### Launch the chat UI

```bash
make ui   # Chainlit at http://localhost:8000

# Directly:
chainlit run src/paperpilot/server/chainlit_app.py --host 0.0.0.0 --port 8000
```

### MCP server (bonus)

```bash
make mcp
# or:
paperpilot mcp serve   # stdio MCP server for Claude Desktop
python -m paperpilot.mcp.server
```

---

## 5. Evaluation

### Generate golden question set

```bash
make golden   # LLM-generate candidate Qs -> data/golden/golden_set.jsonl
```

Then manually curate `data/golden/golden_set.jsonl`.

### RAGAS evaluation

```bash
make eval-v1    # RAGAS on v1 -> reports/ragas_v1.json
make eval-v2    # RAGAS on v2 -> reports/ragas_v2.json
make eval-all   # v1 + v2 + tool call accuracy

# CLI:
paperpilot eval ragas --version v1
paperpilot eval ragas --version v2
```

Metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`

### Tool Call Accuracy

```bash
paperpilot eval tool-call-acc --version v2   # -> reports/tool_call_acc_v2.json
```

### HAIC evaluation

```bash
make haic
paperpilot eval haic --version v2   # -> reports/haic_v2.json
```

Metrics: mean judge score (1-5), accept rate, efficiency score.

---

## 6. Development

### Install locally

```bash
make install   # pip install -e ".[dev]"
```

### Tests

```bash
make test                              # pytest -q
pytest tests/test_chunkers.py -v      # specific test file
pytest -k "test_fixed" -v             # filter by name
```

### Code quality

```bash
make fmt    # ruff format src tests
make lint   # ruff check + mypy src
```

### Healthcheck

```bash
make doctor                    # check all services
```

### Collection stats

```bash
make stats
paperpilot ops stats   # vector count per collection
```

---

## 7. Docker Operations

### Rebuild only the app (after code changes)

```bash
docker compose build app
docker compose up -d app
```

### Full rebuild (broken cache)

```bash
docker compose build --no-cache app
```

### Access service shells

```bash
docker exec -it paperpilot-app bash                     # Python app
docker exec -it paperpilot-qdrant bash                  # Qdrant
docker exec -it paperpilot-postgres psql -U postgres    # Postgres
```

### Delete Qdrant collections (force re-index)

```bash
curl -s -X DELETE http://localhost:6333/collections/papers_v1
curl -s -X DELETE http://localhost:6333/collections/papers_v2
make chunk-index-v1
make chunk-index-v2
```

### View service resource usage

```bash
docker stats --no-stream
```

---

## 8. Portability

PaperPilot uses bind mounts - all data lives in `./data/`. Copy the entire project folder to move between machines.

### Before moving to an external drive

```bash
make down   # stop all containers (flushes Postgres WAL)
# Then copy the project directory
```

### On the destination machine

```bash
bash start.sh   # start services; they find existing data automatically
```

---

## 9. Troubleshooting

### Services won't start

```bash
docker compose ps                  # check status
docker compose logs qdrant         # check specific service
docker compose logs langfuse-web
```

Check ports are free: 6333, 6334, 3001, 9090, 9091, 8000

### Ingestion fails on fetch step

ArXiv rate-limits downloads. The fetcher retries automatically. If it fails:

```bash
# Wait 5 minutes, then:
docker compose exec app python -m paperpilot.ingest
```

### Qdrant collection is empty / stats show 0 vectors

```bash
curl -s -X DELETE http://localhost:6333/collections/papers_v2
make chunk-index-v2
make stats
```

### RAGAS eval is slow / times out

RAGAS calls the judge LLM once per question. With 84 questions this is 300+ API calls. Budget ~30 min.

### Out-of-memory on reranker

The BGE reranker loads a ~280 MB model on first query. If OOM, reduce dense retrieval in `.env`:

```
TOP_K_V2_DENSE=4   # was 8
```

### Langfuse traces not appearing

```bash
docker compose logs langfuse-worker
docker compose logs langfuse-web
```

Langfuse is pre-configured - no manual setup needed. The project, keys, and admin user are created automatically on first start.

### `paperpilot` command not found

```bash
pip install -e ".[dev]"
# or:
python -m paperpilot.cli ask "..."
```

### SQLite cache corruption (after crash or FAT32 copy)

```bash
rm data/cache.db   # embeddings will be re-fetched (slow but safe)
```

### ModuleNotFoundError in Docker

```bash
docker compose build --no-cache app
docker compose up -d app
```

### Reset everything and start from scratch

```bash
make down
rm -rf data/qdrant_storage data/pg_data data/clickhouse_data data/redis_data data/minio_data
make up
docker compose exec app python -m paperpilot.ingest
```

---

## Environment Variables Reference

Copy `.env.example` to `.env` and fill in the required values, or use `bash start.sh` which does this interactively.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | yes | `openai` | `openai` or `google` or `ollama` |
| `OPENAI_API_KEY` | if openai | - | OpenAI API key |
| `OPENAI_LLM_MODEL` | no | `gpt-4.1-mini` | LLM model |
| `OPENAI_EMBED_MODEL` | no | `text-embedding-3-small` | Embedding model |
| `GEMINI_API_KEY` | if google | - | Google Gemini API key |
| `GOOGLE_LLM_MODEL` | no | `gemini-2.0-flash` | Google LLM model |
| `OLLAMA_BASE_URL` | if ollama | `http://host.docker.internal:11434` | Ollama endpoint |
| `OLLAMA_LLM_MODEL` | if ollama | `llama3.1:8b` | Ollama LLM model |
| `OLLAMA_EMBED_MODEL` | if ollama | `bge-m3` | Ollama embed model |
| `QDRANT_URL` | no | `http://localhost:6333` | Qdrant endpoint |
| `RERANKER_MODEL` | no | `BAAI/bge-reranker-base` | HuggingFace reranker |
| `TOP_K_V2_DENSE` | no | `8` | Dense retrieval candidates for v2/v3 |
| `TOP_K_V2_RERANK` | no | `4` | Top-k after reranking |
| `LANGFUSE_HOST` | no | `http://localhost:3001` | Langfuse endpoint |
| `CACHE_DB_PATH` | no | `data/cache.db` | SQLite cache location |
| `AGENT_MAX_ITERATIONS` | no | `6` | Max agent iterations |
