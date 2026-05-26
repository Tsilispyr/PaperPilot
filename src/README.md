# src/

Python source code for PaperPilot. All modules live under `src/paperpilot/`.

## Modules

| Module | Description |
|---|---|
| `agent/` | LangGraph multi-agent graph: Planner, Researcher, Synthesizer nodes |
| `ingest/` | ArXiv fetch, PDF parse, chunking (v1/v2/v3), embedding, Qdrant indexing |
| `retrieval/` | Qdrant dense search, metadata filters, cross-encoder reranking |
| `eval/` | RAGAS, Tool Call Accuracy, and HAIC evaluation scripts |
| `cache/` | SQLite-backed cache for embeddings and LLM judge calls (SHA-256 keyed) |
| `observability/` | Langfuse tracing setup, HAIC event logger |
| `server/` | Chainlit chat UI with streaming node transitions |
| `mcp/` | MCP server exposing `rag_retrieve` to Claude Desktop (bonus) |
| `config.py` | All settings via pydantic-settings (reads from `.env`) |
| `cli.py` | Typer CLI entrypoint (`paperpilot ask / eval / ingest ...`) |

## Entry points

```bash
# Chat UI (via Docker)
chainlit run src/paperpilot/server/chainlit_app.py --host 0.0.0.0 --port 8000

# CLI
paperpilot ask "What is HyDE?"
paperpilot eval ragas --version v2

# Ingestion
python -m paperpilot.ingest
```
