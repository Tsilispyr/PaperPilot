.PHONY: help install up down logs ingest fetch parse chunk-index-v1 chunk-index-v2 chunk-index-v3 \
        golden eval-v1 eval-v2 eval-v3 eval-all haic haic-v3 ui mcp clean reset test fmt lint \
        export-portable

PYTHON ?= python
COMPOSE ?= docker compose

help:
	@echo "PaperPilot - common commands"
	@echo ""
	@echo "Stack lifecycle:"
	@echo "  make up                 - start qdrant + langfuse + app"
	@echo "  make down               - stop everything"
	@echo "  make logs               - tail logs"
	@echo "  make reset              - drop all volumes (destructive)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make install            - install deps locally"
	@echo "  make ingest             - fetch + parse + chunk + index v1 + v2"
	@echo "  make fetch              - download arxiv PDFs only"
	@echo "  make parse              - parse PDFs to markdown only"
	@echo "  make chunk-index-v1     - build v1 collection (fixed chunks)"
	@echo "  make chunk-index-v2     - build v2 collection (section-aware)"
	@echo ""
	@echo "Eval:"
	@echo "  make golden             - LLM-generate candidate golden Qs"
	@echo "  make eval-v1            - run RAGAS + tool-call-acc on v1"
	@echo "  make eval-v2            - run RAGAS + tool-call-acc on v2"
	@echo "  make eval-all           - eval-v1 + eval-v2"
	@echo "  make haic               - run HAIC benchmarking suite"
	@echo ""
	@echo "Apps:"
	@echo "  make ui                 - launch chainlit at :8000"
	@echo "  make mcp                - launch MCP server (stdio)"
	@echo ""
	@echo "Portability:"
	@echo "  make export-portable    - checkpoint SQLite WAL; safe to copy project to external drive"
	@echo ""
	@echo "Dev:"
	@echo "  make test               - pytest"
	@echo "  make fmt                - ruff format"
	@echo "  make lint               - ruff check + mypy"

install:
	$(PYTHON) -m pip install -e ".[dev]"

up:
	$(COMPOSE) up -d
	@echo "Qdrant     http://localhost:6333/dashboard"
	@echo "Langfuse   http://localhost:3001"
	@echo "App UI     http://localhost:8000"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

reset:
	$(COMPOSE) down -v
	rm -rf data/raw data/processed data/cache.db
	@echo "All volumes + local data wiped."

ingest: fetch parse chunk-index-v1 chunk-index-v2

fetch:
	$(PYTHON) -m paperpilot.cli ingest fetch

parse:
	$(PYTHON) -m paperpilot.cli ingest parse

chunk-index-v1:
	$(PYTHON) -m paperpilot.cli ingest index --version v1

chunk-index-v2:
	$(PYTHON) -m paperpilot.cli ingest index --version v2

chunk-index-v3:
	$(PYTHON) -m paperpilot.cli ingest index --version v3 --recreate

golden:
	$(PYTHON) -m paperpilot.cli eval golden-gen --n 50

eval-v1:
	$(PYTHON) -m paperpilot.cli eval ragas --version v1

eval-v2:
	$(PYTHON) -m paperpilot.cli eval ragas --version v2

eval-v3:
	$(PYTHON) -m paperpilot.cli eval ragas --version v3
	$(PYTHON) -m paperpilot.cli eval tool-call-acc --version v3

eval-all: eval-v1 eval-v2 eval-v3
	$(PYTHON) -m paperpilot.cli eval tool-call-acc --version v2

haic:
	$(PYTHON) -m paperpilot.cli eval haic --version v2

haic-v3:
	$(PYTHON) -m paperpilot.cli eval haic --version v3

ui:
	chainlit run src/paperpilot/server/chainlit_app.py --host 0.0.0.0 --port 8000

mcp:
	$(PYTHON) -m paperpilot.mcp.server

doctor:
	$(PYTHON) -m paperpilot.cli ops doctor

stats:
	$(PYTHON) -m paperpilot.cli ops stats

ask:
	@if [ -z "$(Q)" ]; then echo 'Usage: make ask Q="your question"'; exit 1; fi
	$(PYTHON) -m paperpilot.cli ask "$(Q)" --version v2 --trace

compare:
	@if [ -z "$(Q)" ]; then echo 'Usage: make compare Q="your question"'; exit 1; fi
	$(PYTHON) -m paperpilot.cli compare "$(Q)"

test:
	pytest -q

fmt:
	ruff format src tests

lint:
	ruff check src tests
	mypy src

export-portable:
	@echo "Checkpointing SQLite WAL files..."
	$(PYTHON) -c "\
import sqlite3, pathlib; \
db = pathlib.Path('data/cache.db'); \
cx = sqlite3.connect(str(db)) if db.exists() else None; \
cx and (cx.execute('PRAGMA wal_checkpoint(TRUNCATE)'), cx.close()); \
print('WAL checkpointed:', db if db.exists() else '(no cache.db found - OK)') \
"
	@echo ""
	@echo "Docker services use ./data/ bind mounts (no named volumes)."
	@echo "Run 'make down' first, then copy the entire project directory."
	@echo "The project is portable to any machine or exFAT/FAT32 external drive."

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
