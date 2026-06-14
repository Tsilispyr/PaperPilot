"""Central, env-driven configuration for PaperPilot.

Everything that can change between environments lives here, loaded from `.env`
(via pydantic-settings). Modules import `settings` and never read os.environ directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]   # paperpilot/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
REPORTS_DIR = ROOT_DIR / "reports"

for _p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, GOLDEN_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["openai", "ollama", "google"] = "google"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_llm_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_judge_model: str = "gpt-4.1-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_judge_model: str = "llama3.1:8b"

    google_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_llm_model: str = "gemini-1.5-flash"
    google_embed_model: str = "text-embedding-004"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_v1: str = "papers_v1"
    qdrant_collection_v2: str = "papers_v2"
    qdrant_collection_v3: str = "papers_v3"

    # --- Langfuse ---
    # Use http://localhost:3001 for local runs (host machine).
    # Use http://langfuse-web:3000 only inside Docker containers.
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_project: str = "paperpilot"

    # --- Ingestion ---
    arxiv_max_papers: int = 150
    arxiv_queries: str = (
        # Title-targeted queries to get foundational/methodology papers, not application papers.
        # Sort is Relevance (set in arxiv_fetch.py) so the most cited/relevant appear first.
        "ti:(\"retrieval-augmented generation\" OR \"retrieval augmented generation\" OR \"Self-RAG\" OR \"HyDE\" OR \"FLARE\" OR \"RAPTOR\")"
        "|cat:cs.CL AND (ti:(ReAct) OR ti:(Toolformer) OR abs:(\"reasoning and acting in language models\"))"
        "|cat:cs.CL AND (ti:(RAGAS) OR ti:(faithfulness) OR ti:(\"RAG evaluation\") OR ti:(RAGChecker))"
        "|cat:cs.CL AND (ti:(\"dense retrieval\") OR ti:(\"sentence embedding\") OR ti:(reranking) OR ti:(\"text embedding\"))"
        "|cat:cs.CL AND (ti:(LLaMA) OR ti:(Mistral) OR ti:(\"chain-of-thought\") OR ti:(\"in-context learning\") OR ti:(\"instruction tuning\"))"
    )
    arxiv_from_date: str = "2020-01-01"
    arxiv_to_date: str = "2026-04-30"
    # Comma-separated ArXiv IDs to always fetch regardless of queries.
    # Used for foundational papers whose titles don't match our query terms
    # (e.g. HyDE = "Precise Zero-Shot Dense Retrieval..." - ti:HyDE finds nothing).
    arxiv_seed_ids: str = (
        "2212.10496,"   # HyDE
        "2210.03629,"   # ReAct
        "2310.11511,"   # Self-RAG
        "2309.15217,"   # RAGAS
        "2408.08067,"   # RAGChecker
        "2404.16130,"   # GraphRAG (Microsoft)
        "2309.07597,"   # BGE embeddings (C-Pack / BAAI)
        "2005.11401,"   # Lewis et al. original RAG (2020)
        "2004.04906,"   # DPR (Karpukhin et al. 2020)
        "2112.09118"    # Large Language Models as Zero-Shot Reasoners (chain-of-thought context)
    )

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Retrieval ---
    top_k_v1: int = 5
    top_k_v2_dense: int = 8
    top_k_v2_rerank: int = 4
    reranker_model: str = "BAAI/bge-reranker-base"
    # Chunks whose cross-encoder score falls below this are dropped entirely.
    # bge-reranker-base logits range ~[-10, +10]; -2.0 cuts obvious noise.
    reranker_score_threshold: float = -2.0

    # --- Multi-AgentLLMrouting ---
    # Empty string → fall back to the provider's default model for that tier.
    # Planner + Researcher use the fast/cheap model; Synthesizer uses the strong one.
    planner_llm_model: str = ""
    synthesizer_llm_model: str = ""

    # --- Agent ---
    agent_max_iterations: int = 6

    # --- Cache ---
    cache_db_path: str = "data/cache.db"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- Convenienceproperties ---
    @property
    def cache_db_full_path(self) -> Path:
        p = Path(self.cache_db_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def arxiv_query_list(self) -> list[str]:
        return [q.strip() for q in self.arxiv_queries.split("|") if q.strip()]

    @property
    def arxiv_seed_id_list(self) -> list[str]:
        return [i.strip() for i in self.arxiv_seed_ids.split(",") if i.strip()]

    def collection_for(self, version: str) -> str:
        if version == "v1":
            return self.qdrant_collection_v1
        if version == "v3":
            return self.qdrant_collection_v3
        return self.qdrant_collection_v2


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Paper nicknames → [aliases] used to enrich Qdrant payloads for acronym queries.
# Keys are bare arXiv IDs (no version suffix). Values are searchable aliases that
# appear in user questions but not necessarily in the paper's own text.
PAPER_ALIASES: dict[str, list[str]] = {
    "2212.10496": ["HyDE", "Hypothetical Document Embeddings", "Precise Zero-Shot Dense Retrieval"],
    "2210.03629": ["ReAct", "Synergizing Reasoning and Acting"],
    "2310.11511": ["Self-RAG", "Self-Reflective Retrieval-Augmented Generation"],
    "2309.15217": ["RAGAS", "Retrieval Augmented Generation Assessment"],
    "2408.08067": ["RAGChecker", "RAG evaluation checker"],
    "2404.16130": ["GraphRAG", "Graph RAG", "Microsoft GraphRAG"],
    "2309.07597": ["BGE", "C-Pack", "BAAI embeddings", "bge-base", "bge-reranker"],
    "2005.11401": ["Lewis RAG", "original RAG", "RAG 2020", "retrieval-augmented generation"],
    "2004.04906": ["DPR", "Dense Passage Retrieval", "Karpukhin"],
    "2112.09118": ["Zero-Shot CoT", "chain-of-thought prompting", "Zero-Shot Reasoners"],
}
