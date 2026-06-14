"""Embedding provider with SQLite cache + bulk-friendly API."""
from __future__ import annotations

import logging
from typing import Sequence

from paperpilot.cache import get_cache
from paperpilot.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Pluggable embedding provider - supports OpenAI today, easy to add Ollama."""

    def __init__(self) -> None:
        self.cache = get_cache()
        self.provider = settings.llm_provider
        if self.provider == "openai":
            self.model = settings.openai_embed_model
        elif self.provider == "google":
            self.model = settings.google_embed_model
        else:
            self.model = settings.ollama_embed_model
        self._client = None  # lazily created

    @property
    def client(self):
        if self._client is None:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=settings.google_api_key)
                self._client = genai
            elif self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
            else:
                # Ollama exposes an OpenAI-compatible /v1 endpoint
                from openai import OpenAI
                self._client = OpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
        return self._client

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
        """Cache-aware bulk embed."""
        out: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, t in enumerate(texts):
            cached = self.cache.get_embedding(self.model, t)
            if cached is not None:
                out[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(t)

        if miss_text:
            logger.info("Embedding %d new texts (%d cached)", len(miss_text), len(texts) - len(miss_text))
            for j in range(0, len(miss_text), batch_size):
                batch = miss_text[j : j + batch_size]
                if self.provider == "google":
                    m_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                    resp = self.client.embed_content(model=m_name, content=batch, task_type="retrieval_document")
                    vecs = resp['embedding']
                else:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    vecs = [d.embedding for d in resp.data]
                self.cache.set_embeddings_bulk(self.model, batch, vecs)
                for k, v in enumerate(vecs):
                    out[miss_idx[j + k]] = v

        # All slots filled at this point.
        return [v for v in out if v is not None]

    def dimension(self) -> int:
        # Probe via a 1-token embed (cheap, cached after first call).
        v = self.embed(["dimension probe"])[0]
        return len(v)
