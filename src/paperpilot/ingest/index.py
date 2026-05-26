"""Glue: parsed papers → chunks → embeddings → Qdrant collection."""
from __future__ import annotations

import logging
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import re

from paperpilot.config import PAPER_ALIASES, settings
from paperpilot.ingest.chunk import get_chunker
from paperpilot.ingest.embeddings import EmbeddingProvider
from paperpilot.ingest.parse import load_processed
from paperpilot.ingest.schema import Chunk

logger = logging.getLogger(__name__)


def _qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=300)


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        logger.info("Recreating collection %s", name)
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        # Helpful payload indexes for metadata filtering (v2).
        for field, schema in (
            ("paper_id", qm.PayloadSchemaType.KEYWORD),
            ("year", qm.PayloadSchemaType.INTEGER),
            ("primary_category", qm.PayloadSchemaType.KEYWORD),
            ("section_type", qm.PayloadSchemaType.KEYWORD),
        ):
            try:
                client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
            except Exception:  # idempotent
                pass


def _upsert(client: QdrantClient, name: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points = [
        qm.PointStruct(id=c.id_, vector=v, payload=c.to_qdrant_payload())
        for c, v in zip(chunks, vectors)
    ]
    BATCH = 128
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=name, points=points[i : i + BATCH])


def index_corpus(version: Literal["v1", "v2"], recreate: bool = False) -> int:
    """Run the full ingest→index pipeline for one version. Returns #chunks indexed."""
    chunker = get_chunker(version)
    embedder = EmbeddingProvider()
    client = _qdrant()
    collection = settings.collection_for(version)

    # Build chunks
    papers = load_processed()
    if not papers:
        logger.warning("No parsed papers found. Run `make parse` first.")
        return 0
    all_chunks: list[Chunk] = []
    for meta, text in papers:
        try:
            all_chunks.extend(chunker.chunk(meta, text))
        except Exception as exc:
            logger.warning("Chunking failed for %s: %s", meta.paper_id, exc)
    if not all_chunks:
        logger.warning("No chunks produced.")
        return 0

    # Enrich chunks with paper aliases (for acronym-based retrieval in v3).
    for chunk in all_chunks:
        base_id = re.sub(r"v\d+$", "", chunk.paper_id)
        chunk.aliases = PAPER_ALIASES.get(base_id, [])

    logger.info("Built %d chunks (version=%s) from %d papers", len(all_chunks), version, len(papers))

    # Embed (cache-aware)
    vectors = embedder.embed([c.text for c in all_chunks])
    dim = len(vectors[0])

    # Upsert
    ensure_collection(client, collection, dim=dim, recreate=recreate)
    _upsert(client, collection, all_chunks, vectors)
    logger.info("Upserted %d vectors into %s", len(all_chunks), collection)
    return len(all_chunks)
