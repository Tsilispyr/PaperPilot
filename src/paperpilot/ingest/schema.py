"""Shared data models for ingestion + retrieval."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SectionType = Literal[
    "abstract", "introduction", "background", "method", "experiments",
    "results", "discussion", "conclusion", "references", "appendix", "other",
]


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    summary: str
    published: str  # ISO date
    primary_category: str
    categories: list[str]
    entry_id: str
    pdf_url: str
    query: str = ""

    @property
    def year(self) -> int:
        return int(self.published[:4])


class Chunk(BaseModel):
    """One unit indexed in Qdrant.

    Note: `id_` is a deterministic UUID-like string — used as the Qdrant point id.
    """
    id_: str
    paper_id: str
    title: str
    authors: list[str]
    year: int
    primary_category: str
    section_type: SectionType = "other"
    section_title: Optional[str] = None
    chunk_index: int = 0
    text: str
    pdf_url: str
    source_url: str = Field(default="", description="Direct link to paper (arxiv abs page)")
    aliases: list[str] = Field(default_factory=list, description="Searchable paper nicknames (HyDE, ReAct, …)")

    def to_qdrant_payload(self) -> dict:
        # Keep payload flat & filter-friendly.
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "primary_category": self.primary_category,
            "section_type": self.section_type,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "pdf_url": self.pdf_url,
            "source_url": self.source_url,
            "aliases": self.aliases,
        }
