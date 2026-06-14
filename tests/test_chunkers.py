"""Smoke tests for the two chunkers - no network, no LLM."""
from __future__ import annotations

from paperpilot.ingest.chunk import FixedSizeChunker, SectionAwareChunker
from paperpilot.ingest.schema import PaperMetadata


def _meta() -> PaperMetadata:
    return PaperMetadata(
        paper_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice Smith", "Bob Jones"],
        summary="A test.",
        published="2024-01-15T00:00:00+00:00",
        primary_category="cs.CL",
        categories=["cs.CL"],
        entry_id="http://arxiv.org/abs/2401.00001v1",
        pdf_url="http://arxiv.org/pdf/2401.00001v1",
    )


SAMPLE_MD = """\
# Title

## Abstract
This paper explores retrieval-augmented generation for academic QA. We propose a method that improves precision.

## 1 Introduction
RAG has become a dominant paradigm. We focus on academic corpora.

## 2 Method
Our approach combines section-aware chunking with cross-encoder reranking. The pipeline has three stages.

## 3 Experiments
We evaluate on a held-out golden set of 30 questions.

## 4 Results
Context Precision improved from 0.62 to 0.81.

## 5 Conclusion
Section-aware chunking helps.

## References
[1] Some other paper.
"""


def test_fixed_size_chunker_produces_nonempty():
    chunks = FixedSizeChunker(size_tokens=128, overlap_tokens=16).chunk(_meta(), SAMPLE_MD)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text
        assert c.paper_id == "2401.00001v1"
        assert c.section_type == "other"


def test_section_aware_chunker_classifies_sections():
    chunks = SectionAwareChunker(max_tokens=256, overlap_tokens=20, min_tokens=10).chunk(_meta(), SAMPLE_MD)
    types = {c.section_type for c in chunks}
    assert "abstract" in types
    assert "method" in types
    assert "results" in types
    # references must be skipped
    assert "references" not in types


def test_section_aware_chunker_keeps_titles():
    chunks = SectionAwareChunker(max_tokens=256, overlap_tokens=20, min_tokens=10).chunk(_meta(), SAMPLE_MD)
    titles = {c.section_title for c in chunks if c.section_title}
    assert any("Abstract" in t for t in titles)
