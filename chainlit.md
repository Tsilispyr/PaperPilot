# PaperPilot

**Agentic RAG over recent NLP / LLM / RAG / Agents research papers.**

Ask in plain English. PaperPilot searches a curated corpus of ~100 ArXiv papers from 2023–2026, and falls back to a live ArXiv lookup when your question goes beyond the index.

Use the profile selector (top-left) to switch between **v1** (baseline pipeline) and **v2** (section-aware chunks + cross-encoder reranking + filters) - same agent, different RAG.

Every answer is grounded in retrieved chunks; click any source on the right to see exactly what the agent saw.
