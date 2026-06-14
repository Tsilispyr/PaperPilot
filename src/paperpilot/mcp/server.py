"""MCP server exposing PaperPilot's `rag_retrieve` as an MCP tool. (+5 bonus.)

Run as a stdio MCP server (the standard way to expose tools to LLM clients):
    python -m paperpilot.mcp.server

Any MCP-compliant client (Claude Desktop, MCP Inspector, etc.) can then connect and
call `rag_retrieve` against the PaperPilot corpus.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from paperpilot.agent.tools import rag_retrieve_impl as _rag_retrieve_impl

logger = logging.getLogger(__name__)

mcp = FastMCP("paperpilot")


@mcp.tool()
def rag_retrieve(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    primary_category: Optional[str] = None,
    top_k: int = 4,
    version: str = "v2",
) -> str:
    """Semantic search over PaperPilot's indexed corpus of NLP/LLM/RAG/Agents papers (~2023-2026).

    Returns top-k chunks (JSON list) with title, authors, year, section, text, source URL,
    and relevance score. Optional filters: year_from, year_to, primary_category, version (v1|v2).
    """
    chunks = _rag_retrieve_impl(
        query=query,
        year_from=year_from,
        year_to=year_to,
        primary_category=primary_category,
        top_k=top_k,
        version=version,
    )
    return json.dumps(chunks, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting PaperPilot MCP server on stdio …")
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
