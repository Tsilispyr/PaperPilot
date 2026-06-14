"""LangGraph Multi-Agent Orchestrator - Planner → Researcher → Synthesizer.

Architecture (Phase 2):
  +==========+     +============+     +=============+
  |  Planner |====▶| Researcher |====▶| Synthesizer |
  | (fast LM)|     | (fast LM)  |     | (strong LM) |
  +==========+     +============+     +=============+
       |                                      ▲
       | out_of_context                       |
       +======================================+

Planner:     Classifies intent; emits a structured JSON search plan.
Researcher:  Runs rag_retrieve + arxiv_search concurrently (asyncio.gather).
             Distills raw chunks → concise bullet-point facts to save tokens.
Synthesizer: Writes the final Markdown answer from bullet facts only.

Model routing (Phase 3):
  Planner + Researcher → settings.planner_llm_model   (cheap/fast)
  Synthesizer          → settings.synthesizer_llm_model (powerful)

Backward compatibility (Phase 4):
  run_agent()    → {"answer": str, "messages": list, "tool_calls": list}
  stream_agent() → Iterable[state dict]  (legacy shim, sync)
  arun_agent()   → same dict, awaitable
  astream_agent()→ AsyncIterator[{"node_name": update}]
"""
from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any, AsyncIterator, Iterable, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from paperpilot.agent.tools import arxiv_search_impl, rag_retrieve_impl
from paperpilot.config import settings
from paperpilot.observability.langfuse_setup import get_callback_handler

logger = logging.getLogger(__name__)




class AgentState(TypedDict):
    user_query: str
    plan: dict                                          # planner output
    research_notes: str                                 # distilled bullets → Synthesizer
    raw_chunks: list[dict]                              # full chunk dicts → Chainlit UI sources
    tool_calls: Annotated[list[dict], operator.add]     # accumulated for eval scripts
    messages: Annotated[list, add_messages]             # LangChain message history
    answer: str                                         # final output
    session_chunks: list[dict]                          # user-uploaded PDF chunks (with embeddings)




class SearchPlan(BaseModel):
    mode: Literal["quick_qa", "deep_analysis", "out_of_context"] = Field(
        description=(
            "quick_qa: simple factual lookup answerable from one paper; "
            "deep_analysis: multi-paper synthesis or comparison question; "
            "out_of_context: question unrelated to NLP/LLM/RAG/Agents research."
        )
    )
    needs_rag: bool = Field(True, description="Search the local Qdrant corpus.")
    needs_arxiv: bool = Field(
        False,
        description="Search live ArXiv - use only for papers published in the last 3 months.",
    )
    rag_query: str = Field("", description="Primary keyword-dense query for vector search.")
    sub_queries: list[str] = Field(
        default_factory=list,
        description=(
            "For deep_analysis only: 2-3 focused sub-queries that together cover "
            "all aspects of the question. Each sub-query targets a different facet "
            "(e.g. one per method being compared). Empty for quick_qa."
        ),
    )
    arxiv_query: str = Field("", description="ArXiv fielded query (only when needs_arxiv=true).")
    rationale: str = Field("", description="One sentence explaining the mode chosen.")




def _make_llm(tier: Literal["fast", "strong"]) -> Any:
    """
    fast   → Planner + Researcher  (cheap, low-latency)
    strong → Synthesizer           (high capability)

    Override per tier via PLANNER_LLM_MODEL / SYNTHESIZER_LLM_MODEL in .env.
    Falls back to the provider's default model when the override is empty.
    """
    provider = settings.llm_provider

    # Tier-specific model overrides (empty string = use provider default)
    fast_override = settings.planner_llm_model.strip()
    strong_override = settings.synthesizer_llm_model.strip()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        fast_model = fast_override or "gemini-2.0-flash"
        strong_model = strong_override or settings.google_llm_model
        return ChatGoogleGenerativeAI(
            model=fast_model if tier == "fast" else strong_model,
            api_key=settings.google_api_key,
            temperature=0.0 if tier == "fast" else 0.1,
            timeout=40 if tier == "fast" else 90,
        )

    if provider == "openai":
        fast_model = fast_override or "gpt-4o-mini"
        strong_model = strong_override or settings.openai_llm_model
        return ChatOpenAI(
            model=fast_model if tier == "fast" else strong_model,
            api_key=settings.openai_api_key,
            temperature=0.0 if tier == "fast" else 0.1,
            timeout=40 if tier == "fast" else 90,
        )

    # Ollama via OpenAI-compatible endpoint
    fast_model = fast_override or settings.ollama_llm_model
    strong_model = strong_override or settings.ollama_llm_model
    return ChatOpenAI(
        model=fast_model if tier == "fast" else strong_model,
        api_key="ollama",
        base_url=f"{settings.ollama_base_url}/v1",
        temperature=0.0 if tier == "fast" else 0.1,
        timeout=80 if tier == "fast" else 180,
    )




_GRAPH_CACHE: dict[str, Any] = {}




_PLANNER_SYSTEM = """\
You are the Planner for PaperPilot - an academic assistant covering NLP, LLM, \
RAG, and AI Agents research papers (corpus: 2020-2026).

Given the user question, output a JSON search plan with these fields:
  mode        - "quick_qa" | "deep_analysis" | "out_of_context"
  needs_rag   - true to search the local vector corpus (default: true)
  needs_arxiv - true ONLY for papers published in the last 3 months (default: false)
  rag_query   - a short, keyword-dense query optimized for dense vector search
  sub_queries - for deep_analysis ONLY: 2-3 focused sub-queries, one per facet/paper
  arxiv_query - a fielded ArXiv query (only when needs_arxiv=true)
  rationale   - one sentence explaining your choice

Rules:
- Default: needs_rag=true, needs_arxiv=false.
- For out_of_context, set both to false and leave queries empty.
- Keep rag_query short and information-dense (no filler words like "tell me about").
- For deep_analysis, generate 2-3 targeted sub_queries in addition to rag_query.
  Each sub_query should target one specific concept, method, or paper being compared.
  Example: "How does Self-RAG differ from HyDE?" →
    rag_query="Self-RAG HyDE retrieval comparison"
    sub_queries=["Self-RAG selective retrieval token generation", "HyDE hypothetical document embeddings zero-shot"]
- For quick_qa, leave sub_queries empty.
"""


async def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = _make_llm("fast").with_structured_output(SearchPlan)
    result: SearchPlan = await llm.ainvoke(
        [SystemMessage(content=_PLANNER_SYSTEM), HumanMessage(content=state["user_query"])],
        config=config,
    )
    logger.info(
        "Planner → mode=%s needs_rag=%s needs_arxiv=%s | %s",
        result.mode, result.needs_rag, result.needs_arxiv, result.rationale,
    )
    return {
        "plan": result.model_dump(),
        "messages": [AIMessage(content=f"[Planner] mode={result.mode} - {result.rationale}")],
    }




_DISTILL_SYSTEM = """\
You are a research distiller. Given raw retrieved paper chunks, produce a \
concise bullet-point summary (max 12 bullets, ≤ 25 words each).

Rules:
- Include the paper title and year after each fact: (Title, Year)
- Drop any chunk not directly relevant to the question
- Output ONLY the bullet list - no preamble, no conclusion, no markdown headers
"""


async def researcher_node(state: AgentState, config: RunnableConfig) -> dict:
    plan = state["plan"]
    user_query = state["user_query"]
    version = (config.get("configurable") or {}).get("version", "v2")
    s_chunks = state.get("session_chunks") or []

    tool_calls_log: list[dict] = []
    all_chunks: list[dict] = []

    # --- Concurrent retrieval via asyncio.gather ---
    async def _run_rag() -> list[dict]:
        sub_qs = plan.get("sub_queries") or []
        primary = plan.get("rag_query") or user_query
        queries = sub_qs if sub_qs else [primary]

        gathered: list[dict] = []
        for q in queries:
            result = await asyncio.to_thread(rag_retrieve_impl, q, version=version)
            tool_calls_log.append({"name": "rag_retrieve", "args": {"query": q, "version": version}})
            gathered.extend(result)

        # Deduplicate by first 80 chars of chunk text (preserves order)
        seen: set[str] = set()
        unique: list[dict] = []
        for c in gathered:
            key = (c.get("text") or c.get("summary", ""))[:80]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    async def _run_arxiv() -> list[dict]:
        q = plan.get("arxiv_query") or user_query
        papers = await asyncio.to_thread(arxiv_search_impl, q)
        tool_calls_log.append({"name": "arxiv_search", "args": {"query": q}})
        return papers

    async def _run_session() -> list[dict]:
        """Cosine similarity search over user-uploaded PDF chunks."""
        if not s_chunks:
            return []

        def _search() -> list[dict]:
            import numpy as np
            from paperpilot.ingest.embeddings import EmbeddingProvider
            query_text = plan.get("rag_query") or user_query
            q_vec = np.array(EmbeddingProvider().embed([query_text])[0], dtype=np.float32)
            scored = []
            for chunk in s_chunks:
                emb = np.array(chunk["embedding"], dtype=np.float32)
                norm = np.linalg.norm(q_vec) * np.linalg.norm(emb)
                score = float(np.dot(q_vec, emb) / norm) if norm > 0 else 0.0
                scored.append((score, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, c in scored[:4]:
                r = {k: v for k, v in c.items() if k != "embedding"}
                r["score"] = score
                results.append(r)
            return results

        results = await asyncio.to_thread(_search)
        tool_calls_log.append({"name": "session_search", "args": {"n_chunks": len(s_chunks)}})
        return results

    tasks: list[asyncio.Task] = []
    if plan.get("needs_rag", True):
        tasks.append(asyncio.ensure_future(_run_rag()))
    if plan.get("needs_arxiv", False):
        tasks.append(asyncio.ensure_future(_run_arxiv()))
    if s_chunks:
        tasks.append(asyncio.ensure_future(_run_session()))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Researcher sub-task failed: %s", r)
            elif isinstance(r, list):
                all_chunks.extend(r)

    if not all_chunks:
        logger.info("Researcher → no chunks retrieved")
        return {
            "research_notes": "No relevant papers found in the corpus or ArXiv.\n",
            "raw_chunks": [],
            "tool_calls": tool_calls_log,
            "messages": [AIMessage(content="[Researcher] No results found.")],
        }

    # Format chunks for the distillation LLM call
    formatted = "\n\n".join(
        f"[{i + 1}] {c.get('title', 'Unknown')} ({c.get('year', '?')}): "
        f"{c.get('text') or c.get('summary', '')[:800]}"
        for i, c in enumerate(all_chunks)
    )

    llm = _make_llm("fast")
    distilled = await llm.ainvoke(
        [
            SystemMessage(content=_DISTILL_SYSTEM),
            HumanMessage(content=f"Question: {user_query}\n\nChunks:\n{formatted}"),
        ],
        config=config,
    )
    notes = distilled.content if isinstance(distilled.content, str) else str(distilled.content)

    logger.info(
        "Researcher → %d chunks → %d chars distilled", len(all_chunks), len(notes)
    )
    return {
        "research_notes": notes,
        "raw_chunks": all_chunks,
        "tool_calls": tool_calls_log,
        "messages": [
            AIMessage(
                content=f"[Researcher] Retrieved {len(all_chunks)} chunks from "
                        f"{', '.join(t['name'] for t in tool_calls_log)}. Distilled to bullet facts."
            )
        ],
    }




_SYNTHESIZER_SYSTEM = """\
You are PaperPilot's Synthesizer. Write a comprehensive, well-structured Markdown response.

Rules:
- Answer directly and thoroughly using ONLY the provided research notes.
- Use ## and ### headers for multi-part answers.
- Cite papers inline as (Authors, Year) whenever a fact comes from the notes.
- End every response with a "## Sources" section listing all cited papers.
- If the notes say "No relevant papers found", acknowledge it and suggest a refined query.
- NEVER invent facts. If uncertain, state the limitation clearly.
"""

_OUT_OF_CONTEXT_MSG = (
    "I'm PaperPilot, specialized in NLP, LLM, RAG, and AI Agents research papers "
    "(corpus: 2020-2026). Your question appears to be outside that scope.\n\n"
    "Try asking about topics like retrieval-augmented generation, transformer architectures, "
    "evaluation frameworks (RAGAS), or specific papers in the indexed corpus."
)


async def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    mode = (state.get("plan") or {}).get("mode", "quick_qa")
    has_notes = bool((state.get("research_notes") or "").strip())

    if mode == "out_of_context" and not has_notes:
        return {
            "answer": _OUT_OF_CONTEXT_MSG,
            "messages": [AIMessage(content=_OUT_OF_CONTEXT_MSG)],
        }

    user_content = (
        f"**User Question:** {state['user_query']}\n\n"
        f"**Research Notes (distilled facts):**\n"
        f"{state.get('research_notes') or 'No research notes available.'}"
    )

    llm = _make_llm("strong")
    response = await llm.ainvoke(
        [SystemMessage(content=_SYNTHESIZER_SYSTEM), HumanMessage(content=user_content)],
        config=config,
    )
    answer = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("Synthesizer → %d chars", len(answer))
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }




def _route_after_planner(state: AgentState) -> Literal["researcher", "synthesizer"]:
    # If user has uploaded files, always search them even for OOC questions
    if (state.get("plan") or {}).get("mode") == "out_of_context" and not (state.get("session_chunks") or []):
        return "synthesizer"
    return "researcher"




def build_graph(version: str = "v2"):
    """Compile (and memoize) the multi-agent StateGraph."""
    cache_key = f"multi_{version}"
    if cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]

    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"researcher": "researcher", "synthesizer": "synthesizer"},
    )
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)

    graph = g.compile()
    _GRAPH_CACHE[cache_key] = graph
    return graph




def _initial_state(question: str, session_chunks: Optional[list[dict]] = None) -> dict:
    return {
        "user_query": question,
        "plan": {},
        "research_notes": "",
        "raw_chunks": [],
        "tool_calls": [],
        "messages": [],
        "answer": "",
        "session_chunks": session_chunks or [],
    }


def _build_config(
    version: str,
    session_id: Optional[str],
    user_id: Optional[str],
) -> tuple[dict, list]:
    callbacks = []
    cb = get_callback_handler(
        session_id=session_id,
        user_id=user_id,
        tags=[f"version:{version}", "multi-agent"],
    )
    if cb:
        callbacks.append(cb)
    return {
        "callbacks": callbacks,
        "recursion_limit": 10,
        "configurable": {"thread_id": session_id or "single", "version": version},
    }, callbacks


async def _ainvoke(
    question: str,
    version: str = "v2",
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    graph = build_graph(version)
    cfg, _ = _build_config(version, session_id, user_id)
    state = await graph.ainvoke(_initial_state(question), config=cfg)
    return {
        "answer": state.get("answer", ""),
        "messages": state.get("messages", []),
        "tool_calls": state.get("tool_calls", []),
    }




def run_agent(
    question: str,
    version: str = "v2",
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Sync entrypoint for CLI / eval scripts.

    Safe to call from both sync contexts (scripts) and existing async loops
    (e.g. RAGAS eval which internally uses asyncio) - uses a thread when needed.
    Returns {"answer": str, "messages": list, "tool_calls": list}.
    """
    coro = _ainvoke(question, version, session_id=session_id, user_id=user_id)
    try:
        asyncio.get_running_loop()
        # Already inside an event loop - run in a fresh thread to avoid deadlock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


async def arun_agent(
    question: str,
    version: str = "v2",
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Async entrypoint for Chainlit / any async caller."""
    return await _ainvoke(question, version, session_id=session_id, user_id=user_id)


async def astream_agent(
    question: str,
    version: str = "v2",
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_chunks: Optional[list[dict]] = None,
) -> AsyncIterator[dict]:
    """Async generator yielding per-node state updates.

    Each item is {"node_name": {updated_fields}} so callers can react
    to planner / researcher / synthesizer transitions individually.
    """
    graph = build_graph(version)
    cfg, _ = _build_config(version, session_id, user_id)
    async for update in graph.astream(
        _initial_state(question, session_chunks),
        config=cfg,
        stream_mode="updates",
    ):
        yield update


def stream_agent(
    question: str,
    version: str = "v2",
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Iterable[dict]:
    """Legacy sync streaming shim - prefer astream_agent() for new code.

    Bridges the async graph into a synchronous generator by running the
    async producer in a background thread with its own event loop.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue()

    async def _producer() -> None:
        try:
            async for update in astream_agent(
                question, version, session_id=session_id, user_id=user_id
            ):
                q.put(update)
        finally:
            q.put(None)

    t = threading.Thread(target=lambda: asyncio.run(_producer()), daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
