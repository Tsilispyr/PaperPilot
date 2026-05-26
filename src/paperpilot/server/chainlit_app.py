"""Chainlit chat UI for PaperPilot — Multi-Agent edition.

Run: chainlit run src/paperpilot/server/chainlit_app.py --port 8000

Phase 4 changes:
  - Streams node-level state transitions so users see progress immediately:
      "Planner is thinking..." → "Researcher is querying Qdrant in parallel..."
      → "Synthesizer is writing the answer..."
  - Sources are rendered from raw_chunks preserved by the Researcher node.
  - 👍 / 👎 HAIC buttons preserved on every response.
  - Backward-compatible: same HAIC logging, Langfuse callbacks, profile selector.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import chainlit as cl
from langchain_core.messages import AIMessage

from paperpilot.agent.graph import astream_agent
from paperpilot.config import settings
from paperpilot.observability.haic_logger import HAICLogger, compute_haic_metrics
from paperpilot.observability.langfuse_setup import get_callback_handler

logger = logging.getLogger(__name__)



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    

@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="PaperPilot v3",
            markdown_description=(
                "**v3** — Multi-Agent + IRCoT sub-queries, table-aware chunking, "
                "diversity reranking (max 1 chunk/paper), author alias metadata."
            ),
            default=True,
        ),
        cl.ChatProfile(
            name="PaperPilot v2",
            markdown_description=(
                "**v2** — Multi-Agent (Planner → Researcher → Synthesizer) "
                "with section-aware chunking + cross-encoder rerank + score threshold."
            ),
        ),
        cl.ChatProfile(
            name="PaperPilot v1",
            markdown_description="**v1** — Multi-Agent with fixed-size chunks, no rerank. (Baseline.)",
        ),
    ]


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="What is HyDE?",
            message="What is Hypothetical Document Embeddings (HyDE) and how does it differ from standard dense retrieval?",
            icon="/public/icons/sparkles.svg",
        ),
        cl.Starter(
            label="ReAct vs CoT",
            message="How does the ReAct framework differ from Chain-of-Thought, and what problems does it address?",
            icon="/public/icons/branch.svg",
        ),
        cl.Starter(
            label="Self-RAG: who proposed it?",
            message="Which paper proposed Self-RAG and what are its key innovations over plain RAG?",
            icon="/public/icons/quote.svg",
        ),
        cl.Starter(
            label="RAGAS metrics breakdown",
            message="Explain the four core RAGAS metrics — what each measures and how they're computed.",
            icon="/public/icons/ruler.svg",
        ),
    ]



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    

@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile") or "PaperPilot v3"
    version = "v1" if "v1" in profile else ("v3" if "v3" in profile else "v2")
    session_id = str(cl.user_session.get("id") or "unknown")

    cl.user_session.set("version", version)
    cl.user_session.set("turn", 0)

    haic = HAICLogger(session_id=session_id, session_num=1)
    cl.user_session.set("haic", haic)

    await cl.Message(
        content=(
            f"**PaperPilot {version}** is ready — Multi-Agent mode active.\n\n"
            f"Ask me anything about NLP / LLMs / RAG / agents. "
            f"I'll plan, retrieve, and synthesize from ~100 indexed papers.\n\n"
            f"_Use 👍 / 👎 on each answer to help evaluate the system._"
        ),
        author="PaperPilot",
    ).send()


@cl.on_chat_end
async def on_chat_end():
    haic: HAICLogger | None = cl.user_session.get("haic")
    if haic and haic.decisions:
        try:
            art = haic.build_artifact()
            metrics = compute_haic_metrics(art)
            artifact_path = haic.save_artifact()
            logger.info(
                "HAIC session %s ended: Tr=%.3f EL=%.3f HCL=%.3f → %s",
                haic.session_id,
                metrics.get("Tr", 0),
                metrics.get("EL", 0),
                metrics.get("HCL", 0),
                artifact_path,
            )
        except Exception as exc:
            logger.warning("HAIC session save failed: %s", exc)



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    

@cl.action_callback("haic_accept")
async def on_accept(action: cl.Action):
    haic: HAICLogger | None = cl.user_session.get("haic")
    if haic:
        haic.log_accept()
    await action.remove()
    await cl.Message(content="Thanks for your feedback! 👍", author="PaperPilot").send()


@cl.action_callback("haic_reject")
async def on_reject(action: cl.Action):
    haic: HAICLogger | None = cl.user_session.get("haic")
    if haic:
        haic.log_reject()
    await action.remove()
    await cl.Message(content="Thanks for your feedback! 👎 — noted.", author="PaperPilot").send()



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    

def _format_sources(raw_chunks: list[dict]) -> list[cl.Element]:
    """Build collapsible side-panel source cards from raw_chunks preserved by Researcher."""
    elements: list[cl.Element] = []
    seen: set[str] = set()
    for i, item in enumerate(raw_chunks, 1):
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id", str(i))
        if paper_id in seen:
            continue
        seen.add(paper_id)

        title = item.get("title", "Source")
        authors = ", ".join((item.get("authors") or [])[:2])
        year = item.get("year", "")
        section = item.get("section_title") or item.get("section_type", "")
        score = item.get("rerank_score") or item.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        url = item.get("source_url") or ""
        body = (item.get("text") or item.get("summary", ""))[:1200]

        link_part = f"\n\n[arXiv ↗]({url})" if url else ""
        display = (
            f"**{title}**\n\n"
            f"_{authors} — {year} — {section}_  •  score: `{score_s}`\n\n"
            f"{body}{link_part}"
        )
        elements.append(
            cl.Text(name=f"#{i} {title[:60]}", content=display, display="side")
        )
    return elements



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    

@cl.on_message
async def on_message(message: cl.Message):
    version: str = cl.user_session.get("version") or "v2"
    session_id: str = str(cl.user_session.get("id"))
    haic: HAICLogger | None = cl.user_session.get("haic")

    # HAIC: log query
    if haic:
        haic.log_query(duration_s=5.0)

    t0 = time.time()

    # --- Livestatusmessage(updatedasnodesfire) ---
    status = cl.Message(content="**Planner** is thinking...", author="PaperPilot")
    await status.send()

    # --- Streamnode-levelupdates ---
    final_answer = ""
    raw_chunks: list[dict] = []
    tool_calls: list[dict] = []
    plan_info = ""
    tool_latency_ms = 0.0

    async for update in astream_agent(
        message.content,
        version,
        session_id=session_id,
        user_id=(
            cl.user_session.get("user", {}).get("identifier")
            if cl.user_session.get("user")
            else None
        ),
    ):
        # update is {"node_name": {updated_state_fields}}
        if "planner" in update:
            plan = update["planner"].get("plan") or {}
            mode = plan.get("mode", "?")
            rationale = plan.get("rationale", "")
            plan_info = f"mode=`{mode}`"
            needs = []
            if plan.get("needs_rag"):
                needs.append("Qdrant corpus")
            if plan.get("needs_arxiv"):
                needs.append("live ArXiv")
            needs_str = " + ".join(needs) if needs else "direct answer"
            status.content = (
                f"**Planner** → {plan_info}  \n"
                f"_{rationale}_  \n\n"
                f"**Researcher** is querying {needs_str} in parallel..."
            )
            await status.update()

        elif "researcher" in update:
            res = update["researcher"]
            raw_chunks = res.get("raw_chunks") or []
            tool_calls = res.get("tool_calls") or []
            tool_latency_ms = (time.time() - t0) * 1000
            status.content = (
                f"**Planner** → {plan_info}  \n"
                f"**Researcher** → retrieved {len(raw_chunks)} chunks  \n\n"
                f"**Synthesizer** is writing the answer..."
            )
            await status.update()

        elif "synthesizer" in update:
            syn = update["synthesizer"]
            final_answer = syn.get("answer", "")

    total_latency_ms = (time.time() - t0) * 1000

    # HAIC: log retrieve + respond
    if haic:
        if tool_calls:
            haic.log_retrieve(latency_ms=tool_latency_ms)
        haic.log_respond(latency_ms=total_latency_ms)

    # --- Replacestatuswithfinalanswer+sources ---
    status.content = final_answer or "(no answer generated)"
    status.elements = _format_sources(raw_chunks)
    status.actions = [
        cl.Action(
            name="haic_accept",
            label="👍 Helpful",
            value="accept",
            payload={"value": "accept"},
        ),
        cl.Action(
            name="haic_reject",
            label="👎 Not helpful",
            value="reject",
            payload={"value": "reject"},
        ),
    ]
    await status.update()
