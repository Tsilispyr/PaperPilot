"""Chainlit chat UI for PaperPilot - Multi-Agent edition.

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

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import time
from typing import Any

import chainlit as cl
from langchain_core.messages import AIMessage

from paperpilot.agent.graph import astream_agent
from paperpilot.config import settings
from paperpilot.observability.haic_logger import HAICLogger, compute_haic_metrics
from paperpilot.observability.langfuse_setup import get_callback_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread persistence - SQLite data layer so chat history survives across sessions
# ---------------------------------------------------------------------------
try:
    import sqlite3 as _sqlite3
    from sqlalchemy import text as _sa_text
    from sqlalchemy.ext.asyncio import create_async_engine as _make_engine
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    # /app/chainlit_db/ is a Docker named volume (Linux FS) so SQLite's
    # POSIX file locking works correctly — unlike the .chainlit/ Windows bind mount.
    _THREAD_DB = "/app/chainlit_db/thread.db"
    pathlib.Path(_THREAD_DB).parent.mkdir(parents=True, exist_ok=True)

    # Chainlit 2.x passes `tags` (a Python list) directly to SQLite without
    # JSON-encoding. Register a global adapter so SQLite transparently
    # serialises list → JSON string for every connection.
    _sqlite3.register_adapter(list, json.dumps)

    # Schema DDL derived from the SQL Chainlit 2.x actually executes.
    # Using asyncio.run() here is safe: module-level code runs before
    # uvicorn starts its event loop.
    _DDL = [
        """CREATE TABLE IF NOT EXISTS users (
            "id"         TEXT PRIMARY KEY NOT NULL,
            "identifier" TEXT NOT NULL,
            "createdAt"  TEXT,
            "metadata"   TEXT NOT NULL
        )""",
        """CREATE UNIQUE INDEX IF NOT EXISTS ix_users_identifier ON users("identifier")""",
        """CREATE TABLE IF NOT EXISTS threads (
            "id"             TEXT PRIMARY KEY NOT NULL,
            "createdAt"      TEXT,
            "name"           TEXT,
            "userId"         TEXT,
            "userIdentifier" TEXT,
            "tags"           TEXT,
            "metadata"       TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS steps (
            "id"            TEXT PRIMARY KEY NOT NULL,
            "name"          TEXT,
            "type"          TEXT,
            "threadId"      TEXT NOT NULL,
            "parentId"      TEXT,
            "streaming"     INTEGER,
            "waitForAnswer" INTEGER,
            "isError"       INTEGER,
            "input"         TEXT,
            "output"        TEXT,
            "createdAt"     TEXT,
            "start"         TEXT,
            "end"           TEXT,
            "defaultOpen"   INTEGER,
            "autoCollapse"  INTEGER,
            "showInput"     TEXT,
            "metadata"      TEXT,
            "generation"    TEXT,
            "tags"          TEXT,
            "language"      TEXT,
            "error"         TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS elements (
            "id"          TEXT PRIMARY KEY NOT NULL,
            "threadId"    TEXT,
            "stepId"      TEXT,
            "type"        TEXT,
            "url"         TEXT,
            "chainlitKey" TEXT,
            "name"        TEXT NOT NULL,
            "display"     TEXT,
            "objectKey"   TEXT,
            "size"        TEXT,
            "language"    TEXT,
            "page"        INTEGER,
            "forId"       TEXT,
            "mime"        TEXT,
            "createdAt"   TEXT,
            "props"       TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS feedbacks (
            "id"       TEXT PRIMARY KEY NOT NULL,
            "forId"    TEXT NOT NULL,
            "value"    REAL NOT NULL,
            "comment"  TEXT,
            "threadId" TEXT
        )""",
    ]

    async def _bootstrap_schema():
        engine = _make_engine(f"sqlite+aiosqlite:///{_THREAD_DB}")
        async with engine.begin() as conn:
            for stmt in _DDL:
                await conn.execute(_sa_text(stmt))
        await engine.dispose()

    asyncio.run(_bootstrap_schema())

    _DATA_LAYER = SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{_THREAD_DB}")

    @cl.data_layer
    def get_data_layer():
        return _DATA_LAYER

except Exception as _dl_err:
    logger.warning("Thread persistence unavailable: %s", _dl_err)

# ---------------------------------------------------------------------------
# User store - persisted to .chainlit/users.json (mounted volume in Docker)
# ---------------------------------------------------------------------------
_USERS_FILE = pathlib.Path(".chainlit") / "users.json"
_PBKDF2_ITERS = 200_000


def _hash(password: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), b"paperpilot-2026", _PBKDF2_ITERS
    ).hex()


def _load_users() -> dict[str, str]:
    try:
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_users(users: dict[str, str]) -> None:
    _USERS_FILE.parent.mkdir(exist_ok=True)
    _USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """
    Three tiers:
      1. Admin  — CHAINLIT_USER / CHAINLIT_PASSWORD env vars (default paperpilot/research2026)
      2. Guest  — username "guest", any password  → shared account, no history persistence
      3. Users  — stored in .chainlit/users.json
                  - unknown username + password >= 6 chars  → auto-register
                  - known username + correct password        → log in
    """
    # 1. Admin
    admin_user = os.environ.get("CHAINLIT_USER", "paperpilot")
    admin_pass = os.environ.get("CHAINLIT_PASSWORD", "research2026")
    if username == admin_user and password == admin_pass:
        return cl.User(identifier=username, metadata={"role": "admin", "provider": "credentials"})

    # 2. Guest — free access, shared identifier (no per-user history)
    if username.lower() == "guest":
        return cl.User(identifier="guest", metadata={"role": "guest", "provider": "credentials"})

    # 3. Registered users
    users = _load_users()
    if username in users:
        if users[username] == _hash(password):
            return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
        return None  # wrong password
    # Auto-register on first login
    if len(username) >= 3 and len(password) >= 6:
        users[username] = _hash(password)
        _save_users(users)
        logger.info("New user registered: %s", username)
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    return None  # username too short or password too short


@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="PaperPilot v3",
            markdown_description=(
                "**v3** - Multi-Agent + IRCoT sub-queries, table-aware chunking, "
                "diversity reranking (max 1 chunk/paper), author alias metadata."
            ),
            default=True,
        ),
        cl.ChatProfile(
            name="PaperPilot v2",
            markdown_description=(
                "**v2** - Multi-Agent (Planner → Researcher → Synthesizer) "
                "with section-aware chunking + cross-encoder rerank + score threshold."
            ),
        ),
        cl.ChatProfile(
            name="PaperPilot v1",
            markdown_description="**v1** - Multi-Agent with fixed-size chunks, no rerank. (Baseline.)",
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
            message="Explain the four core RAGAS metrics - what each measures and how they're computed.",
            icon="/public/icons/ruler.svg",
        ),
    ]




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
            f"**PaperPilot {version}** is ready - Multi-Agent mode active.\n\n"
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


@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict):
    """Restore session state when a user resumes a previous conversation thread."""
    profile = cl.user_session.get("chat_profile") or "PaperPilot v3"
    version = "v1" if "v1" in profile else ("v3" if "v3" in profile else "v2")
    session_id = str(cl.user_session.get("id") or "unknown")

    cl.user_session.set("version", version)
    cl.user_session.set("turn", 0)

    haic = HAICLogger(session_id=session_id, session_num=1)
    cl.user_session.set("haic", haic)

    n_steps = len(thread.get("steps", []))
    await cl.Message(
        content=(
            f"**PaperPilot {version}** - conversation resumed "
            f"({n_steps} previous messages loaded).\n\n"
            f"You can continue asking questions about NLP / LLMs / RAG / agents."
        ),
        author="PaperPilot",
    ).send()




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
    await cl.Message(content="Thanks for your feedback! 👎 - noted.", author="PaperPilot").send()


@cl.action_callback("switch_version")
async def on_switch_version(action: cl.Action):
    new_ver = action.payload.get("version", "v3")
    old_ver = cl.user_session.get("version", "v3")
    if new_ver == old_ver:
        return
    cl.user_session.set("version", new_ver)
    await cl.Message(
        content=(
            f"Switched to **PaperPilot {new_ver}** - chat history kept.\n"
            f"Next question will be answered by {new_ver}."
        ),
        author="PaperPilot",
    ).send()




async def _process_pdf_upload(elem: Any) -> None:
    """Parse, chunk, embed and store an uploaded PDF in the current session."""
    filename = getattr(elem, "name", "uploaded.pdf")
    path = str(getattr(elem, "path", ""))

    status = cl.Message(content=f"Processing `{filename}`...", author="PaperPilot")
    await status.send()
    try:
        import tiktoken
        import pymupdf4llm
        from paperpilot.ingest.embeddings import EmbeddingProvider

        md_text: str = await asyncio.to_thread(pymupdf4llm.to_markdown, path)

        # Simple sliding-window chunking (no schema dependency)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(md_text, disallowed_special=())
        chunk_size, overlap = 512, 50
        step = chunk_size - overlap
        texts: list[str] = []
        for i in range(0, len(tokens), step):
            piece = enc.decode(tokens[i : i + chunk_size]).strip()
            if len(piece) > 60:
                texts.append(piece)

        if not texts:
            status.content = f"`{filename}`: no extractable text found."
            await status.update()
            return

        embeddings: list[list[float]] = await asyncio.to_thread(
            EmbeddingProvider().embed, texts
        )

        new_chunks = [
            {
                "text": text,
                "embedding": emb,
                "title": filename,
                "authors": [],
                "year": "",
                "paper_id": f"upload:{filename}:{i}",
                "section_type": "other",
                "source_file": filename,
                "source": "session_upload",
            }
            for i, (text, emb) in enumerate(zip(texts, embeddings))
        ]

        existing: list[dict] = cl.user_session.get("session_chunks") or []
        existing.extend(new_chunks)
        cl.user_session.set("session_chunks", existing)

        status.content = (
            f"Indexed **{len(texts)} chunks** from `{filename}`. "
            f"Session corpus: **{len(existing)} chunks total**. "
            f"Ask a question to search it alongside the main papers."
        )
        await status.update()
    except Exception as exc:
        status.content = f"Failed to process `{filename}`: {exc}"
        await status.update()
        logger.error("PDF upload failed for %s: %s", filename, exc)


@cl.action_callback("clear_uploads")
async def on_clear_uploads(action: cl.Action):
    cl.user_session.set("session_chunks", [])
    await action.remove()
    await cl.Message(
        content="Uploaded files cleared. Queries now use only the main corpus.",
        author="PaperPilot",
    ).send()


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
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        url = item.get("source_url") or ""
        body = (item.get("text") or item.get("summary", ""))[:1200]

        link_part = f"\n\n[arXiv ↗]({url})" if url else ""
        display = (
            f"**{title}**\n\n"
            f"_{authors} - {year} - {section}_  •  score: `{score_s}`\n\n"
            f"{body}{link_part}"
        )
        elements.append(
            cl.Text(name=f"#{i} {title[:60]}", content=display, display="side")
        )
    return elements




@cl.on_message
async def on_message(message: cl.Message):
    version: str = cl.user_session.get("version") or "v3"
    session_id: str = str(cl.user_session.get("id"))
    haic: HAICLogger | None = cl.user_session.get("haic")

    # --- Handle uploaded PDFs ---
    if message.elements:
        pdfs = [
            e for e in message.elements
            if getattr(e, "mime", "") == "application/pdf"
            or str(getattr(e, "name", "")).lower().endswith(".pdf")
        ]
        for pdf_elem in pdfs:
            await _process_pdf_upload(pdf_elem)
        if pdfs and not message.content.strip():
            return  # upload-only message, no question to answer

    session_chunks: list[dict] = cl.user_session.get("session_chunks") or []

    # HAIC: log query
    if haic:
        haic.log_query(duration_s=5.0)

    t0 = time.time()

    # --- Live status message (updated as nodes fire) ---
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
        user_id=getattr(cl.user_session.get("user"), "identifier", None),
        session_chunks=session_chunks or None,
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
            session_hits = sum(1 for c in raw_chunks if c.get("source") == "session_upload")
            corpus_hits = len(raw_chunks) - session_hits
            sources_desc = f"{corpus_hits} corpus chunks"
            if session_hits:
                sources_desc += f" + {session_hits} from uploaded files"
            status.content = (
                f"**Planner** → {plan_info}  \n"
                f"**Researcher** → retrieved {sources_desc}  \n\n"
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

    # --- Replace status with final answer + sources ---
    elapsed_s = total_latency_ms / 1000
    retrieval_s = tool_latency_ms / 1000
    timing = (
        f"*{version} · {elapsed_s:.1f}s total"
        + (f" (retrieval {retrieval_s:.1f}s)" if tool_calls else "")
        + "*"
    )
    status.content = (final_answer or "(no answer generated)") + f"\n\n---\n{timing}"
    status.elements = _format_sources(raw_chunks)
    actions = [
        cl.Action(name="haic_accept", label="👍 Helpful", value="accept", payload={"value": "accept"}),
        cl.Action(name="haic_reject", label="👎 Not helpful", value="reject", payload={"value": "reject"}),
        *[
            cl.Action(
                name="switch_version",
                label=f"[{v}]" if v == version else v,
                value=v,
                payload={"version": v},
            )
            for v in ("v1", "v2", "v3")
        ],
    ]
    if session_chunks:
        actions.append(
            cl.Action(
                name="clear_uploads",
                label=f"Clear uploads ({len(session_chunks)} chunks)",
                value="clear",
                payload={},
            )
        )
    status.actions = actions
    await status.update()
