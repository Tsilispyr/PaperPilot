"""Langfuse client + LangChain callback handler.

If credentials are missing we degrade gracefully: callers get None and tracing is skipped,
so the system stays runnable without Langfuse (e.g. on a fresh install before the user
has logged into the local Langfuse and copied API keys into .env).
"""
from __future__ import annotations

import logging
import sys
import types
from functools import lru_cache
from typing import Optional

from paperpilot.config import settings

logger = logging.getLogger(__name__)


def _patch_langchain_compat() -> None:
    """Inject shim modules so langfuse v2 (written for langchain 0.x) works with langchain 1.x.

    langchain 1.x moved everything to langchain_core. langfuse v2 still imports from
    the old paths (langchain.callbacks.base, langchain.schema.{agent,document}).
    """
    try:
        import langchain_core.callbacks.base as _cb
        import langchain_core.agents as _agents
        import langchain_core.documents as _docs
    except ImportError:
        return  # nothing to patch

    def _shim(name: str, **attrs) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    _shim("langchain.callbacks", base=_shim("langchain.callbacks.base",
          BaseCallbackHandler=_cb.BaseCallbackHandler))
    _shim("langchain.schema")
    _shim("langchain.schema.agent",
          AgentAction=_agents.AgentAction,
          AgentFinish=_agents.AgentFinish)
    _shim("langchain.schema.document", Document=_docs.Document)


@lru_cache
def get_langfuse_client():
    if "langfuse-web" in settings.langfuse_host:
        logger.warning(
            "LANGFUSE_HOST=%s is a Docker-internal address — unreachable from the host. "
            "Set LANGFUSE_HOST=http://localhost:3001 in .env for local pipeline runs.",
            settings.langfuse_host,
        )
        return None

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.warning(
            "Langfuse keys not set — tracing disabled. "
            "Visit %s to log in and create keys, then set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env",
            settings.langfuse_host,
        )
        return None
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # Verify credentials before returning — prevents 500-spam on wrong/placeholder keys.
        try:
            if not client.auth_check():
                logger.warning(
                    "Langfuse auth check failed — tracing disabled. "
                    "Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env "
                    "(log in at %s, create an API key, paste it in .env)",
                    settings.langfuse_host,
                )
                return None
        except Exception as auth_exc:
            logger.warning("Langfuse auth check error (%s) — tracing disabled.", auth_exc)
            return None
        logger.info("Langfuse client → %s", settings.langfuse_host)
        return client
    except Exception as exc:
        logger.warning("Langfuse client init failed: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush pending traces before process exit. Call at the end of every eval run."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush: %s", exc)


def get_callback_handler(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    """Return a LangChain CallbackHandler bound to Langfuse, or None if disabled."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        import os
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        _patch_langchain_compat()
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
        )
    except Exception as exc:
        logger.warning("Langfuse CallbackHandler unavailable: %s", exc)
        return None
