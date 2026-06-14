# agent/

Multi-Agent LangGraph orchestration: **Planner → Researcher → Synthesizer**.

## Files

### graph.py
The LangGraph `StateGraph`. Three async nodes:

| Node | LLM tier | Responsibility |
|---|---|---|
| `planner_node` | fast (gpt-4o-mini) | Classifies intent; emits `SearchPlan` (mode, queries, flags) |
| `researcher_node` | fast | Runs `rag_retrieve` + `arxiv_search` concurrently via `asyncio.gather`; distils chunks → bullet facts |
| `synthesizer_node` | strong (gpt-4.1-mini) | Receives bullet facts + user question; writes cited Markdown |

**State** (`AgentState`):
```python
user_query      str           # original question
plan            dict          # planner output
research_notes  str           # distilled bullets → synthesizer
raw_chunks      list[dict]    # full chunk dicts → UI sources panel
tool_calls      list[dict]    # accumulated for eval scripts
messages        list          # LangChain message history
answer          str           # final output
```

**Routing**: `out_of_context` mode skips the Researcher entirely.

**Public API** (backward-compatible with eval scripts):
```python
run_agent(question, version)          # sync, returns {"answer", "messages", "tool_calls"}
arun_agent(question, version)         # async version
astream_agent(question, version)      # async generator, per-node updates
stream_agent(question, version)       # sync legacy shim
build_graph(version)                  # returns compiled StateGraph
```

### tools.py
Two retrieval tools, used by the Researcher node:

| Function | Description |
|---|---|
| `rag_retrieve_impl(query, version)` | Dense search over Qdrant → optional cross-encoder rerank (v2) |
| `arxiv_search_impl(query)` | Live ArXiv API search for recent / out-of-corpus papers |
| `get_tools(version)` | Returns `[StructuredTool, StructuredTool]` for legacy ReAct usage |

### prompts.py
System prompts for each node. Edit here to change agent behavior without touching graph logic.

## Model Routing

```
PLANNER_LLM_MODEL      →  Planner + Researcher  (cheap, fast)
SYNTHESIZER_LLM_MODEL  →  Synthesizer            (powerful, quality)
```

Both default to the provider's default model if empty. Set in `.env`.

## Flow Diagram

```
START
  |
  ▼
planner_node  --(out_of_context)--► synthesizer_node --► END
  |
  | (quick_qa | deep_analysis)
  ▼
researcher_node
  |  asyncio.gather:
  |   +- rag_retrieve_impl(rag_query)
  |   +- arxiv_search_impl(arxiv_query)   ← only if needs_arxiv=True
  |
  ▼
synthesizer_node
  |
  ▼
END  →  {"answer": str, "tool_calls": list}
```
