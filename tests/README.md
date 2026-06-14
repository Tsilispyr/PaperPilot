# tests/

Unit tests for core components.

## Test files

| File | Tests |
|---|---|
| `test_chunkers.py` | FixedSizeChunker (v1) and SectionAwareChunker (v2) - token counts, overlap, section split |
| `test_filter_inference.py` | Heuristic metadata filter inference from query text |
| `test_tool_call_acc.py` | Tool Call Accuracy scorer - expected vs actual tool calls |

## Run

```bash
pytest tests/ -q          # all tests
pytest tests/ -v          # verbose
pytest -k "test_fixed"    # filter by name
```
