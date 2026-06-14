# eval/

Three evaluation frameworks as required by the course spec.

## Files

### ragas_eval.py
**RAGAS** - reference-free evaluation of the full RAG pipeline.

Runs the agent on every question in `data/golden/golden_set.jsonl` and scores:

| Metric | Measures |
|---|---|
| `context_precision` | Are retrieved chunks actually relevant? (precision of retrieval) |
| `context_recall` | Are all relevant facts retrieved? (recall of retrieval) |
| `faithfulness` | Is the answer supported by retrieved chunks? (no hallucination) |
| `answer_relevancy` | Does the answer address the question? |

Output: `reports/ragas_v{1,2}.json` + timestamped CSV.

```bash
paperpilot eval ragas --version v2
```

### tool_call_acc.py
**Tool Call Accuracy** - did the agent call the right tool(s)?

Compares actual tool calls against `expected_tools` in the golden set.
Score = 1.0 if actual ⊇ expected, else 0.0 (partial credit not given).

Output: `reports/tool_call_acc_v2.json`

```bash
paperpilot eval tool-call-acc --version v2
```

### haic_eval.py
**HAIC** - Human-AI Collaboration evaluation.

Simulates the interaction loop (query → retrieve → respond) and scores on:

| Metric | Scale | Description |
|---|---|---|
| `helpfulness` | 1-5 | Does the answer address the user's intent? |
| `trust_calibration` | 1-3 | Does confidence match correctness? |
| `effort_saved` | 1-5 | How much reading effort is saved vs. going to source? |
| `harm_potential` | 1-3 | Worst-case cost if the user trusts a wrong answer |

Also processes real user feedback from `data/haic/session_*.jsonl` (👍/👎 from the Chainlit UI).

Output: `reports/haic_v2.json`

```bash
paperpilot eval haic --version v2
```

### golden_gen.py
LLM-assisted golden question set generation.

Reads `data/processed/*.md` and generates candidate Q&A pairs. Output goes to `data/golden/golden_set.jsonl` (requires human review before use).

```bash
paperpilot eval golden-gen --n 50
```
