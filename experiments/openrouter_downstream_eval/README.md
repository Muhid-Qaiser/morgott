# OpenRouter downstream evaluation

Disposable, completed development study of whether a text-only remote LLM can
improve the advisory mmBERT cascade. The durable findings, cost accounting,
limitations, and evidence links are in
[`reports/openrouter-downstream-evaluation.md`](../../reports/openrouter-downstream-evaluation.md).

`run.py` prepares the frozen panel, validates it, and runs a bounded provider
study; `analyze.py` and `followup.py` analyze already-recorded ledgers. Typical
local checks are:

```bash
uv run --locked python experiments/openrouter_downstream_eval/run.py self-check
uv run --locked python experiments/openrouter_downstream_eval/analyze.py --help
```

Running provider calls is not a routine test. It requires an explicit bounded
development experiment, a positive `--max-cost`, and the privacy rules in
`AGENTS.md`: never persist panel text, raw provider responses, or credentials.
The runner records hashes, row identities, structured verdicts, failures,
latency, token usage, and cost under `artifacts/openrouter_downstream_eval/`.
All learned outputs remain advisory.
