# Deferred attack-source contracts

This directory records field and label invariants for sources that are useful
to study but are not active inputs to the POC.

`hackaprompt/hackaprompt-dataset` is still an auto-gated Hugging Face dataset
that requires sharing contact information. The project does not accept that
gate, use `HF_TOKEN` to access it, or fetch a mirror. `audit.py` makes no network
request; synthetic tests lock a possible future local projection to
`user_input` only. Full `prompt`, model `completion`, and
`expected_completion` fields are excluded because they leak challenge/system
templates, target outputs, and model behavior into a text detector.

Every competition submission is an attack **attempt** by collection context.
The `correct` field means that a particular target model produced the expected
challenge output; it is retained separately as target-specific attack success.
An unsuccessful attempt must never become a benign/negative label. Exact
normalized duplicate user inputs aggregate their target outcomes.

If access is explicitly approved later, first verify the gated Parquet content
hash and real schema against the pinned repository audit. Keep the source
evaluation-only initially, locally scan/redact PII and secrets, quantify
offensive/toxic content separately, and report successful-only plus all-attempt
recall. If a later ablation trains on it, hold out whole challenge levels and
remove cross-level text/near duplicates; the published schema exposes no
participant identifier suitable for participant-group splitting.

The pinned Yaklang `llm-prompt-injection` skill is not installed or vendored.
It is a useful scenario checklist, not a labelled corpus or validated generator.
Its direct override, prompt extraction, encoding, and basic indirect examples
mostly overlap existing deepset, Tensor Trust, multi-turn, and BIPIA coverage.
The useful gaps are stateful tool-description/tool-output injection, cross-tool
exfiltration, Markdown egress, and MCP provenance. Those belong in typed
reference-monitor/AgentDojo scenarios, not duplicated keyword-heavy training
rows. See `reports/hackaprompt-yaklang-audit.md` for pinned provenance.

```bash
PYTHONPATH=src python -m unittest discover -s experiments/exploratory_sources -v
```
