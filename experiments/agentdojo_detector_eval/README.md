# AgentDojo Banking cascade evaluation

This study scores the pinned AgentDojo `v1.2.2` Banking suite through the frozen advisory cascade.
It materializes the complete serialized tool-return that contains each `direct` or `important_instructions` attack and the 16 unattacked user prompts.

Prepare the text-free panel before scoring:

```bash
uv run --extra cascade --with 'agentdojo==0.1.35' python experiments/agentdojo_detector_eval/run.py prepare
```

Preparation pins the package and source hashes, validates the exact 304-case projection, exact-deduplicates the 88 model inputs, and applies Morgott's existing full-fit overlap guard.

Run the already-selected cascade without changing any threshold:

```bash
uv run --extra cascade --with 'agentdojo==0.1.35' python experiments/agentdojo_detector_eval/run.py run --allow-remote
uv run --extra cascade --with 'agentdojo==0.1.35' python experiments/agentdojo_detector_eval/run.py analyze
```

The ledger stores hashes, lineage, routes, typed reviewer scores, latency, token counts, and artifact identities, but no task text, injected text, tool result, or raw provider response.
Re-running `run --allow-remote` retries only missing or operationally incomplete assessments.
The retained channel-low artifact completed all 88 unique assessments with zero provider failures.
Do not analyze an incomplete ledger with this frozen runner: its historical analyzer includes fail-safe restrictions in rate summaries, so any new execution requires a corrected versioned runner rather than reusing this artifact identity.
The result is an already-public detector-transfer diagnostic, not AgentDojo attack success, agent utility, or a prospective final test.
