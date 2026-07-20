# Nemotron agentic indirect-injection audit

This evaluation-only experiment projects the pinned public NVIDIA source to its
`injection.injection_text` field and safe categorical lineage. It deliberately
does not persist the synthetic environment, identities, system/user prompts,
tool schemas, attack goal, or target arguments.

After rebuilding the pinned data and the unchanged core model:

```bash
make data
make benchmark
PYTHONPATH=src:. python experiments/nemotron_agentic_ipi/audit.py
```

The report measures direct-fallback, indirect-sensor, and combined shadow recall
by domain, attack category, injection vector, and target tool. It also checks
normalized exact and word-ngram cosine near-overlap against every other processed
fit/evaluation partition. The source has no clean controls, so none of these
results estimates false-positive rate, precision, benign utility, or production
safety.
