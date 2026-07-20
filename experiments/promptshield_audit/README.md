# PromptShield audit

This isolated experiment audits the public, ungated
[`hendzh/PromptShield`](https://huggingface.co/datasets/hendzh/PromptShield)
release at commit `a5234cb1f5cdb256600cab64b8c961195b5e8404`. The originally
suggested `NVIDIA/PromptShield` path is not the public dataset identifier.

PromptShield remains evaluation-only. Its files contain only `prompt` and a
binary source label; they omit source, conversation, task, attack-template, and
group lineage. The accompanying paper says it aggregates public instruction,
chat, and injection sources that can overlap this project's active corpora.
Nothing here is added to training, and no source is guessed from prompt text.

Run the pinned download and CPU audit from the repository root:

```bash
PYTHONPATH=src python experiments/promptshield_audit/audit.py --fetch
python -m unittest discover -s experiments/promptshield_audit -v
```

`--fetch` downloads exactly 31,174,217 bytes into ignored `data/`, validates
the pinned sizes and SHA-256 digests, and never prints prompt text. The audit:

- checks raw, normalized, and strict approximate near-overlap against every
  active processed fit/evaluation file;
- checks within- and cross-split duplication and label conflicts;
- reports length and whitespace-token truncation proxies; and
- scores only PromptShield's test split with every already-declared character
  model precision profile, without threshold tuning.

The compact outputs are [`results.json`](results.json) and
[`REPORT.md`](REPORT.md). No neural scores are reported because no reusable
PromptShield score cache existed; this experiment downloads no model weights,
uses no GPU, and performs no training.
