# SWE-bench Verified long-benign evaluation

This prospective development diagnostic evaluates the unchanged registered local gate on every privacy-screened, fit-disjoint SWE-bench Verified problem statement in the canonical dev-test source.

The panel is benign-only and grouped by repository.
It measures local hard restrictions and the DeepSeek review workload by repository and fixed character-length bucket.
It cannot measure attack recall, balanced long-context robustness, production prevalence, or a production false-positive rate.

`prepare` freezes the source, fit-overlap filter, model contract, analysis rules, and text-free panel before scoring.
`run` performs provider-free local inference with the registered artifact.
`analyze` seals the result and permits a separately frozen remote phase only when hard restrictions do not exceed 1% and at most 4,000 windows need review.

```bash
uv run python experiments/swebench_long_benign_eval/run.py prepare
uv run python experiments/swebench_long_benign_eval/run.py run
uv run python experiments/swebench_long_benign_eval/run.py analyze
```

No raw issue text is written under `artifacts/`.
No threshold, prompt, model, or provider may be selected from this panel.
