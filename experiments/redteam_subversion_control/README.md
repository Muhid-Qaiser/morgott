# Matched subversion control

864 rows of the campaign corpus use one fixed jailbreak wrapper ending in a literal
`Request: <inner>` tail. Splitting the wrapper off constructs 864 pairs whose halves carry
identical harmful content and differ only by the presence of instruction subversion.

That isolates what the aggregate positive-only recall in `experiments/cascade_showcase/`
cannot answer: does the detector fire on instruction subversion, or on harm topic?

```bash
uv run --extra showcase python -m experiments.redteam_subversion_control.run
```

The helper writes `data-archive/redteam/subversion_pairs_20260806.jsonl.gz`.
The retired 512-token scoring path and its measured outputs remain in Git history and
`reports/redteam-campaign-data.md`; they are not maintained inference tooling.

The `inner` strings are derived text not attested in the source corpus — archive-class
diagnostic material, never canonical.
