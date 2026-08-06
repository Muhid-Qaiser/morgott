# Matched subversion control

864 rows of the campaign corpus use one fixed jailbreak wrapper ending in a literal
`Request: <inner>` tail. Splitting the wrapper off constructs 864 pairs whose halves carry
identical harmful content and differ only by the presence of instruction subversion.

`score` first drops the 82 pairs whose wrapped half exceeds the scorer's 512-token limit.
The scorer truncates from the end without windowing (`core.py:129-135`) and the request
sits at the end of the wrapper, so those pairs would compare a partly-cut wrapped half
against a complete bare one. 782 pairs are scored.

That isolates what the aggregate positive-only recall in `experiments/cascade_showcase/`
cannot answer: does the detector fire on instruction subversion, or on harm topic?

```bash
uv run --extra showcase python -m experiments.redteam_subversion_control.run pairs
uv run --extra encoder python -m experiments.redteam_subversion_control.run score
```

`pairs` writes `data-archive/redteam/subversion_pairs_20260806.jsonl.gz`. `score` runs both
halves through a registered advisory shadow via `morgott.models.mmbert.inference.score_file`
and writes `artifacts/redteam_subversion_control/`. Scoring inputs are reduced to exactly
`{id, text, input_channel}` because that schema is enforced (`inference.py:277`).

The `inner` strings are derived text not attested in the source corpus — archive-class
diagnostic material, never canonical. Every score is advisory. Measured result and its
reading are in `reports/redteam-campaign-data.md`.
