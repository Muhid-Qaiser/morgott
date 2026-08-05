# SWE-rebench long legitimate-task evaluation

This experiment freezes the `test` split from `nebius/SWE-rebench` revision `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881`.

It projects only `problem_statement`, keeps repository and commit lineage, removes local privacy matches, normalized exact duplicates, and retained-model fit overlap, and stores no raw task text in artifacts.

Run it against the two pinned test Parquet shards:

```bash
uv run --locked python experiments/swerebench_long_task_eval/run.py prepare --source-dir PATH/data
uv run --locked python experiments/swerebench_long_task_eval/run.py run --source-dir PATH/data
uv run --locked python experiments/swerebench_long_task_eval/run.py analyze
```

The result is a restriction and review-load diagnostic over legitimate repository tasks.

It is not a production false-positive estimate and selects no threshold.
