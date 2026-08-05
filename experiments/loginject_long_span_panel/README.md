# Sealed LogInject long-log panel

This preparation freezes matched clean and attacked 50-entry batches from Zenodo record `20436935` without scoring them.

The attacked arm replaces one or more contiguous generated benign entries with one source attack entry or its ordered context-stitching fragments.

Artifacts keep hashes, lineage, token counts, and strict-normalized complete-entry spans, but no raw log text.

Run preparation against the extracted artifact root:

```bash
uv run --locked python experiments/loginject_long_span_panel/prepare.py --source-root PATH/loginject_usenix26_artifact_submission/artifact
```

The panel must remain unscored until one candidate architecture and operating point are frozen.

It is a generated long-log diagnostic, not natural-document evidence.
