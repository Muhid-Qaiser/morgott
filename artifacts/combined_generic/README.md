# Retained generic mmBERT artifacts

This directory contains the owner-approved internal research artifacts registered in `model-artifacts.json`.
They are advisory only and are not approved for blocking, authorization, transaction approval, or privilege grants.

The public base encoder is not vendored.
Load `jhu-clsp/mmBERT-base` at revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`.
The upstream model card identifies that base model as MIT licensed.

The retained heads and adapter were trained from a mixed research corpus whose component redistribution terms are not uniformly established.
Keep this repository and its LFS objects private until a separate release review resolves source licensing and model-weight memorization risk.
No raw prompts, source JSONL, provider responses, credentials, or base-model weights are included in the retained artifact set.

Binary weights and NumPy evidence arrays use Git LFS.
`retained_SHA256SUMS` records every committed artifact byte hash.
The `evaluation_generic_v3` directories were produced by the versioned evaluator and preserve current-source metrics.
Earlier local `evaluation_generic_v2` directories are historical and are not part of the retained Git artifact set.

All four downstream checkpoints normalize and truncate each input to its first 512 tokens.
They do not chunk long documents or identify an injected span.
