# Retained mmBERT shadows

This directory contains the owner-approved internal research artifacts registered in `model-artifacts.json`.
They are advisory only and are not approved for blocking, authorization, transaction approval, or privilege grants.

The public base encoder is not vendored.
Load `jhu-clsp/mmBERT-base` at revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`.
The upstream model card identifies that base model as MIT licensed.

The retained heads and adapter were trained from a mixed research corpus whose component redistribution terms are not uniformly established.
Keep this repository and its LFS objects private until a separate release review resolves source licensing and model-weight memorization risk.
No raw prompts, source JSONL, provider responses, credentials, or base-model weights are included in the retained artifact set.

The active tree retains the selected frozen head and the one-seed LoRA head plus adapter.
Binary weights use Git LFS, and `model-artifacts.json` is the sole registry for their paths and hashes.
The JSON result and evaluation records are immutable evidence produced by source commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`.
Paths embedded inside those records describe the originating run and are not current filesystem locations.
Control seeds, evaluation arrays, and the exact completed runners remain available from Git history.
The compact maintained successor is `src/morgott/models/mmbert/`; any new run records its own provenance and does not rewrite these historical records.

Both retained shadows normalize and truncate each input to its first 512 tokens.
They do not chunk long documents or identify an injected span.
