# Contributing

Morgott is a private research repository with a lightweight pull request workflow.
Keep changes focused and preserve the security, data, and evaluation contracts in `AGENTS.md`.

## Setup

Install the locked project and development dependencies:

```bash
uv sync --locked
```

The pre-commit hooks are optional and apply Ruff fixes and formatting to staged Python files:

```bash
make hooks
```

CI remains the source of truth when hooks are not installed or are bypassed.

## Workflow

1. Create a short-lived branch using `feat/`, `fix/`, `docs/`, `chore/`, or `codex/` followed by a concise slug.
2. Make the smallest change that satisfies the intended outcome.
3. Run `make check` and any domain-specific verification required by `AGENTS.md`.
4. Open a focused pull request and describe its data and security impact.
5. Squash merge after CI passes so the pull request title becomes the durable commit summary.

GitHub cannot enforce branch protection for this private repository on its current plan.
Treat a green CI run as required team policy even though an administrator can technically bypass it.

## Verification

Every change must pass:

```bash
make check
git diff --check
```

Use the impact-based data verification rule in `AGENTS.md`; touching corpus-builder files alone does not require a rebuild.
For output-neutral refactors, run focused equivalence tests and state why generated data is unaffected.
For data-affecting changes, run the applicable rebuild and inspect its manifest hashes, counts, split invariants, and quarantine summary before handoff.

## Data and credentials

Never commit credentials, `.env` files, raw provider responses, ignored corpus outputs, or unreviewed local model artifacts.
The only model-artifact exceptions are the reviewed advisory research set listed in `model-artifacts.json` and documented comparison-only binaries.
Their binary weights must use Git LFS, retain SHA-256 provenance, and remain separate from the public base encoder and raw training data.
Comparison-only binaries must remain outside maintained inference and the model registry.
Adding or replacing a registered artifact requires licensing, privacy, and
reproducibility review plus an explicit `model-artifacts.json` update; it does
not require a separate owner-approval step.
Do not push the registered LFS objects to an external remote until the mixed-corpus redistribution review is complete.
Only `data/manifest.json` is versioned from the local data tree.
Do not send corpus text to an external provider without an explicit, separately reviewed experiment.
