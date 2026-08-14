# Guard-model baseline panel

Scores pinned third-party guard models on the **same row identities** the
incumbent advisory shadow is scored on, under the same shared-threshold
protocol, plus a positive-only contamination control.

Read these three sentences before reading any number this produces:

1. **These are already-open development baselines, not a prospective final
   test.** PromptShield test and SEP are published benchmarks, canonical
   dev_test has been scored repeatedly, and third-party training-source overlap
   is undisclosed for every baseline here. `docs/roadmap.md` still requires a
   genuinely untouched final test, which does not exist.
2. **No baseline result promotes a model by itself.** Promotion requires the
   repository reviews and a `model-artifacts.json` registry entry. A baseline
   losing does not promote the incumbent, and a baseline winning does not
   promote it either.
3. **Every score is advisory.** `scan` and `cascade` still return
   `decision: allow`.

## What it does

Two populations, one threshold protocol.

**(a) The standard panel** — canonical dev_test, PromptShield test, and SEP,
with the canonical calibration components supplying the threshold. This is the
population `experiments/evaluate_prompt_guard_2_full_mixture.py` established;
the slice breakdown and the shared-threshold protocol are preserved. The
threshold is selected on calibration only, at a 1% component-level false-alarm
target within each trusted channel, Bonferroni corrected — never on the test
slice.

**(b) The archived first-party red-team reserve** (`data-archive/redteam/`) —
5,112 rows with zero exact overlap against all 2,177,717 routing rows. This is
the contamination control: published third-party work measured Qwen3Guard
dropping from 85.3% to 33.8% on prompts not derived from public datasets, and
this panel tests exactly that. The reserve is frozen. The harness only reads
it; it must never enter training.

Reported per baseline: AUROC, PR-AUC, recall and FPR at the shared threshold
and at descriptive 1% and 0.1% FPR coordinates, per-source and per-channel
slices, per-instruction-subtype slices, SEP pair ordering, real-finance false
positives, truncation counts per slice, latency per slice, and peak VRAM.

Output goes to `artifacts/comparisons/<slug>/{evaluation.json,scores.npz}`,
`schema_version: 2`; archived schema-1 reports remain immutable legacy evidence.

## Baselines

| slug | repo | revision | ctx | what its scalar is |
| --- | --- | --- | ---: | --- |
| `modernguard-1` | `guardion/ModernGuard-1` | `a7c09c89…` | 8192 | softmax over the `INJECTION` class |
| `prompt-guard-2-86m-current-panel` | `meta-llama/Llama-Prompt-Guard-2-86M` | `a8ded8e6…` | 512 | softmax over class index 1, the pinned binary attack score |
| `protectai-deberta-v3-prompt-injection-v2` | `protectai/deberta-v3-base-prompt-injection-v2` | `90c9989b…` | 512 | softmax over the pinned config label `INJECTION` |
| `qwen3guard-stream-4b` | `Qwen/Qwen3Guard-Stream-4B` | `27a8f4e5…` | 8192 | `1 - P(Safe)` from the query risk head |
| `qwen3guard-stream-4b-jailbreak` | `Qwen/Qwen3Guard-Stream-4B` | `27a8f4e5…` | 8192 | `P(Jailbreak)` from the query category head |
| `kanana-safeguard-prompt-2.1b` | `kakaocorp/kanana-safeguard-prompt-2.1b` | `167d74d4…` | 8192 | `1 - P(<SAFE>)`, renormalized over the three fixed first-output label tokens |
| `granite-guardian-3.2-3b-a800m` | `ibm-granite/granite-guardian-3.2-3b-a800m` | `3de033d8…` | 8192 | first-position `P(Yes)` for `risk_name=jailbreak`, summing the documented trimmed/case-folded Yes / No token sets |
| `granite-guardian-4.1-8b` | `ibm-granite/granite-guardian-4.1-8b` | `69820a3f…` | 8192 | `prob_of_risk`, if the path still exists |
| `aprielguard` | `ServiceNow-AI/AprielGuard` | `e7e936d1…` | 32768 | softmax over the first-line `safe`/`unsafe` tokens |

Adapters live in `adapters.py` behind one interface — `load()`, then
`score(texts) -> (scores, overflow flags)`. Each declares its own context
limit, and every slice records how many of its rows exceeded it, because
context length is the variable under study. Adding a baseline is a `BASELINES`
entry, not new plumbing.

**An adapter may never invent a score.** When a documented extraction path is
absent on the pinned revision, it raises `ExtractionUnavailable` and the run
writes an `evaluation.json` with `status: "extraction_unavailable"` and the
reason. That is a recorded gap, not a zero and not a silent omission. Every
adapter also runs a two-row polarity smoke test before scoring 460k rows.

## Running it

Fetch the pinned snapshots first. The adapters use `local_files_only=True`, so
a run never downloads anything; a missing snapshot is a clear failure, not a
surprise network fetch.

```bash
hf download meta-llama/Llama-Prompt-Guard-2-86M \
  --revision a8ded8e697ce7c355e395a0df51f94adb4a2fd27
hf download protectai/deberta-v3-base-prompt-injection-v2 \
  --revision 90c9989b1a342275dd0d1a95aad283c04e075671
hf download guardion/ModernGuard-1 \
  --revision a7c09c891f539689c57a0e016f2b394d91b4586b
hf download Qwen/Qwen3Guard-Stream-4B \
  --revision 27a8f4e52e66dc01a03d20f41e362bb9c9bda7bf
hf download kakaocorp/kanana-safeguard-prompt-2.1b \
  --revision 167d74d4706b236580b0e48318337c7ac6ba7848
hf download ibm-granite/granite-guardian-3.2-3b-a800m \
  --revision 3de033d89b499a18d9a573b5192bf3b967ef48c5
hf download ibm-granite/granite-guardian-4.1-8b \
  --revision 69820a3f3c8f265e2fe61b5a8fcea2146c2fcb16
hf download ServiceNow-AI/AprielGuard \
  --revision e7e936d158cf054e9f078580e432a477bfdd5436
```

The red-team reserve is gitignored and lives in Azure. If it is absent the
harness **fails closed** with the pull command rather than skipping it:

```bash
scripts/azsync.sh pull data-archive
sha256sum -c data-archive/SHA256SUMS
```

Pin the panel identity once, then hold every baseline to it:

```bash
uv run --locked --extra encoder python -m experiments.guard_baselines.run --panel-only
```

Panel assembly reuses the trainer's verified prepared-corpus cache at
`artifacts/mmbert/prep-cache` by default. A valid cache turns the million-row
overlap stage into a few-second integrity-checked load; `--prep-cache DIR`
selects another cache and `--no-prep-cache` deliberately performs a cold
rebuild.

The current assembled identity is
`57a3f9362333a4649f78c17f4909dc21edf9ee713076ccade73b16eb53f1016c`.
Re-run `--panel-only` after any corpus change; do not carry this digest across
a rebuild. Then score each baseline. Order matters only in that baseline 0
sets the reference line.

```bash
uv run --locked --extra encoder python -m experiments.guard_baselines.run \
  --baseline <slug> \
  --require-panel-sha256 57a3f9362333a4649f78c17f4909dc21edf9ee713076ccade73b16eb53f1016c
```

Use each slug from the table above. Three baselines need extra arguments:

| slug | extra arguments |
| --- | --- |
| `kanana-safeguard-prompt-2.1b` | `--score-journal artifacts/comparisons/.kanana-safeguard-prompt-2.1b.rendered-length-v1.score-journal` |
| `granite-guardian-3.2-3b-a800m` | `--score-journal artifacts/comparisons/.granite-guardian-3.2-3b-a800m.rendered-length-v1.score-journal` |
| `aprielguard` | `--batch-size 1` |

Useful flags: `--list`; `--batch-size N` to override the per-baseline default;
`--require-panel-sha256 HEX` to refuse to score a panel that is not the pinned
row set; `--prep-cache DIR`; `--no-prep-cache`; `--output DIR`; `--skip-redteam` (recorded loudly in the output, never
silent); `--allow-population-drift` when the corpus has genuinely moved and you
intend to re-run the whole ladder.

The harness refuses to overwrite an existing output directory, and writes
through a temporary directory replaced atomically, as the existing comparison
runner does.

### Resuming an interrupted score pass

Numeric score journaling is enabled by default. For output
`artifacts/comparisons/<slug>`, the journal is written beside it as
`artifacts/comparisons/.<slug>.score-journal/`. Relaunch the exact same command
after an interruption and it resumes from the last atomic, model-batch-aligned
shard. This is particularly important for AprielGuard at batch 1 and for a
full Prompt Guard pass.

The journal contains only float64 `score` and `overflow` arrays plus hashes and
row ranges. It contains no prompt text or row IDs. Its identity binds the exact
model and revision, ordered panel text and all slicing metadata, scoring code
and lockfile, batch size, and column schema. The model identity hashes every
runtime model, tokenizer, template, remote-code, safetensors-index, and weight
shard byte. A same-size local cache mutation therefore changes the identity and
cannot silently resume an old numeric journal. Documentation and plot files in
the Hub snapshot are excluded because they do not affect scoring. Computing the
identity once can read several gigabytes for a sharded model; that cost is paid
before resumable scoring rather than trusting file size alone. Any mismatch
fails instead of reusing a partial score under a changed experiment. Use
`--score-journal DIR` to choose another root, or `--no-score-journal` for a
deliberately non-resumable pass. A completed output directory is still
immutable; the journal does not bypass the overwrite refusal.

Kanana and Granite Guardian 3.2 use a stricter path for their long causal
prompts. Each contiguous 512-panel-row journal shard is rendered and exactly
capped once, then stably sorted by `(rendered token count, original row
offset)` for model batches. Scores and overflow flags are scattered back to
the exact panel order before the numeric shard is appended. Rendered IDs are
ephemeral and never enter the journal or result. The fixed bucket boundary
makes an uninterrupted pass and a resumed pass use the same batches.

The batching strategy, bucket size, tie-break rule, restoration rule, and
requested attention backend are part of `preprocessing` and therefore the
model/journal identity. Runtime records prepared, rendered, padded, and
hypothetical panel-order padded-token counts for the current invocation. The
scoring contract is version 2, so a journal created by the old panel-order
causal path intentionally fails identity validation. Keep that evidence
untouched and start with the explicit `rendered-length-v1` journal paths in the
commands above.

Those same two pins explicitly request Transformers' `sdpa` implementation
and fail loading if a different implementation is resolved. Both pinned model
classes advertise native SDPA support; PyTorch chooses the applicable CUDA
SDPA kernel at runtime. The harness does not request `flash_attention_2`,
because that separate backend has not been validated for these exact pins.
Granite Guardian 4.1, AprielGuard, and every encoder/stream baseline retain
their prior panel-order behavior.

The current-panel Prompt Guard output uses the new
`prompt-guard-2-86m-current-panel` slug. It never replaces the historical
`prompt-guard-2-86m-full-mixture` artifact, whose rows and panel identity are
different.

ProtectAI v2 uses the generic `EncoderGuard`: the pinned config names exactly
two classes, `SAFE` and `INJECTION`, so the adapter resolves the positive class
by name and reports its softmax probability. It runs in BF16, uses the first
512 model-native tokens including special tokens, keeps the vendor cutoff of
0.5 as a descriptive native operating point, and never enables remote code.

Before any expensive full panel pass, run the generic deterministic 4,096-row
canary spanning calibration, canonical dev-test, PromptShield, and SEP. Pass
`--baseline SLUG`; omitting `--batch-size` uses that baseline's registered
default (Granite 3.2: 4, Qwen3Guard: 8, AprielGuard: 2), while an explicit value
is recorded as an override. The report records the ordered sample identity,
exact single- or sharded-model identity, source hashes, elapsed time, peak VRAM,
polarity smoke result, and descriptive cutoff metrics. A model with no usable
native threshold reports `quality_at_fixed_cutoff_0_5`, never a fabricated
native operating point. A projected full current-panel pass over 60 minutes is
a stop condition, not permission to launch it.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra encoder python -m \
  experiments.guard_baselines.canary \
  --baseline protectai-deberta-v3-prompt-injection-v2 \
  --output artifacts/comparisons/protectai-deberta-v3-prompt-injection-v2-canary-4096 \
  --require-panel-sha256 57a3f9362333a4649f78c17f4909dc21edf9ee713076ccade73b16eb53f1016c
```


The archived ProtectAI schema-1 canary remains checksum-bound and readable:
its single `model.safetensors` digest stays in `model_weights_sha256` and its
native 0.5 field remains `quality_at_native_cutoff_0_5`. The artifact binds the
producing canary and adapter source SHA-256 digests, but those exact source
bytes are not retained locally, so do not claim bit-for-bit source
reproducibility. New schema-2 reports bind the complete adapter model identity
in `model_identity_sha256`; sharded models leave the legacy single-weight field
null and carry every exact shard digest under `baseline.model_identity`.

## Known extraction risks, verified against the pinned revisions

These were checked against each repo's `config.json`, `chat_template.jinja`,
and modelling code at the pinned revision. Several unified baselines have now
completed; these notes describe the extraction contracts and remaining risks,
not the run-status ledger.

- **Kanana is one three-class first-token distribution, not two separately
  selected detectors.** The pinned tokenizer maps `<SAFE>`, `<UNSAFE-A1>`,
  and `<UNSAFE-A2>` to the dedicated IDs 128257, 128256, and 128258. The
  adapter verifies those exact identities at load time, then reports the
  predeclared primary scalar
  `1 - P(<SAFE>)` after renormalizing only over those tokens. A1 is prompt
  injection and A2 is prompt leaking. The numeric journal deliberately keeps
  one score plus overflow; it does not add a post-hoc A1-only column. The
  model's documented native verdict is a three-way argmax, which no threshold
  on pooled unsafe mass reproduces exactly, so `native_threshold` is null.
  Its config explicitly supplies `head_dim=128`; the adapter narrowly repairs
  Transformers 5's stricter legacy divisibility validator, which otherwise
  rejects the official 1792-hidden / 24-head config before tokenizer loading.
- **Granite Guardian 3.2 uses the still-documented guardian path at an 8k
  operational cap.** The adapter supplies
  `guardian_config={"risk_name": "jailbreak"}`, verifies that this changes the
  rendered template, and matches the official helper's
  `decoded_token.strip().lower()` rule. On the pinned tokenizer that is six
  exact vocabulary IDs for Yes and six for No, including leading-space and
  case variants; the adapter sums each class at the first generated position
  and refuses a changed set. The architecture declares 131072 positions, but
  this comparison predeclares 8192 total rendered tokens. Overflow is decided
  from the fully rendered token sequence, not raw content plus estimated empty
  template overhead. If it is over cap, only the content prefix is shortened
  until the re-rendered sequence is at most 8192 tokens; the exact IDs sent to
  the model are then padded without re-tokenization. This new slug and output
  directory are separate from the Granite 4.1 extraction-gap record.
- **Granite Guardian 4.1 has lost `prob_of_risk`.** The documented 3.x path
  passed `guardian_config` to the chat template and read a Yes/No softmax at
  the first generated position. The 4.1 `chat_template.jinja` contains no
  `guardian_config` and no guardian logic at all — it is a generic Granite
  chat template — and the 4.1 card documents only a regex over a
  `<score>yes|no</score>` block that is *not* at the first generated position.
  The adapter probes this at load time by rendering with and without
  `guardian_config` and comparing; if the kwarg is ignored it records the gap
  and produces no score. The probe is a runtime check, not a hardcoded
  verdict — pointing the same adapter at a 3.x revision scores normally.
- **AprielGuard's extractable scalar is the safety axis, not the adversarial
  axis.** Its template emits `safe`/`unsafe` on line 1 and
  `adversarial`/`non_adversarial` on line 2, after a variable-length category
  list; neither line-2 string is a single token, so there is no first-position
  injection scalar. `docs/data-contract.md` treats harm without instruction
  subversion as a *different label*, so this baseline is not measuring the same
  target as the rest of the ladder. Its `measures` field says so.
- **AprielGuard is ROC-only.** Its own technical report puts its aggregate
  operating point at ~11% FPR — an order of magnitude outside this repository's
  1% regime. It is never read at its native decision; `native_threshold` is
  `null` and `native_operating_point.usable_here` is `false`. No AUROC is
  published for it, so there is no vendor number to check against.
- **Qwen3Guard-Stream-4B is 8k, not 32k.** The 32k sibling is
  `Qwen3Guard-Gen-4B`, whose only documented readout is a regex over generated
  text. The Stream model's documented helper `stream_moderate_from_ids` returns
  `max(softmax(...))` rounded to two decimals — the probability of whichever
  class won, not of a fixed class — which cannot support an ROC. The adapter
  therefore calls `forward` and reads `query_risk_level_logits` directly, at
  the `<|im_end|>` token that closes the user turn. The first streaming call is
  itself a plain forward over that same span, so this reproduces the documented
  user verdict. Class indices are looked up by name from the pinned config,
  because the card's prose orders them differently from the config
  (`1` is `Unsafe`, `2` is `Controversial`).
- **ModernGuard-1's 8192 window is architectural, not trained.** Its own
  `eval_data.json` records training at `max_length: 2048` with `stride: 128`,
  and the card recommends 2048 at inference. Rows between 2048 and 8192 tokens
  are outside its trained regime, so truncation counts at 8192 understate the
  effective context. Read the "16x context" comparison with that caveat.
- Batch composition perturbs padded scores slightly, as
  `reports/redteam-campaign-data.md` records for the shared scorer. Baseline 0
  keeps the registered batch size of 8 for that reason.

## Deliberately excluded

**NemoGuard-JailbreakDetect, gliguard-300M, and Opir are excluded because they
train on WildJailbreak / WildGuard**, which this corpus also draws on. Scoring
a model on data it was trained on measures memorisation, not detection, and
would put a contaminated number in the same table as clean ones. Their absence
from the ladder is a decision, not an oversight.

## Reading the red-team panel

It is positive-only. There is no benign denominator, so **FPR, precision,
AUROC, and PR-AUC are undefined there and are reported as `null`**. Only recall
is meaningful.

The aggregate flag rate is a weighted average across two label classes and
should never be quoted as attack recall: roughly two thirds of the corpus is a
bare harmful request with no instruction subversion, which
`docs/data-contract.md` treats as a separate label. Read
`by_subversion_basis`, not the aggregate — this is the same correction
`reports/redteam-campaign-data.md` records for the cascade's 39.44% figure.

`contamination_control` reports canonical dev_test recall against reserve
recall on the subversion-attested subset at the same threshold. The two
populations differ in composition as well as provenance, so a large delta is
consistent with contamination but does not establish it.

`verdict` and `breached` are outcome metadata for one target model on one day,
never detector labels. `category` is confounded with `attack_mode`, so
per-category slices measure the campaign, not the topic.
