# Research review, July 2026

An outside-in review of what morgott is trying to do, what is actually working,
what is not, and what to do next. Written to be argued with. Numbers are taken
from `data/manifest.json`, the versioned reports, and the retained artifacts;
where a number is recomputed here the derivation is stated.

## 1. What this project is actually trying to do

Two claims, stacked:

1. **Predictive.** An advisory sensor that routes direct jailbreaks, direct
   prompt injection, and indirect prompt injection to review, without drowning
   ordinary conversation — especially legitimate finance and security talk.
2. **Architectural.** A deterministic reference monitor that bounds impact when
   the sensor is wrong, so that text never becomes authority.

Claim 2 is the differentiator and the durable one. Claim 1 is a cost-reduction
layer in front of it. The repository's own documents already say this
("prediction reduces exposure; deterministic authorization bounds impact when
prediction fails"). Almost all of the effort so far has gone into claim 1.

## 2. What is genuinely strong

This is worth stating plainly, because the rest of this document is critical.

- **The evaluation hygiene is better than most published work in this area.**
  Grouped splits by real lineage, exact-conflict quarantine (76,202 rows),
  SimHash near-overlap quarantine, official-holdout locking, sealed dev-test,
  source-heldout folds, predeclared gates, and a standing refusal to promote.
  Most prompt-injection classifier papers would not survive this process.
- **The shortcut audit is the most valuable artifact in the repo.** Source
  identity explains 87.75% of direct-user label entropy; a source-majority
  lookup that never reads the prompt reaches 95.94% validation accuracy at 100%
  recall / 9.40% FPR. That single result invalidates the headline metric of
  most public injection detectors, including several this project evaluated.
- **The direct-route repair is a real, well-earned win.** The English
  ModernBERT + mmBERT-base mean-route ensemble reaches 0.101% FPR at 77.16%
  recall on the open direct suite, from a starting point of ~6% FPR. It was
  achieved by adding *matched counterexamples*, not by architecture.
- **The build is genuinely reproducible.** One command, fail-closed on any
  digest/schema/gate mismatch, manifest published last, 61 fast tests green.
- **The runners are not lost.** `artifacts/research-source-archive/2026-07-26/`
  contains `routing_encoder.py` (5,502 lines) plus 954 lines of tests, with
  `masked_bce_loss`, `aligned_pair_ranking_loss`, `document_bag_bce_loss`,
  fold logic, and threshold profiling already implemented. Re-running an
  experiment costs a `pyproject` edit, not a rewrite.

## 3. The core finding: the corpus is large and thin

The corpus is 4,061,614 canonical rows and ~20 GB on disk. The recipe that
produced the best model trained on roughly **31,000 rows**.

That is not a bug in the recipe. It is forced by the corpus:

| Fact | Value |
|---|---|
| Train-view rows | 1,478,191 |
| Rows reachable under the 2,000-per-source-per-label cap the best recipe uses | **33,103 (2.24%)** |
| Train sources containing **both** routing classes | **2 of 19** |
| Train rows living in single-class sources | **90.0%** |

The two mixed-class train sources are `browsesafe` (9,566 rows, and see §4.2)
and `wildjailbreak` (138,625, synthetic, whose benign side is
adversarial-benign and formally auxiliary). Everything else — Taskmaster
(469,255 rows), HackAPrompt (270,705), Schema-Guided Dialogue (195,747),
Tensor Trust raw (182,111), LLMail (84,276) — is 100% one class.

**Consequence: the marginal value of another row from an existing source is
approximately zero.** Scale in this corpus is redundancy inside single-class
distributions. The per-source cap that fights the shortcut also throws away
97.76% of the corpus, and it has to. The binding constraint is the number of
distinct *(source × label)* cells, not rows.

This reframes P0. The corpus build was executed well, but "finish and freeze
the corpus" solved a problem that was not the bottleneck. The bottleneck is
that almost no source in it contains both an attack and a legitimate example
drawn from the same distribution.

Every time matched contrast was added, it worked:

| Intervention | Effect |
|---|---|
| Style-matched multi-turn rows | Multi-turn PR-AUC 0.6421 → 0.9321; ROC-AUC 0.4556 → 0.8752 (below random → useful) |
| Clean non-adversarial WildGuardMix counterexamples | Direct-suite FPR cut by more than an order of magnitude |
| Pair-balanced BIPIA contexts | Removed the direct/finance regression while keeping the BIPIA recall gain |
| Boundary pairs + ranking loss | Both-correct 9.17% → 90.00% |

Every time undifferentiated data or a new objective was added, it failed
(OR-Bench cost 5.1 recall points for 0.32 FPR points; symmetric BrowseSafe max
loss dropped recall 19.70% → 3.11%; 2,048-token training made BrowseSafe worse;
full fine-tune hit 51.22% held-out FPR at ~0 training loss).

The evidence is one-directional and the project has already written it down.
It has not yet acted on it at scale.

## 4. Losses and failure scenarios

### 4.1 The indirect axis rests on a single benign distribution

`browsesafe` is the **only** source contributing benign `untrusted_content`
rows to train or validation. Everything else on that axis is positive-only:
LLMail is 461,850 rows with **203 benign**; Nemotron agentic IPI is 1,272 rows,
**100% positive, dev-test only**; BIPIA train is 436 rows, all positive.

So every indirect false-positive number the project has ever produced is,
effectively, a BrowseSafe number.

### 4.2 BrowseSafe as used is close to ill-posed

A sampled row is a 107,462-character whole HTML document, `known_attack_span:
false`, with **no associated user task**. That is ~27k tokens, ~53 windows of
512, one document-level label, no span, and no notion of what the user asked.

Without the task, "click here to subscribe" and an injected instruction are not
separable in principle. Best result to date is PR-AUC 0.7674 and 18.07% recall
at 0.88% FPR. That is not a model failure so much as a specification failure.

### 4.3 Zero finance attack data, in a project about finance agents

All four finance sources are 100% benign: `banking77` 13,083/0,
`harper_valley_bank` 19,301/0, `tatqa` 32,464/0, `financebench` 339/0. There is
not one finance or transaction attack row in the corpus. Nemotron's domains do
not include banking. `policy.py`'s `transfer_funds` scenario is hand-written
with nothing behind it.

The mission statement names finance explicitly. The corpus supports only the
benign half of it.

### 4.4 Multilingual FPR is unmeasurable, and a multilingual model was selected

There is not a single non-English benign row in train, validation, or dev-test.
OASST1 (88,838 rows, the one multilingual source) is entirely auxiliary/
uncertain and reaches no view. MASSIVE is deliberately en-US only. The only
non-English supervised rows are ~2,724 Chinese Do-Not-Answer rows, all label 1.

mmBERT-base was then selected as the strongest direct backbone. Its multilingual
false-positive behaviour is completely unmeasured, and the only non-English
signal it saw during evaluation was 100% positive. If it has learned
"non-English ⇒ escalate," nothing in the current evidence would reveal it.

### 4.5 Nothing above 256 tokens is benign

All 8,355 selected train rows over 256 tokens and all 2,323 validation rows over
256 tokens are non-benign. Length is a perfect classifier on this data. Raising
context therefore cannot be evaluated — the FPR denominator does not exist.
This is correctly documented, and it silently poisons any long-context claim.

### 4.6 No prospective test exists, and dev-test is now open

Every number in the ledger is on already-open development data. The
three-way dev-test comparison "opens that dev-test role permanently." Rogue
Security, the obvious external check, overlaps the canonical corpus by
**54.28%**. Everything is single-seed. There is no variance estimate anywhere.

Practical effect: the project cannot currently distinguish a real improvement
from selection noise, and has run enough recipes for that to matter.

### 4.7 The Boundary gate is too small to decide anything

"Boundary both-correct" is a headline decision metric across the ledger. The
selection is 160 rows / 80 pairs, and several reported comparisons use 40 pairs.
Moving from 87.5% to 95.0% on 40 pairs is three items. Differences of that size
have been used to rank recipes.

### 4.8 The retained active model does not cover the indirect channel at all

`routing_baseline.py` hard-filters to `input_channel == "direct_user"`,
discarding all 104,731 untrusted-content train rows. The only indirect sensor
in the active tree is the legacy char n-gram detector, trained on **986 BIPIA
rows**.

### 4.9 Every number here is single-shot; the adversary is not

This is the gap I missed on the first pass, and it may be the largest one.

Every metric in the ledger is one pass over a static row. Real attackers retry.
The frontier labs have all moved to reporting attack success as a *function of
attempt count*, and the curves are steep:

- Anthropic's Claude Opus 4.5 system card, Table 5.2.2.1.A (coding, Gray Swan
  "Shade" adaptive attacker, no safeguards): Opus 4.5 with extended thinking
  goes **0.3% at 1 attempt → 10.0% at 200**. Sonnet 4.5 goes **17.7% → 70.0%**.
  Computer use, Table 5.2.2.2.A: Sonnet 4.5 extended thinking **14.2% → 85.7%**.
- Google's Gemini defence report is blunter still. On a non-adaptive holdout of
  1,799 prompts across five held-out tools, Gemini 2.5 scored **18% ASR** — and
  **94.6%** under adaptive TAP on the held-out calendar scenario. Their own
  sentence: *"Had we not conducted further adaptive attacks and evaluations, we
  would have incorrectly concluded that Gemini 2.5 exhibits a higher degree of
  robustness than it does."*
- NIST CAISI's AgentDojo extension reports **57% at one attempt rising to 80%
  at 25 attempts**, and that stock AgentDojo attacks achieved **11% ASR** while
  purpose-built attacks reached **81%** on Workspace tasks.

(The widely-circulated "Opus 4.5: 4.7% / 33.6% / 63% at k=1/10/100" figures are
**not** verifiable from the system card's text layer — they are chart-only, and
come from the *combined* direct + indirect + jailbreak figure rather than the
indirect-only split. Do not cite them as indirect-injection ASR. The Table
5.2.2.x numbers above were read from the PDF text and are safe.)

Two consequences for morgott:

1. A single-shot FPR/recall pair does not characterise this system against an
   adaptive adversary. The reported 77.16% recall is an upper bound that decays
   with every retry an attacker is allowed.
2. "Passing" any benchmark with its shipped attacks proves very little. Anthropic
   has retired its own static internal benchmark in favour of adaptive
   red-teaming plus a live bug bounty. A static suite measures memorisation of
   that suite.

morgott has no adaptive-attack evaluation of any kind. Adding attempt-scaling —
even crudely, by allowing an attack generator N tries against a frozen
checkpoint and plotting ASR against N — would produce more decision-relevant
evidence than another recipe comparison.

(Attempt-scaling figures above are from lab system cards and a NIST blog, cited
via a secondary summary; the pattern is attested across independent sources but
the specific decimals were not re-read from primary tables.)

### 4.10 morgott has never been placed on the external FPR-versus-recall board

Every number in `reports/model-experiments.md` is internal. The published
board is harsher than morgott's internal framing suggests, and it cuts in
several directions at once. All figures below were read from primary sources.

**The open-source detector floor is very low at deployable FPR.** PromptShield
(arXiv 2501.15145 — Jacob, Alzahrani, Hu, Alomair, **Wagner**; UC Berkeley +
KACST, CODASPY 2025) evaluates on ~24k held-out samples:

| Detector | AUC | TPR@1% FPR | TPR@0.1% FPR |
|---|---|---|---|
| Meta PromptGuard **v1** | 0.874 | 12.78% | 9.39% |
| ProtectAI v1 / v2 | 0.646 / 0.705 | 7.05% / 1.97% | **0.00% / 0.00%** |
| InjecGuard | 0.765 | 20.37% | 6.61% |
| PromptShield (DeBERTa) | 0.976 | 43.22% | 31.45% |
| PromptShield (Llama-3.1-8B) | 0.998 | 94.80% | 65.33% |

Two shipped detectors detect *literally nothing* at 0.1% FPR. This is the
context in which morgott's 0.101% FPR at 77.16% recall should be read — but
that comparison is **not yet legitimate**, because morgott's number is on its
own dev-test and PromptShield's is on theirs. Running morgott's checkpoints
against the PromptShield eval split is the cheapest way to find out whether
77.16% survives contact with someone else's benign distribution. That is
workstream 1.

⚠️ This table does **not** cover Prompt Guard **2** (it shipped three months
after submission), Azure, or Lakera. Do not present the 12.78% as a PG2
measurement. Separately: Meta self-reports PG1 at 21.2%@1%FPR where Wagner's
group measures 12.78% — a ~1.7× optimism factor on the vendor's own OOD claim,
which is the best available prior for how to read Meta's unreplicated
**97.5%@1%FPR** claim for PG2.

**The strongest argument against morgott's detector arm is in AgentDojo.**
Table 5, `important_instructions` on GPT-4o:

| Defense | Benign utility | Targeted ASR |
|---|---|---|
| No defense | 69.00% | 57.69% |
| Transformers PI detector | **41.49%** | 7.95% |
| **Tool filter** | **73.13%** | **6.84%** |

The detector buys a 49.7-point ASR reduction for **27.5 points of benign
utility** — a ~40% relative capability loss. The tool filter reaches
*equivalent* ASR at **no utility cost at all**. Restated for this repo: the
reference monitor in `policy.py` is the arm with the better published
cost/benefit, and the detector is the arm that has to justify itself. That
inverts the effort allocation in the current corpus work.

**But the anti-classifier story is weaker than it looks — three
counterweights, and I have to report them.**

1. *AgentDojo's detector failure was a weak model, not the concept.*
   PromptArmor (arXiv 2507.15219, Shi/Zhu/Wang/…/Song) simply *prompts* GPT-4o
   to spot and strip injected content, and reports **FPR, FNR, and ASR all
   below 1% on AgentDojo**. The authors argue it should be the standard
   baseline. morgott has never measured against a prompted-LLM detector, and
   this is the cheapest strong baseline in the field. If a zero-training GPT-4o
   prompt beats a fine-tuned ModernBERT ensemble on AgentDojo, that is the
   single most important thing to know before another encoder run.
2. *Classifiers degrade **least** under adaptation, not most.* This contradicts
   the framing I used in §7.2. Google DeepMind (arXiv 2505.14534, Gemini 2.0
   Flash, ASR non-adaptive → adaptive): Spotlighting **18.0 → 82.4**, ICL
   **67.6 → 98.4**, but Retrieved-Data Classifier **68.2 → 5.0** and User
   Instruction Classifier **0.4 → 3.0**. Appendix H.7: the instruction
   classifier *"proved to be the strongest of those tested, maintaining a high
   attack detection rate even against adaptive attacks."* What collapses under
   adaptation is **in-context** defenses. ⚠️ The catch is fatal for
   cost/benefit though: that defense has **zero measured FPR** — verbatim,
   *"The precision of this defense is subject to further investigation, which
   we did not conduct."* And the paper's *other* classifier, reusing
   ShieldGemma on email, hit **FPR 82.1% (Flash) / 31.1% (Pro)**. Their own
   recommendation is a narrowly fine-tuned injection classifier over a
   repurposed safety model — which is what morgott is.
3. *Layered classifiers demonstrably raise attacker cost, with the only
   quantified evidence in the field.* Anthropic's Constitutional Classifiers
   (2501.18837): 405 red-teamers, ~4,720 hours, $95K paid, **no universal
   jailbreak**; ablation ASR ~16% helpful-only → ~14% after harmlessness
   training → 2% with input classifier → **0.5% with input+output**. RLHF
   bought 2 points; classifiers bought 13.5. Constitutional Classifiers++
   (2601.04603, Jan 2026) then drove **compute overhead to 3.5%** of the
   prior system at a **0.05% production refusal rate**, with 1,736 red-team
   hours / ~198K attempts yielding **one** high-risk vulnerability.
   ⚠️ Do not repeat that paper's own "0.38% → 0.05%, 7.6× lower" comparison —
   0.38% is an absolute *increase over baseline* from the 2025 paper, 0.05% is
   an absolute *rate* in the 2026 one. Apples to oranges.

**Over-defense is measurable and morgott has never measured it.** NotInject
(InjecGuard, arXiv 2410.22770 — published at ACL 2025 as **PIGuard**; cite v3,
v1 numbers differ materially and still circulate) is 339 benign sentences
seeded with trigger words like "ignore" and "cancel". Implied FPR: Prompt
Guard v1 **99.71%**, Fmops 71.68%, Deepset 70.50%, ProtectAI v2 42.77%, Lakera
12.39%, GPT-4o 13.27%. PG1 flags **100%** of two- and three-trigger benign
sentences. CAPTURE (2505.12368, ACL 2025 LLMSEC) finds PromptGuard at
**100.00% FPR** on Stock/Movies/Travel/Covid domains. A trigger-vocabulary
sweep over morgott's checkpoints costs nothing and is the direct test of
whether the 0.101% FPR is real or an artifact of benign distributions that
never say "ignore the above".

**And high FPR is not only a UX cost — it is an attack primitive.** arXiv
2410.02916 shows a **~30-character** adversarial prompt that universally blocks
**over 97%** of user requests on Llama Guard 3. A detector with a bad FPR tail
is a denial-of-service surface, which belongs in `docs/threat-model.md` and is
not currently there.

**Vendor claims are mostly unfalsifiable; do not benchmark against them.**
Azure Prompt Shields and Google Model Armor publish **no detection or FPR
numbers whatsoever** — Model Armor's release notes are six qualitative
"reduced false positives" entries with no operating points. Lakera's PINT
leaderboard is topped by Lakera on a dataset only Lakera holds, and its own
FPR claims are mutually inconsistent across properties (0.5% / 0.1–0.2%); the
widely-quoted "0.01%" appears on no Lakera or Check Point property at all.
NVIDIA is the honourable exception, publishing reproducible weak numbers
(31.19% jailbreak detection at 7.44% FPR for length-per-perplexity).

The gap that matters most: **no published guard-model FPR on genuine
production traffic exists anywhere**, from anyone except Anthropic, and theirs
is for jailbreak rather than injection classifiers. Any morgott FPR claim is a
claim about LMSYS/WildChat-shaped proxies, and should say so.

## 5. Claim-versus-code gaps

These are places where the documentation is ahead of the implementation. None
are dishonest — the threat model flags most as deferred — but they should not be
described as working.

1. **`authorize()` never reads provenance.** `context["provenance"]` is
   shape-validated (`policy.py:229-243`) and then never used in any decision.
   The only semantic check is the caller-supplied `contains_sensitive_data`
   boolean. There is no taint lattice, no label join, no flow-to-sink rule, no
   propagation between actions. "Provenance-aware routing" is not implemented.
2. **The policy ablation is tautological.** `input_filter_only_committed` is
   literally `not scenario["detector_blocked"]`, and all ten scenarios hard-code
   `detector_blocked: False`. The 8/8-versus-0/8 table is a restatement of the
   scenario definitions, not a measurement. Five of the eight attacks are denied
   solely because the tool is absent from the policy.
3. **The capability language cannot express the examples in the threat model.**
   `constrained_arguments` supports only literal allowlists. You cannot write
   "any address at `@morgott.test`", "amount ≤ 100", or "path under
   `/workspace/`". All argument values must be strings. There is no expiry, no
   identity binding, no delegation or attenuation, no revocation, no audit log,
   no cross-action session state.
4. **`reports/routing-baseline.{json,md}` are stale and untracked.** They were
   written against manifest `0744…40ed` (current: `27bd…8f81`) using a
   superseded four-task schema that still references an `aegis_harmfulness`
   source no longer in the manifest.
5. **`failed_attack_attempt` is computed and never used** in
   `routing_baseline.py`.
6. **The `trusted_instruction` channel contributes zero rows to any view** — all
   118,326 are auxiliary.
7. **No test asserts anything about the real corpus.** All 61 tests use
   synthetic fixtures. Invariants like "the uncertain view is 100% label 1" or
   "no train hash appears in dev-test" are enforced by the builder but not
   guarded by a test against the materialized files.

8. **`normalize_text` does not defeat obfuscation, and the leakage hash inherits
   that.** `data.py:261` is
   `" ".join(unicodedata.normalize("NFKC", text).casefold().split())`.
   Verified locally against 13 known evasion techniques: **12 produce a
   different `text_hash` than the unobfuscated baseline.** NFKC folds full-width
   forms, and nothing else. Zero-width characters (U+200B/200C/200D/FEFF), soft
   hyphen, Unicode tag characters, variation selectors, bidi controls, Cyrillic
   homoglyphs, combining diacritics, and single-character spacing all survive.

   The common claim that NFKC strips zero-width characters is false — every
   Cf-category codepoint passes through unchanged.

   Two consequences, and the second is the serious one:
   - The model view is trivially evadable by character spacing, which is the
     exact technique that took Prompt Guard 1 from 100% detection to 99.8%
     attack success.
   - **`text_hash` is the basis for exact deduplication, conflict quarantine,
     and locked-evaluation collision checks.** An obfuscated restatement of a
     dev-test row will not collide with it. The "exact normalized text does not
     cross train/validation/dev-test" invariant therefore holds only for
     non-obfuscated text — in a corpus whose largest positive sources
     (HackAPrompt, Tensor Trust) are competitions where obfuscation *is* the
     game.

   **Do not fix this by editing `normalize_text`.** That would change every
   `normalized_text_sha256` in the manifest and force a full ~20 GB rebuild,
   invalidating every recorded hash. The cheap correct fix is a *second*,
   stricter normalization used only for the overlap audit and as an additional
   model view: strip category Cf, NFD then drop Mn combining marks, map
   confusables through a skeleton, and collapse intra-word single-character
   spacing. The canonical hash stays as-is; the audit gets a companion column.

## 6. What I would do, in order

### P0. Manufacture matched contrast at scale (highest value, cheapest)

This is the one intervention with a perfect track record here, and it has never
been done at more than a few hundred rows.

You own ~800k benign task-oriented dialogue rows (Taskmaster, Schema-Guided
Dialogue, BANKING77, MASSIVE, Harper Valley Bank, TAT-QA). Programmatically
inject attacks into them to produce genuinely matched (clean, attacked) pairs
from the *same* source, the same speaker style, the same length distribution.

This directly destroys the source shortcut, because after it every benign-only
source contains both classes. It converts "2 of 19 mixed sources" into "16 of
19." Nothing else on the list changes that number.

Rules that make it evidence rather than augmentation:
- One clean copy per attacked copy, grouped as a pair (the pair-balanced BIPIA
  lesson — the positive-heavy pilot regressed direct and finance scores).
- Known payload spans by construction; prefix/middle/suffix position variation.
- Payload families held out across splits, not just pairs.
- Never inject into a dev-test lineage; fail the recipe on exact collision with
  locked evaluation hashes (the existing runner already does this).
- Report with and without the synthetic pairs, always.

**What this does and does not buy — bounding my own recommendation.** Matched
pairs attack the *shortcut* problem: they make source identity stop predicting
the label. They do **not** buy adversarial robustness, and it would be wrong to
claim otherwise. Nasr et al. (arXiv 2510.09023) give the mechanism: *"training
against a fixed set of pre-computed or weak perturbations does not generalize
and quickly fails under adaptive attacks. Only adversarial training that
performs robust optimization — where perturbations are optimized inside the
training loop — has been shown to yield meaningful robustness."* A generated
pair set is exactly such a fixed, precomputed distribution.

The evidence that this gap is real and large, across every defence measured both
ways: Meta SecAlign 70B goes **0.5% static → 47.3% under GCG → 96% under search
attack**. SecAlign and StruQ go ~0% → 85–95% under an attention-aware white-box
attack. Adversarial fine-tuning on a Llama3-8B agent goes 2% → 57%.

So the honest framing for P0 is: it should fix generalization across sources,
which is morgott's measured failure. It should not be reported as robustness.
Those are different claims, and the attempt-scaling harness is what keeps them
separate.

**This is also the right use for the OpenRouter key**: generate *new* attack
payloads and new benign agent tasks, seeded by taxonomy, not by shipping corpus
text out. Note that `AGENTS.md` requires any remote path to be explicit,
bounded, locally redacted, development-only, and separately reviewed — payload
*generation* satisfies that far more comfortably than corpus *labelling* does.
A cheap model is fine for this; diversity matters more than quality, and every
generated row is a training row, never a label of record.

### P0. Fill the two holes that make whole axes unmeasurable

- **Finance attacks.** Currently zero. Inject transaction-subversion payloads
  into Harper Valley Bank and BANKING77 turns, and into TAT-QA contexts. This
  is the same machinery as above and it makes the flagship claim testable for
  the first time.
- **Benign tool output.** There is no `tool_output` channel in the enum and no
  benign agentic content anywhere. Nemotron gives you 1,272 positives and no
  denominator. Without matched benign tool output, indirect precision is not
  estimable at any threshold.

### P1. Move to agent-level evaluation

This is the strategic recommendation, and it is a genuine disagreement with the
current roadmap ordering, which has containment at P3.

The project's own evidence says input classification does not solve this. The
external evidence agrees: CaMeL reports 77% of AgentDojo tasks solved with
provable security versus 84% undefended — i.e. the architectural defense buys
near-elimination of injection-driven task compromise for ~7 points of utility,
which no classifier in this ledger comes close to matching.

AgentDojo (97 tasks, 629 security cases across banking, email, travel,
workspace) gives you four things you currently lack:

1. A **prospective, uncontaminated** evaluation surface — it is not in your
   corpus, so it cannot be leaked into.
2. An end-to-end metric (attack success rate **and** task utility) instead of
   PR-AUC on benchmarks that public checkpoints trained on.
3. A **banking domain**, which is exactly the gap in §4.3.
4. A setting where the detector's job becomes tractable. Behind a working
   monitor, a sensor at 60% recall and 1% FPR is useful. Measured against
   nothing, 77% recall at 0.1% FPR is unpromotable — which is why nothing has
   been promoted.

Two caveats on this recommendation, both of which narrow it rather than kill it:

- A catalogue of 40 agent-safety benchmarks finds **no ranking concordance
  across evaluation dimensions (Kendall's W = 0.10, p = 0.94)** — benchmark
  choice alone can flip a safety conclusion. So an AgentDojo number is one task
  suite's outcome, not a safety claim. State it that way.
- Per §4.9, running a suite with its *shipped* attacks measures very little.
  Whatever suite is chosen must be run with adaptive attacks and an
  attempt-scaling curve.

**Concrete cheapest path.** Gray Swan's `GraySwanAI/ipi_arena_os` is MIT-licensed
and public: 41 behaviours across tool use (18), coding (15), and browser (8),
with an **OpenRouter backend** already supported — which this workspace has a key
for. It is the artifact behind a 13-model, 272,000-attack public competition, so
its attacks are adaptive by construction rather than a fixed template set. It is
lightly maintained (2 commits), so treat it as a starting corpus of scenarios
rather than a dependency. UK AISI's `inspect_evals` and NIST's
`usnistgov/agentdojo-inspect` are the alternatives if a maintained harness
matters more than adaptive attacks.

Before any of that, close the gaps in §5: make `authorize()` actually consume
provenance, add predicate constraints (prefix/suffix/range/set) so the policy
can express the threat model's own examples, and replace the tautological
ablation with one where the detector is a real variable.

### P2. Only then, more encoder work

And when you do, the two things worth trying are in §7.

## 7. Training strategies: what is worth trying, and what is not

You asked about novel training strategies, custom losses, and new architectures.
My honest read of your own ledger is that **most of that list is the wrong
lever**, and you have already proven it four times: full fine-tune (51.22% FPR),
top-four-layer (35.14%), discriminative LR (40.77%), independent projection
towers (no gain for 4× parameters), symmetric max loss (recall 19.70% → 3.11%),
longer context (BrowseSafe PR-AUC 0.7674 → 0.5636). Meanwhile every data
intervention worked.

So: two exceptions, both motivated by a specific diagnosed failure, and both
cheap on the existing harness.

### 7.1 Energy-based OOD penalty on benign inputs — try this first

Meta's Prompt Guard 2 model card states the objective directly: in addition to
cross-entropy, "we apply a penalty for large negative energy predictions on
benign prompts," citing energy-based out-of-distribution detection. Meta's
stated reason for it is reducing false positives on benign distributions the
model was not trained on.

That is *precisely* morgott's dominant failure mode: benign rows from an unseen
source (MASSIVE, Mind2Web, NotInject, multi-turn benign) get high scores. It
is the one custom loss in the literature aimed at exactly this, and it drops
into `masked_bce_loss` as an extra term, falsifiable in one run against the
existing frozen-head baseline.

Two caveats found after the first draft, both of which shape the experiment
rather than cancel it:

- **Meta has not published the formula.** The model card cites Liu et al.,
  *Energy-Based Out-of-distribution Detection* (NeurIPS 2020) and says only that
  a penalty is applied to large negative energy on benign prompts. A PurpleLlama
  issue asking for the exact formulation has sat unanswered since 2025-06-10.
  So this is an implementation from the cited paper, not a reproduction of
  Meta's recipe, and should be described that way.
- **The objective is implicated in a specific bypass.** Zenity Labs reports that
  verbatim payload duplication raises bypass ~30% on Prompt Guard 2 86M, and
  that the effect did **not** reproduce on ProtectAI, Deepset, or DistilBERT —
  isolating the cause to the training objective rather than the architecture.
  If morgott adopts the penalty, the ablation gate must include a duplication
  robustness check, or it risks importing a known weakness along with the gain.

Still the highest-value objective change available, but predeclare the
duplication check as part of the gate.

### 7.2 Source-adversarial training (gradient reversal) — try this second

You have already *measured* the nuisance variable: source identity explains
87.75% of label entropy, and you built a source classifier that reaches 87.78%
top-1. Standard practice when the nuisance is known and measured is to remove
it from the representation with a gradient-reversal source-discriminator head
(DANN-style), rather than only reporting it.

This is better targeted than the deferred alternatives:
- **Group DRO** is correctly deferred — with current source IDs as groups it
  optimizes benchmark identity.
- **PCGrad** addresses head conflict, which the independent-tower ablation
  already showed is not the binding problem.
- Gradient reversal attacks the actual diagnosed confound.

**Downgraded after review — read before spending time on this.** "When
Benchmarks Lie" (arXiv 2602.14161) runs exactly this experiment class across 18
pooled injection datasets with leave-one-dataset-out evaluation. It finds
standard cross-validation scores **8.0–16.5 AUC points higher than LODO**, that
a **dataset-identity classifier reaches 96.6% accuracy** — an independent
replication of §3's finding on a different corpus — and, critically, that
**adversarial training, subspace projection, reweighting, and class balancing
all failed to close the gap**.

Gradient reversal sits squarely in that family. The paper does not name DANN
specifically, so this is suggestive rather than conclusive, but the prior is now
against it. Treat §7.2 as a low-confidence option behind §7.1, not as the second
thing to try. If run at all, run it as a one-variable ablation with LODO
selection and kill it if fold-macro PR-AUC drops.

The same paper's positive recommendation is one morgott should adopt regardless:
**leave-one-dataset-out is the correct evaluation protocol** for a corpus that
pools ~30 sources, and it is stronger than the current source-heldout folds.

### 7.3 Reframe indirect detection — the one architecture change I would make

Not a bigger model. A different input.

Indirect injection is **relational**: the question is whether this content
contains an instruction that conflicts with, or is unrelated to, what the user
actually asked. A document-only classifier cannot represent that question,
which is why BrowseSafe sits near 0.77 PR-AUC and why the corpus needed 76,202
exact label conflicts quarantined — the same text is genuinely both labels
depending on context.

Two concrete moves, in order of cost:

1. **Span/window supervision instead of document supervision.** You already
   generate known payload spans. Token-level BIO tagging over windows gives
   dense supervision, handles long documents natively, and answers the
   operationally useful question ("where is it") instead of the ill-posed one
   ("does this 107 KB page contain something"). The document score becomes an
   aggregate of span scores, which is also what inference already does.
2. **Cross-encode (user task, content).** AgentDojo and BIPIA both supply the
   task. This is where the P1 recommendation pays a second dividend: it gives
   you the task field that BrowseSafe lacks, which is the actual missing input.

### 7.4 Activation probing — evaluated and ruled out (deployment decision, 2026-07-26)

**Decision: not available. morgott targets applications calling closed API
models, so there is no white-box access to the served model. This section is
retained as the rationale for a closed question, not as a proposal.**

Activation probing requires reading the internals of the model being defended.
morgott is deliberately architected the other way: an external text classifier
with its own encoder, independent of the application's LLM. Probing works when
the application serves open weights (vLLM, Llama, Qwen, Mistral); it does not
work behind Claude, GPT, or Gemini APIs. Probes also do not transfer across
backbones, so it would be one probe per served model.

This is worth recording because the numbers are the strongest in this review,
and they land on the axis morgott cannot currently solve — so the question will
come back if the deployment target ever changes:

- **InstructDetector** (arXiv 2505.06311, EMNLP 2025 Findings) reads hidden
  states and gradients from intermediate layers on the external data block.
  Verified from the abstract: **99.60% in-domain and 96.90% out-of-domain
  detection accuracy, and attack success rate reduced to 0.03% on BIPIA.**
  Reported training set is ~200 examples, with an ablation claiming >95% at 50
  (PDF-extracted, not verified from the abstract).
- **TaskTracker** (arXiv 2406.00799, Microsoft) fits linear probes on the
  *activation delta* before versus after the external block, trained only on
  benign task-drift contrasts, and reportedly outperforms Prompt Guard and
  Azure Prompt Shields.
- **ESLD** (arXiv 2605.18918) reports the gain concentrating on indirect
  injection specifically — guard LLMs near chance (0.50–0.72) while the latent
  probe reaches 0.91–0.92.

Why this matters here in particular: BIPIA ASR of 0.03% and indirect balanced
accuracy above 0.90 are on the axis where morgott's best is 0.7674 PR-AUC and
18.07% recall. And the reported data requirement — hundreds to a few thousand
examples — makes §3's "2.24% reachable" problem largely irrelevant. That
literature's own conclusion is that **distribution diversity, not volume, is the
binding constraint**, which is the same conclusion §3 reached independently.
P0 survives unchanged: matched pairs *are* the diversity.

**The counterweight, which deserves equal weight.** "When AUC 0.998 Is Not
Enough" (arXiv 2606.22864, EvalMG '26 at SIGIR 2026) states the thesis
verbatim: *"a high probing AUC on a clean-vs-attack split is not, on its own,
evidence of malicious-content detection."* Its reported findings (PDF-extracted,
decimals not independently verified) are that on the **tool-output surface the
probe reaches only ~0.77** — the weakest surface it measured; that a **four-scalar
baseline** over step index, horizon, and prompt length reaches ~1.00 on the
text-side surfaces, beating a 107,520-dimensional hidden-state probe; that a
**scrambled meaningless-text control** scores ~0.998 with real-versus-scrambled
discrimination at ~0.49; and that cross-surface transfer collapses to ~0.51.

That is morgott's own source-shortcut finding reappearing in the probe
literature — a headline number produced by a nuisance correlate rather than the
intended signal. The relevant asymmetry is that **morgott is unusually well
equipped to do this correctly**, because nuisance baselines, scrambled controls,
shuffled-label sanity checks, and held-out transfer are already the house style
here and are largely absent from that literature.

There is also a stated open niche: no published work trains an activation probe
to *localize* an injected instruction inside a long tool-output span in a live
agent trajectory. That is precisely morgott's weakest axis, and it is the same
span-localization reframing already proposed in §7.3 — which remains available,
because §7.3 operates on text and needs no model internals.

**What carries over despite the ruling-out.** Two findings from this literature
survive and apply directly to the external-classifier architecture:

1. **Diversity beats volume.** Multiple independent results (probe training
   plateaus, Anthropic's "training on more examples did not improve probe
   performance," DeepMind's heavy subsampling plus diverse mixtures) converge on
   the same conclusion §3 reached from the manifest. This reinforces P0.
2. **The nuisance-baseline protocol is mandatory.** Before believing any
   headline number, run a scalar baseline over length and position, a
   scrambled-text control, a shuffled-label sanity check, and a held-out
   transfer test. That protocol is borrowed from 2606.22864 and costs almost
   nothing.

### 7.5 What I would skip

- **RL** — already correctly rejected; no environment, no reward, no preference
  signal. (This changes only if you go to AgentDojo, where a real environment
  and a real outcome signal exist. Even then it is not the next step.)
- **Focal loss / class weighting** — the failure is shortcut, not imbalance.
- **LoRA / bigger backbones / longer context** — all three were tried or are
  blocked by missing denominators.
- **More sources of the same shape.** Adding a 20th single-class source makes
  the shortcut worse, not better.

## 8. On fine-tuning Prompt Guard 2

Short answer: **not as the next move, and not the way it is usually done.**

The facts, from Meta's model card: Prompt Guard 2 86M is mDeBERTa-base (the 22M
is DeBERTa-xsmall), 512-token window, Llama 4 Community License, trained with
cross-entropy plus the energy penalty above. Meta does recommend it: "Fine-tuning
Prompt Guard on domain-specific prompts improves accuracy and reduces false
positives," with a llama-cookbook tutorial. For long inputs they say to "split
prompts into segments and scan them in parallel." Stated limitations include
vulnerability to adaptive attacks and variance with application-specific
distributions. Meta dropped Prompt Guard 1's injection sublabel because they
"found this objective too broad to be useful."

So the idea is reasonable on its face. Four arguments against it being *next*:

1. **You would be destroying your best measured result.** Your strongest
   indirect number is the OR of ModernBERT and Prompt Guard: 59.16% recall at
   1.68% FPR on long untrusted rows, versus ~33% each alone. That gain exists
   *because* the two models were trained on different data with different
   objectives. Fine-tuning Prompt Guard on the morgott corpus pulls it into the
   same shortcut basin as your own heads, and the decorrelation that produces
   the gain is exactly what you would be removing. The ensemble is the asset;
   the checkpoint is replaceable.

2. **The result would be unmeasurable.** Meta states the training data is "a
   mix of open-source datasets," i.e. plausibly the same public corpora you
   train and evaluate on. You already quantified this class of problem: 54.28%
   of Rogue Security overlapped your corpus. Fine-tune a possibly-contaminated
   model on contaminated data and evaluate on an already-open dev set, and you
   cannot separate "learned the task" from "re-memorized the overlap."

3. **Your own evidence predicts the failure mode.** Every encoder fine-tuned on
   this corpus overfit source style. Prompt Guard 2 is a smaller encoder, and
   you would be fine-tuning it on the *same* data whose shortcut structure
   caused those failures. There is no reason to expect a different outcome, and
   §3 explains why: there is not enough matched contrast in the corpus to fine-
   tune against.

4. **The ontology fights you.** Prompt Guard 2 is deliberately binary and
   deliberately narrow. Morgott's central thesis is that direct/indirect/
   jailbreak/harmful must stay separate axes. Fine-tuning it means either
   flattening that or bolting on masked heads — at which point you have rebuilt
   what you already have on ModernBERT/mmBERT, on a weaker backbone.

**What to do instead, in order:**

- **Keep the pristine checkpoint** as an independently-trained ensemble member
  and external reference. It is currently your best indirect sensor: 0.7838
  zero-shot BrowseSafe PR-AUC, which beats your best trained model's 0.7674.
  That is a slightly embarrassing but very useful fact.
- **Steal the loss, not the weights.** §7.1. You get the part of Prompt Guard
  that is aimed at your actual failure, without inheriting its contamination or
  collapsing your ensemble diversity.
- **Fit a stacker, not a fine-tune.** Freeze all members and train a small
  calibration layer over `[PG2 score, ModernBERT heads, mmBERT heads, provenance,
  length, window count]`. This is cheap, preserves decorrelation, and directly
  optimizes the OR-rule you currently hand-pick. It also lets provenance enter
  the score, which text alone cannot supply.
- **Revisit fine-tuning later, as LP-FT, on the indirect axis only** — after
  §6 P0 supplies matched data and §6 P1 supplies a prospective test. At that
  point it is a legitimate ablation rather than the main line. Keep an untouched
  copy either way.

One genuine argument in favour that survives all of the above: PG2 86M is
mDeBERTa, i.e. multilingual, and §4.4 says you cannot measure multilingual FPR
at all. If multilingual coverage becomes a goal, PG2 is a better starting point
than anything you have. But that argues for *evaluating* it multilingually
first, which requires the multilingual benign data you do not yet have.

## 9. Summary

- The corpus is finished and large, and its size is not the useful quantity.
  2.24% of it is reachable by the recipe that produced your best model, and
  only 2 of 19 training sources contain both labels.
- The direct axis is in decent shape. The indirect axis rests on one ill-posed
  benign distribution. The finance axis has no attacks. The multilingual axis
  has no benign data and a multilingual model selected on it.
- Every number is single-shot against a static corpus, while the labs this work
  would be compared against now report attack success as a curve over attempt
  count. That gap probably matters more than any remaining recipe choice.
- Activation probing reports far better indirect numbers than anything here, on
  hundreds of examples rather than millions — but it needs white-box access to
  the served model, and its own literature has the same shortcut pathology
  morgott already knows how to detect. It is a fork in the deployment model, not
  a drop-in, and it should be decided as one.
- Every data intervention has worked; every architecture and loss intervention
  except pairing has failed. Act on that asymmetry.
- Build matched pairs at scale, fill the finance and tool-output holes, then go
  to AgentDojo and make the reference monitor real — including making
  `authorize()` actually read the provenance it already validates.
- Do not fine-tune Prompt Guard 2 next. Take its loss function, keep its
  weights frozen and independent, and stack on top of it.
