# Checkpoint selection for a low-false-positive detector

Status: design decision for a future scientifically gated training campaign. It
does not select or promote a model.

## Decision

Checkpoint selection must be a constrained, component-level decision, not the
minimum of a blended validation loss. Binary cross-entropy remains useful for
training and for screening a small number of checkpoints, but it cannot trade
away a finance or channel false-positive limit.

Use three mutually disjoint development roles:

1. a repeatedly viewed checkpoint screen;
2. a threshold-calibration panel; and
3. a once-opened selector-evaluation panel.

Freeze every recipe, context, seed, checkpoint, panel digest, threshold rule,
and candidate count before opening the third role. Select only among candidates
that pass simultaneous false-positive gates. Among those candidates, maximize
the worst supported attack-family recall bound. Keep a separate locked test for
reporting after the model and operating thresholds are frozen.

This replaces both the registered
`0.5 * (Morgott row-micro BCE + PromptShield BCE)` rule and the later context
campaign's `0.5 * (Morgott source-macro BCE + PromptShield BCE)` rule for
future campaigns. Both old rules remain provenance for runs that already used
them.

## Statistical unit

All counts use the independent security component: a conversation, task,
document, repository, or lineage group as declared by the source contract.
Reduce windows and turns to one component decision before counting errors.
Rows, windows, or mutations from the same component are not independent
Bernoulli observations.

Strata use trusted `(input_channel, semantic_source_family, label)` metadata.
Do not classify an entire mixed-label source as either "missed attacks" or
"false flags." An unknown channel is reported separately and cannot borrow a
direct-user or untrusted-content threshold.

## Three-panel protocol

### A. Checkpoint screen

This role may be evaluated at every frozen update. It never provides a final
metric or an operating threshold. Per recipe, retain no more than three unique
snapshots:

- lowest attack-family-macro BCE;
- lowest clean-family-macro BCE; and
- lowest worst `(channel, semantic family, label)` BCE.

Break an exact screening tie in favor of the earlier update. Deduplicate when
more than one rule chooses the same snapshot. This bounds the opportunity to
select a lucky checkpoint while retaining visibly different failure modes.

### B. Selector threshold calibration

After screening, freeze the candidate hashes and total candidate count `K`.
`K` includes the incumbent and every recipe, context, seed, and checkpoint to
be eligible. Choose one component-level threshold for each candidate and each
trusted channel using the Neyman--Pearson order-statistic construction. Do not
use the largest threshold whose empirical false-positive rate happens to be at
or below one percent.

The order-statistic construction must control threshold violation
simultaneously across `K * C` candidate-channel pairs, where `C` is the number
of trusted channels. Freeze all candidate-threshold pairs before opening panel
C. Adding a candidate after this point starts a new protocol.

### C. Selector evaluation

Score all frozen candidate-threshold pairs once on the same components. This
panel must be row-, normalized-text-, strict-text-, near-, lineage-, and
repository-disjoint from fitting and panels A and B. Its scorer must verify the
identities rather than accept a boolean attestation.

Compute simultaneous one-sided exact-binomial bounds for every clean gate and
required attack family. The panel can select one winner because the confidence
bounds cover the complete frozen candidate family. It cannot be reused to tune
a loss, add a context length, change a threshold, or admit another checkpoint.

If the selected model needs a newly calibrated threshold, use a fourth fresh
calibration panel. Otherwise carry forward panel B's frozen threshold.

## Multiplicity and power

Preregister a total failure budget `delta = 0.05`, divided as follows:

- `delta_threshold = 0.01` for panel-B threshold construction;
- `delta_clean = 0.02` for panel-C clean gates; and
- `delta_attack = 0.02` for panel-C attack-family comparisons.

For `K` candidates, `C` thresholded channels, `G` clean gate cells per
candidate, and `F` required efficacy cells (attack-family recall plus the
matched-pair both-correct metric), use:

```text
panel-B violation allowance per threshold = delta_threshold / (K * C)
panel-C FPR UCB tail probability          = delta_clean / (K * G)
panel-C recall LCB tail probability       = delta_attack / (K * F)
```

These Bonferroni bounds are deliberately simple and auditable. A preregistered
Holm or Learn-then-Test procedure may replace them, but a per-candidate
correction over channels alone may not.

For example, with `K = 7` and `G = 4`, a clean cell with zero observed errors
needs at least 721 independent components for its simultaneous one-sided
Clopper--Pearson upper bound to establish an FPR no greater than one percent:

```text
ceil(log(0.02 / (7 * 4)) / log(0.99)) = 721
```

Zero false positives on a smaller cell is inconclusive, not a passing gate.
Replace a generic minimum such as 300 attack examples with a prospective
binomial power calculation. For every attack family, freeze a minimum useful
recall `r0`, improvement margin `Delta`, multiplicity-adjusted type-I error
`delta_attack / (K * F)`, and at least 80% power under `r0 + Delta`. An
underpowered family fails closed.

Do not extend training or change the checkpoint grid after inspecting panel C.
If adaptive or indefinite monitoring is required, use anytime-valid confidence
sequences; freezing the grid is the simpler default.

## Hard gates and deterministic order

A candidate is feasible only if all of these conditions hold:

- population identities, digests, disjointness, and denominators match the
  frozen protocol;
- the panel-B threshold construction is valid for every trusted channel;
- every ordinary-clean and long/untrusted-clean channel has a simultaneous
  panel-C FPR upper bound no greater than one percent;
- every finance channel has both zero observed false-positive components and a
  simultaneous FPR upper bound no greater than one percent;
- harmful-but-not-subversive traffic passes its off-target clean gate;
- every required attack family is present and prospectively powered, including
  direct transfer, indirect documents, obfuscation/evasion, and matched
  boundary behavior; and
- any frozen latency, memory, and failure-rate limits pass.

If no candidate is feasible, return `no_feasible_candidate`. Otherwise:

1. Sort each candidate's attack-family recall lower bounds from worst to best
   and lexicographically maximize that vector.
2. Maximize the simultaneously covered matched-pair both-correct lower bound.
3. Within a preregistered practical-equivalence tolerance, prefer lower serving
   cost, then shorter context, then the earlier update.

Use `0.005` absolute recall (one half percentage point) as that tolerance
unless a different value is frozen before any candidate is trained.

Do not use pooled recall, full AUROC, BCE, or a weighted sum to break these
ties. They remain descriptive diagnostics. A Pareto frontier may also be
reported, but it does not override the deterministic decision.

## Loss and weighting decision

Keep binary cross-entropy as the baseline. Do not begin with focal loss,
asymmetric loss, effective-number class weighting, a global-AUC surrogate, or
a calibration regularizer:

- focal loss was designed to suppress abundant easy negatives in dense object
  detection;
- asymmetric loss targets sparse multi-label imbalance;
- effective-number weighting addresses class-count redundancy, not Morgott's
  source and channel shortcut;
- global AUC spends most of its objective outside the zero-to-one-percent FPR
  region; and
- calibration penalties cannot repair source shift. An independent threshold
  panel is still required.

The smallest justified custom candidate is regularized group DRO on the
canonical primary BCE only. Define coarse trusted
`(channel, semantic family, label)` training groups in advance. Preserve
within-group lineage de-duplication and multiplicity control; replace the fixed
top-level canonical mixture with the group-DRO aggregation. Keep PromptShield,
the matched-pair BCE and ranking term, their coefficients, the optimizer, and
all other training settings unchanged. Merge or omit an underpowered training
group by a training-only frozen rule rather than creating unstable tiny groups.

Naive group DRO is not enough in an overparameterized network. Its primary
study found that regularization or early stopping is necessary for worst-group
generalization. Panel A supplies the early-stopping screen; do not also sweep a
regularization strength in the first comparison.

A partial-AUC hybrid restricted to the zero-to-one-percent FPR region is a
later experiment, not a parallel first sweep. It needs a large fit-only
negative pool because the low-FPR tail of one ordinary minibatch is too sparse.
Retain BCE in that hybrid and never optimize whole AUC as a proxy for the
operating region.

## Minimal next experiment

After the three disjoint development roles are available and frozen, run one
matched 1,024-token A/B:

- GPU 0: a fresh no-harm BCE control;
- GPU 1: the same recipe with only canonical group-DRO aggregation changed.

Use identical hardware, manifest and pair-archive digests, seed, LoRA rank,
three-epoch update budget, optimizer, batch/token budget, validation grid,
PromptShield loss, pair loss, and no harmful auxiliary head. A training-only
numerical canary is allowed; there is no validation hyperparameter sweep.

Panel A retains at most three checkpoints per arm. With one fixed incumbent,
the frozen family therefore contains at most `K = 7` candidates. Panels B and C
then apply the protocol above. One seed is development evidence only. If group
DRO wins, seed stability requires a separately preregistered confirmation with
fresh selection evidence; never retain the luckiest seed.

Reusing the current frozen 1,024-token run as the control is cheaper, but it is
not a clean causal loss ablation because its runtime history spans different
hardware and resume conditions. Label that comparison as a promotion screen,
not an A/B, if compute cost makes the fresh control undesirable.

Do not add 2,048 tokens to this loss experiment. A 2,048-token run is a separate
single-variable context study after the 1,024-token recipe is chosen. It needs
its own tail audit, adequately powered long clean and attack components,
matched train/evaluation context, frozen candidate budget, and unconsumed
selector evidence.

## Existing 17,000 and 18,500 snapshots

The two frozen snapshots can be compared descriptively on identical current
panels with corrected component-level confidence intervals. That does not
retrofit prospective selection:

- the current canonical, PromptShield, SEP, finance, reserve, and long-code
  panels are already open development evidence;
- update 18,500 was retained after repeated BCE monitoring; and
- update 17,000 was frozen for a different registered comparison.

Neither snapshot is therefore an unbiased "best checkpoint" or evidence of a
portable one-percent production FPR. Both may enter a future protocol without
retraining if genuinely new panels A, B, and C are collected and frozen before
their scores are viewed.

## Remove the selector prototype

The entire `experiments/checkpoint_selection/` prototype and
`tests/test_checkpoint_selector.py` are rejected YAGNI, not the implementation
of this design. They can be removed after this report is retained because no
maintained scorer or manifest produces their input and no completed decision
depends on them.

Repairing that prototype in place would preserve several unsafe assumptions:

- its clean bounds correct over two channels inside each block, but omit the
  number of candidates and the multiplicity across all clean blocks;
- its attack bounds omit candidate multiplicity;
- its threshold-calibration counts do not bind an NP order-statistic selection
  rule;
- a generic 300-component attack floor is not a power calculation;
- disjointness is accepted as booleans rather than verified from identities;
  and
- source-macro BCE can choose between otherwise feasible candidates before the
  earlier-update tie-break.

Implement the eventual selector only after the three panel manifests and the
component-level scorer exist. Bind it to those artifacts, freeze `K`, and test
the complete end-to-end statistical contract rather than another free-standing
aggregate JSON schema under `experiments/`.

## Semalith boundary

Semalith's contamination audit, slice-level error analysis, and compact encoder
are useful comparison ideas. Do not copy its row-stratified five-percent
validation, shared 22-class harm/BFSI/prompt-injection head, auxiliary-loss grid
on that same validation, 256-token truncation, post-hoc binary mapping, or
benchmark-driven data iteration. Those choices do not implement Morgott's
channel-specific low-FPR contract. If access is granted, freeze a Semalith
prompt-injection score mapping before scoring the exact Morgott components;
published results on overlapping benchmark names are not an apples-to-apples
comparison.

## Primary sources

- Angelopoulos et al., [Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control](https://arxiv.org/abs/2110.01052).
- Tong, Feng, and Li, [Neyman--Pearson classification algorithms and NP
  receiver operating characteristics](https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/).
- Cawley and Talbot, [On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
- Howard et al., [Time-uniform, nonparametric, nonasymptotic confidence
  sequences](https://arxiv.org/abs/1810.08240).
- Sagawa et al., [Distributionally Robust Neural Networks for Group Shifts: On
  the Importance of Regularization for Worst-Case
  Generalization](https://arxiv.org/abs/1911.08731).
- Narasimhan and Agarwal, [A Structural SVM Based Approach for Optimizing
  Partial AUC](https://proceedings.mlr.press/v28/narasimhan13.html).
- Lin et al., [Focal Loss for Dense Object
  Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).
- Cui et al., [Class-Balanced Loss Based on Effective Number of
  Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).
- Ridnik et al., [Asymmetric Loss for Multi-Label
  Classification](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).
- Guo et al., [On Calibration of Modern Neural
  Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Kumar, Sarawagi, and Jain, [Trainable Calibration Measures for Neural
  Networks from Kernel Mean Embeddings](https://proceedings.mlr.press/v80/kumar18a.html).
- Addagada, [Semalith
  v1.4](https://arxiv.org/html/2607.22545v1).
