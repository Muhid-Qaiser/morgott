# PromptShield paper and release audit

This note verifies the PromptShield training recipe, split intent, evaluation protocol, and reported baselines against primary sources only.
Claims labelled **Paper** or **Release/code** are stated by those sources, while claims labelled **Inference for morgott** are conclusions drawn for this project.

## Primary sources

- **Paper:** Dennis Jacob et al., [PromptShield: Deployable Detection for Prompt Injection Attacks](https://arxiv.org/html/2501.15145), especially Sections 2.2, 2.4, 3.1.3, 3.2.1, 3.2.2, 4, 5.2-5.5, Appendix A.3, and Tables 2 and 4-9.
- **Official dataset release:** [hendzh/PromptShield at pinned revision `a5234cb1f5cdb256600cab64b8c961195b5e8404`](https://huggingface.co/datasets/hendzh/PromptShield/tree/a5234cb1f5cdb256600cab64b8c961195b5e8404).
- **Historical dataset revision:** [hendzh/PromptShield at complete revision `514ff8f73f1cbb9b0e1330d1e7d27129b161bedb`](https://huggingface.co/datasets/hendzh/PromptShield/tree/514ff8f73f1cbb9b0e1330d1e7d27129b161bedb) is used only to audit release drift.
- **Official source repository:** [wagner-group/PromptShield at commit `bc03ac195670700ed1cf684cafe46154623e71b9`](https://github.com/wagner-group/PromptShield/tree/bc03ac195670700ed1cf684cafe46154623e71b9).

## Task and split intent

- **Paper, Section 2.2:** The work focuses on indirect prompt injection in user-controlled or third-party data rather than direct misuse of an LLM prompt.
- **Paper, Section 2.4:** Multi-turn interaction and function calling are outside the paper's taxonomy and experiments.
- **Paper, Table 2:** Training benign data comes from UltraChat, Alpaca, and IFEval, while evaluation benign data comes from LMSYS, Databricks Dolly, Natural Instructions, and Synthetic Python Problems.
- **Paper, Table 2:** Training attacks come from FourAttacks applied to Alpaca plus HackAPrompt, while evaluation attacks come from FourAttacks applied to Databricks Dolly and Synthetic Python Problems plus OpenPromptInjection.
- **Paper, Section 3.1.3:** The source datasets used for training and evaluation are mutually exclusive by construction, and the authors explicitly describe the evaluation split as measuring OOD performance.
- **Paper, Section 3.1.3 and Appendix A.2:** The generated training attacks use ten link phrases and the generated evaluation attacks use eleven different link phrases to discourage phrase memorization.
- **Release/code:** The official Hugging Face release exposes `train`, `validation`, and `test` splits with only a flattened `prompt` and binary `label` as task fields.
- **Release/code:** The [dataset card](https://huggingface.co/datasets/hendzh/PromptShield/blob/a5234cb1f5cdb256600cab64b8c961195b5e8404/README.md#L18-L29) assigns training to fitting, validation to hyperparameter tuning and early stopping, and test to evaluation.
- **Release/code:** Historical revision `514ff8f73f1cbb9b0e1330d1e7d27129b161bedb` contains 18,996 training rows with 9,504 benign and 9,492 positive labels, 1,000 validation rows with 497 benign and 503 positive labels, and 24,000 test rows with 17,513 benign and 6,487 positive labels.
- **Release/code:** The pinned release contains 18,909 training rows with 9,457 benign and 9,452 positive labels, 1,000 validation rows with 497 benign and 503 positive labels, and 23,516 test rows with 17,030 benign and 6,486 positive labels.
- **Inference for morgott:** The current release is not byte-for-byte the approximately 20,000-row training and 24,000-row evaluation artifact described by the paper, so reproducing Table 4 requires identifying and pinning the evaluated revision rather than silently using current `main`.
- **Inference for morgott:** PromptShield train and test are source-disjoint within the release in the paper's intended sense.
- **Inference for morgott:** A model also fitted on Morgott cannot claim source-OOD PromptShield performance because known LMSYS-family overlap and missing row-level PromptShield provenance prevent proving independence from the complete fit.
- **Inference for morgott:** The flattened release does not preserve trusted channel, payload span, component source, or attack-family lineage, so it cannot supply Morgott's direct-versus-indirect subtype provenance.

## Training recipe stated by the paper

- **Paper, Section 3.2.1:** The baseline training sample contains 20,000 English examples with approximately 10,000 benign and 10,000 injection examples.
- **Paper, Section 3.2.1 and Appendix A.3:** Approximately 1,000 random examples are isolated from the training data for validation and checkpoint selection.
- **Paper, Section 4:** Training examples are augmented by independently inserting one to three newline delimiters before the application prompt, before its input data, and after its input data.
- **Paper, Section 4:** Llama models of at least one billion parameters are fine-tuned with LoRA for three epochs at an initial learning rate of `2e-4`, with early stopping when validation performance plateaus.
- **Paper, Section 4:** FLAN models below one billion parameters are fully fine-tuned for three epochs with cross-entropy loss and an initial learning rate of `5e-5`, with early stopping.
- **Paper, Section 4:** DeBERTa is trained using the FLAN procedure except for an initial learning rate of `5e-6`.
- **Paper, Section 4:** Cross-entropy is stated explicitly for FLAN and inherited by DeBERTa's "same as FLAN" procedure, but the paper does not name the Llama LoRA objective.
- **Paper, Appendix B.1:** Llama fine-tuning and evaluation use a fixed system prompt that requests a single binary output token, and an arbitrary output token is mapped to benign.
- **Paper:** For the Llama LoRA runs, the paper does not specify rank, alpha, target modules, adapter dropout, batch size, optimizer, weight decay, scheduler, warmup, maximum sequence length, random seed, early-stopping patience, or the exact checkpoint-selection statistic.
- **Release/code:** [`general_finetune.py`](https://github.com/wagner-group/PromptShield/blob/bc03ac195670700ed1cf684cafe46154623e71b9/general_finetune.py#L42-L97) implements three epochs, cross-entropy, AdamW, a linear schedule with zero warmup, validation loss measurement, and one saved checkpoint per epoch.
- **Release/code:** The same script defaults to batch size four and `5e-6`, but these command-line defaults are implementation details rather than paper-wide recommendations.
- **Release/code:** The checked-in [`finetuning.slurm`](https://github.com/wagner-group/PromptShield/blob/bc03ac195670700ed1cf684cafe46154623e71b9/scripts/finetuning.slurm#L16-L31) invokes FLAN-T5-large at `5e-7`, which conflicts with both the script default and the paper's stated FLAN rate of `5e-5`.
- **Release/code:** The visible three-epoch loop saves all epoch checkpoints and does not implement the early-stopping condition stated in the paper.
- **Release/code:** [`generate_val_split.py`](https://github.com/wagner-group/PromptShield/blob/bc03ac195670700ed1cf684cafe46154623e71b9/generate_val_split.py#L49-L61) uses NumPy seed `12345` to sample 1,000 validation examples after its length filter.
- **Release/code:** [`random_newlines.py`](https://github.com/wagner-group/PromptShield/blob/bc03ac195670700ed1cf684cafe46154623e71b9/data/training_data/2024-12-07/random_newlines.py#L24-L32) defines the paper's one-to-three-newline transform, but the calls in that visible export script are commented out at [lines 53-66](https://github.com/wagner-group/PromptShield/blob/bc03ac195670700ed1cf684cafe46154623e71b9/data/training_data/2024-12-07/random_newlines.py#L53-L66).
- **Release/code:** Every prompt in the current train and validation JSON begins with one to three newline characters, whereas 23,491 of 23,516 test prompts begin with none, so the published fitting files already materialize at least the leading-newline augmentation.
- **Inference for morgott:** The public repository does not provide one clearly identified turnkey command that reproduces every paper-stated training detail.
- **Inference for morgott:** The released prompts can be consumed with their materialized newlines, but the paper's three-position transformation cannot be regenerated or independently audited exactly because the original application-prompt and input-data boundaries are not released separately.

## Evaluation and threshold protocol

- **Paper, Section 4:** Evaluation uses approximately 24,000 examples from source datasets that are disjoint from the training sources.
- **Release/code:** The current pinned test has 23,516 rows, while the historical complete revision has 24,000, so the paper's approximately 24,000-row description does not identify the current artifact unambiguously.
- **Paper, Sections 3.2.2 and 4:** The reported metrics are ROC AUC and TPR at target FPR values of 1%, 0.5%, 0.1%, and 0.05%.
- **Paper, Section 3.2.2:** The authors cache scores on the evaluation split, construct its ROC curve, use linear interpolation to seek a threshold within 25% of the target FPR, and apply iterative bisection if necessary.
- **Paper, Section 3.2.2:** The authors acknowledge that this calibrates thresholds on test data and explicitly recommend that future work select thresholds on validation data instead.
- **Inference for morgott:** Table 4 values are valid same-test empirical ROC operating points, but they are not evidence for a threshold selected before seeing the test distribution.
- **Inference for morgott:** Morgott should report validation-selected threshold performance as primary and retain same-test TPR-at-FPR only as a clearly labelled comparison with the paper.
- **Inference for morgott:** With only 497 released validation negatives, empirical FPR resolution is approximately 0.201% per false positive and cannot directly resolve a 0.1% operating point without a larger independent benign calibration set.

## Reported Table 4 baselines

The following values are transcribed from [Paper Table 4](https://arxiv.org/html/2501.15145), and each TPR is measured at a threshold derived from the same evaluation split.

| Detector | Base model | AUC | TPR@1% | TPR@0.5% | TPR@0.1% | TPR@0.05% |
|---|---|---:|---:|---:|---:|---:|
| PromptGuard | mDeBERTa-v3-base | 0.874 | 12.78% | 12.43% | 9.39% | 1.54% |
| ProtectAI v1 | DeBERTa-v3-base | 0.646 | 7.05% | 3.36% | 0.00% | 0.00% |
| ProtectAI v2 | DeBERTa-v3-base | 0.705 | 1.97% | 1.34% | 0.00% | 0.00% |
| InjecGuard | DeBERTa-v3-base | 0.765 | 20.37% | 16.30% | 6.61% | 4.32% |
| Fmops | DistilBERT | 0.754 | 13.00% | 8.39% | 2.10% | 1.48% |
| PromptShield | DeBERTa-v3-base | 0.976 | 43.22% | 40.50% | 31.45% | 0.00% |
| PromptShield | Llama-3.1-8B-Instruct | 0.998 | 94.80% | 87.80% | 65.33% | 47.53% |

- **Paper, Table 4 footnote:** A zero marked by the paper means no threshold other than `1.0` achieved the requested FPR.

## Ablation findings and recommendations

- **Paper, Table 6:** On Llama-3.1-8B-Instruct, increasing training data from 1,000 to 20,000 examples raises TPR@1% FPR from 62.04% to 94.80% and TPR@0.05% FPR from 20.89% to 47.53%.
- **Paper, Section 5.4:** The authors conclude that more training data is particularly valuable in the low-FPR tail, while acknowledging that the 1K, 5K, 10K, and 20K results come from one architecture.
- **Paper, Tables 7-9:** Adding conversational benign data lowers false-positive rates on conversational evaluation data while modestly reducing some very-low-FPR TPR values on application-structured data.
- **Paper, Section 5.5:** The authors recommend the full mixture for a general-purpose detector and application-structured-only training when the detector is known to serve only a client application.
- **Paper, Table 5:** Larger models generally perform better at low FPR, but FLAN-T5-base outperforms the larger Llama-3.2-1B at several operating points, so architecture matters in addition to parameter count.
- **Inference for morgott:** The paper supports using balanced attack and hard-benign data, explicit low-FPR measurement, validation checkpointing, and a larger benign calibration denominator.
- **Inference for morgott:** The paper does not test or recommend top-k tail penalties, energy objectives, focal loss, pair-ranking loss, mmBERT, or ModernBERT.
- **Inference for morgott:** The paper's use of LoRA for billion-parameter Llama models does not establish that LoRA is better than full fine-tuning or frozen-head training for mmBERT or ModernBERT.
- **Inference for morgott:** A paper-aligned encoder control at this scale would use ordinary cross-entropy and full fine-tuning, while a LoRA run on mmBERT or ModernBERT should be labelled a separate architecture ablation.

## Bottom line for morgott

- **Inference for morgott:** PromptShield training can test whether benchmark-matched data moves the low-FPR tail, but its test remains intentionally OOD and already-consumed development data.
- **Inference for morgott:** The closest paper-faithful control is balanced PromptShield training, cross-entropy, three epochs, validation checkpoint selection, and a `5e-6` initial learning rate for the DeBERTa control.
- **Inference for morgott:** The released fields are insufficient for exact newline-augmentation or subtype-provenance reproduction, and the paper leaves several optimizer and LoRA details unspecified.
- **Inference for morgott:** No PromptShield result should be described as an independent final test, a production-calibrated operating point, or evidence specifically about finance or Web3 traffic.

## Morgott bounded experiment

On 2026-07-28 the repository owner authorized one artifact-only exception to the canonical exclusion decision.
The experiment uses the release's train split outside the canonical corpus, uses validation only for checkpoint selection, and treats test as already-consumed PromptShield-internal source-disjoint development.
It is not source-OOD relative to the complete Morgott plus PromptShield fit.
It does not reproduce the paper's architecture or establish that mmBERT LoRA is superior to full fine-tuning.
Its exact recipe, comparison, and limitations are recorded in `reports/model-experiments.md`.
