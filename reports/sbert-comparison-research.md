# SBERT compared with Morgott's mmBERT

Research snapshot: 2026-07-29.

## Bottom line

There is no primary-source evidence that an off-the-shelf SBERT embedding plus a shallow classifier would outperform Morgott's task-trained mmBERT at its source-held-out, low-FPR prompt-injection objective.
"SBERT versus mmBERT" is also a false dichotomy: Sentence Transformers is an encoder, pooling, and training setup that can use mmBERT itself as the backbone, and the official mmBERT card demonstrates exactly that path in its dense-retrieval example ([Sentence Transformers training overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html#model), [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base#dense-retrieval-with-sentence-transformers)).
My best current inference is that frozen SBERT embeddings plus logistic regression would be a cheap diagnostic, but are not a likely replacement for the current task-specific head or LoRA without a direct experiment.

Morgott's maintained scorer already makes one encoder pass per text, pools CLS, mean, and maximum token representations, and applies a small binary head ([implementation](../src/morgott/models/mmbert/core.py)).
Its full-data frozen recipe trains directly on the binary instruction-subversion label with the retained matched-pair ranking loss, while the reduced-mixture one-seed LoRA gate deliberately excluded generated pairs and still does not establish a general method win ([model ledger](model-experiments.md#full-data-frozen-mmbert-first-line-shadow-2026-07-28)).
The completed full-mixture LoRA includes the retained pairs but still does not establish a general method win.
SBERT may change the representation geometry, but it does not by itself supply the missing matched multilingual, long-document, or prospective evaluation evidence.

## Architecture tradeoffs

| Design | What it does | Fit for Morgott |
|---|---|---|
| Sentence Transformer, or bi-encoder | Independently maps each text to one reusable fixed-size vector. | Useful when embeddings are reused for retrieval, clustering, or many pair comparisons. |
| Current mmBERT classifier | Jointly contextualizes tokens within one text, pools them once, then predicts one route score. | Directly matches the current one-text binary classification task. |
| Cross-encoder | Processes a pair jointly and emits a score or class rather than reusable sentence embeddings. | Relevant only if Morgott introduces a trusted second input such as a policy or exemplar. |
| Token classifier | Predicts labels from individual contextual token representations. | Relevant to payload localization, but requires span labels and is not obtained by adopting SBERT. |

The original SBERT result is primarily an efficiency result for semantic search and pair comparison: independent embeddings avoid recomputing every possible text pair ([SBERT paper](https://aclanthology.org/D19-1410/)).
That advantage mostly disappears for Morgott's present inference shape because every new prompt still needs one transformer pass and the shallow classifier is negligible beside the encoder.
It would matter if the same texts were indexed or compared with many references.

Official Sentence Transformers guidance says cross-encoders generally perform better on pairwise tasks because both inputs attend jointly, but they are slower because each pair must be recomputed and their embeddings cannot be precomputed ([quickstart](https://www.sbert.net/docs/quickstart.html#cross-encoder), [CrossEncoder reference](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)).
A cross-encoder is therefore not a drop-in alternative to Morgott's single-text route.
Adding a policy, trusted instruction, or attack prototype as the second text would define a new task and require its own labels, threat analysis, and evaluation.

The current mmBERT is not a token classifier because it pools all token representations before classification.
The original BERT design distinguishes sequence classification from token-level outputs, so span localization would require a separate token-labelled objective and usable payload annotations ([BERT paper](https://arxiv.org/abs/1810.04805)).

## Multilingual and long-context limits

"SBERT" does not imply multilingual support.
Official multilingual Sentence Transformer checkpoints cover model-specific language sets, while multilingual knowledge distillation explicitly trains translated sentences to align across languages ([pretrained multilingual models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#multilingual-models), [multilingual distillation paper](https://arxiv.org/abs/2004.09813)).
By contrast, the pinned mmBERT backbone advertises training across more than 1,800 languages and an 8,192-token architectural maximum, although Morgott deliberately truncates to the first 512 normalized tokens ([mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base#model-architecture), [local implementation](../src/morgott/models/mmbert/core.py)).
Language coverage is not prompt-injection accuracy, so either approach still needs paired benign and attack transformations evaluated per language.

Sentence Transformers does not automatically solve the first-512-token limitation.
Its maximum length is checkpoint-specific, over-length input is truncated, and its documentation warns that a model trained on short texts may produce poor long-text representations even when the backbone accepts more tokens ([input sequence length](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#input-sequence-length)).
Both SBERT and Morgott's current pooling reduce a document to one vector and lose payload location.
Long-document improvement still requires known-span or carefully grouped chunk and document-level supervision, not merely a different pooling library.

## Training options

The cheapest option is a frozen embedding probe with logistic regression.
The original SBERT paper evaluated fixed embeddings through logistic-regression transfer tasks, but its experiments were semantic similarity and general transfer tasks rather than prompt-injection detection ([SBERT paper](https://aclanthology.org/D19-1410/)).

A stronger option is SetFit-style training: contrastively fine-tune a Sentence Transformer, then train a shallow classifier on its embeddings ([SetFit paper](https://arxiv.org/abs/2209.11055)).
SetFit is evidence that the design is viable for few-shot classification, not evidence that it beats task-specific encoder adaptation on Morgott's million-row mixture.

Sentence Transformers also supports class-labelled single inputs through batch triplet losses and paired or triplet data through several ranking and similarity losses ([official loss table](https://www.sbert.net/docs/sentence_transformer/loss_overview.html)).
The loss must be chosen for Morgott's matched clean/attack structure rather than because it is popular on retrieval benchmarks.
Random same-label pairs could reinforce source and template shortcuts, so this is a hypothesis to test with source-held-out evaluation, not an assumed robustness gain.

## Fair experiment

1. Use the same pinned `jhu-clsp/mmBERT-base` initialization in every causal arm so the comparison does not confound backbone size, tokenizer, language coverage, or context support.
2. Keep strict normalization, the first-512-token policy, leakage filtering, training rows, domain weights, grouped roles, matched pairs, and validation checkpoint rule identical.
3. First fit a deterministic frozen mean-pooled mmBERT plus logistic-regression probe, which cheaply tests whether a simpler fixed representation is competitive without calling it SBERT fine-tuning.
4. Only if that probe is promising and a new run is authorized, compare the retained BCE plus pair-ranking classifier with one predeclared Sentence Transformer objective followed by the same fixed logistic-regression design, matching encoder updates and seeds.
5. Select the classifier and operating threshold on validation only, then report canonical dev-test, SEP, and PromptShield results using the existing Track B caveat.
6. Report TPR at the supported FPR point, ROC AUC, PR AUC, per-source and source-held-out slices, channel, language, length, finance false positives, matched-pair separation, mutation ASR at multiple attempts, latency, and peak memory.
7. Treat an off-the-shelf multilingual SBERT plus logistic regression only as a practical secondary control because it changes the backbone, pretraining data, tokenizer, and often the maximum sequence length at once.

This remains a lower-priority proposal until its cheap frozen probe and the
evidence gates above justify encoder work. Do not add SBERT machinery merely on architectural intuition.
