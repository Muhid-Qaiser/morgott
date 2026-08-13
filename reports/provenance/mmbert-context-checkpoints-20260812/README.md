# ModernBERT context campaign checkpoints

These are exact copies of the three snapshots used by the completed 512/1,024
context and update-17,000/update-18,500 comparisons. They are retained through
Git LFS as reproducibility evidence, not as maintained model artifacts.

Files:

- `trainctx512-update-017000.pt` — fixed 512-token training snapshot.
- `trainctx1024-update-017000.pt` — fixed 1,024-token comparison snapshot.
- `trainctx1024-update-018500.pt` — historical-selector packaged snapshot.

`SOURCE_PATHS.tsv` binds each copy to its RunPod source path and recorded hash.
Verify the copies from this directory:

```bash
cd reports/provenance/mmbert-context-checkpoints-20260812
sha256sum -c SHA256SUMS
```

These snapshots are advisory and intentionally absent from
`model-artifacts.json`; maintained inference must not discover or load them.
Reproduction also needs the pinned base-model revision, tokenizer, campaign
source archive, and evaluation manifest recorded alongside this evidence.

PyTorch checkpoint files are pickle-based. Load only these trusted,
checksum-verified files and never load a replacement from an untrusted source.
