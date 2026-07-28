"""Inference-only reconstruction of the retained routing-encoder members.

This module deliberately does NOT import the archived experiment runner.
`AGENTS.md` forbids reviving old experiment runners as the default trainer, so
only the ~60 lines of forward-pass architecture are reproduced here, with no
training path at all. Everything needed to score text is:

  frozen encoder (bfloat16) -> 3-way pooling (cls | mean | max) -> MultipoolHead

The head checkpoints on disk were produced by the archived runner. Their exact
provenance, including model id, revision and sha256, is recorded in
`artifacts/direct_failure_repair_ensemble/ensemble-audit.json`.

Correctness of this reconstruction is established by `reproduce_check.py`,
which re-scores the frozen dev-test suite and compares against the metrics the
original runner recorded. Do not trust any number this module produces until
that check passes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

HEADS = (
    "direct_instruction_subversion",
    "indirect_instruction_subversion",
    "jailbreak",
    "harmful_intent",
)

# Mirrors routing_encoder.py:41-43. Direct-user text is truncated to a single
# window; untrusted content is chunked with overlap and max-pooled.
DIRECT_MAX_TOKENS = 256
UNTRUSTED_MAX_TOKENS = 512
UNTRUSTED_OVERLAP = 128


@dataclass(frozen=True)
class Member:
    """One frozen-encoder + trained-head pair from the retained ensemble."""

    name: str
    model_id: str
    model_revision: str
    head_path: Path
    head_sha256: str

    def resolved_head(self) -> Path:
        path = self.head_path
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path


# Read from artifacts/direct_failure_repair_ensemble/ensemble-audit.json.
# The ensemble uses the `wildguard_weak_transfer` recipe for both members.
ENSEMBLE_MEMBERS = (
    Member(
        name="english_modernbert",
        model_id="answerdotai/ModernBERT-base",
        model_revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        head_path=Path(
            "artifacts/direct_failure_repair_ablation/direct_failure_repair"
            "/modernbert-base/wildguard_weak_transfer/seed_42/head.safetensors"
        ),
        head_sha256="0230ff640ee0b90bffc6c75694fa52a1ba8ea2a7454d677f15c8c0f461183f0f",
    ),
    Member(
        name="mmbert_base",
        model_id="jhu-clsp/mmBERT-base",
        model_revision="",  # filled from the audit at load time if present
        head_path=Path(
            "artifacts/direct_failure_repair_mmbert_base/direct_failure_repair"
            "/mmbert-base/wildguard_weak_transfer/seed_42/head.safetensors"
        ),
        head_sha256="",
    ),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_hidden(hidden, attention_mask):
    """Concatenated CLS, masked-mean and masked-max pooling.

    Byte-for-byte the pooling in routing_encoder.py:927.
    """
    import torch

    mask = attention_mask.bool()
    expanded = mask.unsqueeze(-1)
    cls = hidden[:, 0] * expanded[:, 0]
    mean = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
    maximum = hidden.masked_fill(~expanded, torch.finfo(hidden.dtype).min).amax(dim=1)
    return torch.cat((cls, mean, maximum), dim=-1)


def _build_head(hidden_size: int):
    """The MultipoolHead architecture from routing_encoder.py:889.

    Constructed with no seed because every parameter is overwritten by the
    checkpoint load; a seed here would be misleading.
    """
    from torch import nn

    class MultipoolHead(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size * 3)
            self.projection = nn.Linear(hidden_size * 3, 384)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(0.1)
            self.outputs = nn.Linear(384, len(HEADS))

        def forward(self, pooled):
            return self.outputs(
                self.dropout(self.activation(self.projection(self.norm(pooled))))
            )

    return MultipoolHead(hidden_size)


def load_member(
    member: Member,
    *,
    device: str = "cuda",
    verify_hash: bool = True,
    attention_implementation: str | None = None,
):
    """Load one member's tokenizer, frozen encoder and trained head."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    head_path = member.resolved_head()
    if not head_path.exists():
        raise FileNotFoundError(f"missing head checkpoint: {head_path}")
    if verify_hash and member.head_sha256:
        actual = _file_sha256(head_path)
        if actual != member.head_sha256:
            raise ValueError(
                f"head checkpoint hash mismatch for {member.name}: "
                f"expected {member.head_sha256}, found {actual}"
            )

    kwargs = {"dtype": torch.bfloat16}
    if member.model_revision:
        kwargs["revision"] = member.model_revision
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    tokenizer_kwargs = (
        {"revision": member.model_revision} if member.model_revision else {}
    )

    tokenizer = AutoTokenizer.from_pretrained(member.model_id, **tokenizer_kwargs)
    encoder = AutoModel.from_pretrained(member.model_id, **kwargs)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.to(device).eval()

    head = _build_head(encoder.config.hidden_size)
    head.load_state_dict(load_file(str(head_path)))
    head.to(device).eval()
    return tokenizer, encoder, head


def _windows(tokenizer, text: str, channel: str) -> list[np.ndarray]:
    """Tokenize one record into the windows the runner would have produced."""
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if special != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("tokenizer must add exactly one CLS and one SEP token")

    encoding = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = np.asarray(encoding["input_ids"], dtype=np.int64)

    if channel == "direct_user":
        limit = DIRECT_MAX_TOKENS - special
        chunks = [token_ids[:limit]]
    else:
        limit = UNTRUSTED_MAX_TOKENS - special
        step = limit - UNTRUSTED_OVERLAP
        chunks = [
            token_ids[start : start + limit]
            for start in range(0, max(len(token_ids), 1), step)
            if len(token_ids[start : start + limit]) or start == 0
        ]

    windows = []
    for chunk in chunks:
        values = np.empty(len(chunk) + 2, dtype=np.int64)
        values[0] = tokenizer.cls_token_id
        values[1:-1] = chunk
        values[-1] = tokenizer.sep_token_id
        windows.append(values)
    return windows


def score_texts(
    loaded,
    texts: list[str],
    *,
    channel: str = "direct_user",
    device: str = "cuda",
    token_budget: int = 2048,
) -> np.ndarray:
    """Return per-row logits of shape (len(texts), len(HEADS)).

    Rows longer than one window are max-pooled across windows, matching
    `_aggregate_logits` in the archived runner.
    """
    import torch

    tokenizer, encoder, head = loaded

    examples: list[tuple[int, np.ndarray]] = []
    for row_index, text in enumerate(texts):
        for window in _windows(tokenizer, text, channel):
            examples.append((row_index, window))

    order = sorted(range(len(examples)), key=lambda i: len(examples[i][1]))
    row_logits = np.full((len(texts), len(HEADS)), -np.inf, dtype=np.float32)

    batch: list[int] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    def flush(indices: list[int]) -> None:
        if not indices:
            return
        width = max(len(examples[i][1]) for i in indices)
        input_ids = np.full((len(indices), width), pad_id, dtype=np.int64)
        attention = np.zeros((len(indices), width), dtype=np.int64)
        for slot, index in enumerate(indices):
            window = examples[index][1]
            input_ids[slot, : len(window)] = window
            attention[slot, : len(window)] = 1
        input_tensor = torch.from_numpy(input_ids).to(device)
        mask_tensor = torch.from_numpy(attention).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(
                input_ids=input_tensor, attention_mask=mask_tensor
            ).last_hidden_state
            pooled = _pool_hidden(hidden, mask_tensor)
            values = head(pooled).float().cpu().numpy()
        for slot, index in enumerate(indices):
            row = examples[index][0]
            row_logits[row] = np.maximum(row_logits[row], values[slot])

    for index in order:
        width = len(examples[index][1])
        if (
            batch
            and (len(batch) + 1) * max(width, max(len(examples[i][1]) for i in batch))
            > token_budget
        ):
            flush(batch)
            batch = []
        batch.append(index)
    flush(batch)

    return row_logits


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values.astype(np.float64)))


def direct_head_probability(logits: np.ndarray) -> np.ndarray:
    """Return the direct-instruction-subversion head probability."""
    return sigmoid(logits[:, HEADS.index("direct_instruction_subversion")])


def route_probability(logits: np.ndarray) -> np.ndarray:
    """Return the direct-user route used by the retained ensemble."""
    direct = direct_head_probability(logits)
    jailbreak = sigmoid(logits[:, HEADS.index("jailbreak")])
    return np.maximum(direct, jailbreak)


def direct_route_probability(logits: np.ndarray) -> np.ndarray:
    """Historical direct-head signal retained for archived-script compatibility."""
    return direct_head_probability(logits)


def load_audit_members() -> tuple[Member, ...]:
    """Refresh member metadata from the ensemble audit so paths stay honest."""
    audit_path = (
        REPO_ROOT / "artifacts/direct_failure_repair_ensemble/ensemble-audit.json"
    )
    audit = json.loads(audit_path.read_text())
    members = []
    for name, spec in audit["members"].items():
        head = Path(spec["head"])
        try:
            head = head.relative_to(REPO_ROOT)
        except ValueError:
            pass
        members.append(
            Member(
                name=name,
                model_id=spec["model_id"],
                model_revision=spec.get("model_revision", ""),
                head_path=head,
                head_sha256=spec.get("head_sha256", ""),
            )
        )
    return tuple(members)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for member in load_audit_members():
        print(f"{member.name:20} {member.model_id:32} {member.resolved_head().name}")
        print(f"{'':20} revision={member.model_revision or '(unpinned)'}")
        print(f"{'':20} exists={member.resolved_head().exists()}")
