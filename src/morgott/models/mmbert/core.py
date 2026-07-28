from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ...normalization import strict_normalize

MODEL_ID = "jhu-clsp/mmBERT-base"
MODEL_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
ATTENTION_IMPLEMENTATION = "sdpa"
MAX_TOKENS = 512
INSTRUCTION_SUBVERSION_TAGS = (
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "direct_jailbreak",
)
LORA_TARGETS = r"layers\.\d+\.attn\.(Wqkv|Wo)"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MODULES = 44
LORA_PARAMETERS = 811_008


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(*paths: Path) -> dict:
    root = Path(__file__).resolve().parents[4]
    lock = root / "uv.lock"
    sources = {}
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"source path escapes repository: {path}")
        sources[str(path.relative_to(root))] = file_sha256(path)
    return {
        "sources": dict(sorted(sources.items())),
        "uv_lock_sha256": file_sha256(lock),
    }


def new_head(hidden_size: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.LayerNorm(hidden_size * 3),
        nn.Linear(hidden_size * 3, 384),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(384, 1),
    )


def pool(hidden, attention_mask):
    import torch

    expanded = attention_mask.bool().unsqueeze(-1)
    cls = hidden[:, 0] * expanded[:, 0]
    mean = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
    maximum = hidden.masked_fill(
        ~expanded,
        torch.finfo(hidden.dtype).min,
    ).amax(dim=1)
    return torch.cat((cls, mean, maximum), dim=-1)


def load_base_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("mmBERT requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        dtype=torch.bfloat16,
    ).to("cuda")
    return encoder, tokenizer


def add_lora(encoder):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        encoder,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=LORA_TARGETS,
        ),
    )
    modules = [module for module in model.modules() if hasattr(module, "lora_A")]
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if len(modules) != LORA_MODULES or parameters != LORA_PARAMETERS:
        raise ValueError("pinned LoRA target contract changed")
    return model


def batch_logits(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    train_encoder: bool,
):
    import torch

    encoded = tokenizer(
        [strict_normalize(text) for text in texts],
        add_special_tokens=True,
        max_length=MAX_TOKENS,
        padding=True,
        return_tensors="pt",
        truncation=True,
    ).to("cuda")
    context = nullcontext() if train_encoder else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = encoder(**encoded).last_hidden_state
        features = pool(hidden, encoded["attention_mask"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return head(features)[:, 0]


def score_texts(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    encoder.eval()
    head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            values = batch_logits(
                encoder,
                tokenizer,
                head,
                texts[start : start + batch_size],
                train_encoder=False,
            )
            logits.append(values.float().cpu().numpy())
    if not logits:
        return np.empty(0, dtype=np.float64)
    values = np.concatenate(logits).astype(np.float64)
    scores = np.empty_like(values)
    positive = values >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    scores[~positive] = exponent / (1.0 + exponent)
    return scores
