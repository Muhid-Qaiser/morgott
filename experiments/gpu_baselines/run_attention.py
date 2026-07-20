"""Pinned off-the-shelf DeBERTa prompt-injection sensor evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    DebertaV2TokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from run_embeddings import (
    HERE,
    INPUT_SHA256,
    DIRECT_REVIEW_PRECISION_FLOOR,
    operating_points,
    precision_profiles,
    read_rows,
    validation_mask,
)
from vulsight_guard.detector import _rates, choose_threshold


MODELS = {
    "protectai": {
        "id": "protectai/deberta-v3-base-prompt-injection-v2",
        "revision": "90c9989b1a342275dd0d1a95aad283c04e075671",
        "license": "Apache-2.0",
        "known_overlap": [
            "XSTest-derived data appears in the model card's training-source list"
        ],
    },
    "piguard": {
        "id": "leolee99/PIGuard",
        "revision": "dd78b24e330193a22d2293ac66922dd4f982f563",
        "license": "MIT",
        "known_overlap": [
            "NotInject is authored by PIGuard's authors",
            "BIPIA is reported in PIGuard development/evaluation",
        ],
    },
}
MAX_LENGTH = 384


class PIGuardConfig(DebertaV2Config):
    """Audited equivalent of the checkpoint's 18-line remote config class."""

    model_type = "piguard"


class PIGuard(DebertaV2ForSequenceClassification):
    """Audited equivalent of the checkpoint's published classifier code."""

    config_class = PIGuardConfig

    def __init__(self, config: PIGuardConfig) -> None:
        super().__init__(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return SequenceClassifierOutput(
            logits=self.classifier(outputs.last_hidden_state[:, 0, :])
        )


class Sensor:
    def __init__(self, batch_size: int, model_name: str, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this experiment")
        self.spec = MODELS[model_name]
        self.name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        if model_name == "piguard":
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
                self.spec["id"], revision=self.spec["revision"]
            )
            config = PIGuardConfig.from_pretrained(
                self.spec["id"], revision=self.spec["revision"]
            )
            self.model = PIGuard.from_pretrained(
                self.spec["id"],
                revision=self.spec["revision"],
                config=config,
                dtype=dtype,
            ).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec["id"], revision=self.spec["revision"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.spec["id"],
                revision=self.spec["revision"],
                dtype=dtype,
            ).to(self.device)
        if self.model.config.id2label[1].lower() not in ("injection",):
            raise RuntimeError(f"unexpected labels: {self.model.config.id2label}")
        self.model.eval()

    @torch.inference_mode()
    def score(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        output = []
        batch_size = batch_size or self.batch_size
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits.float()
            output.append(logits.softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(output)


def cached_scores(name: str, rows: list[dict], sensor: Sensor) -> np.ndarray:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    path = (
        cache
        / f"{sensor.name}-scores32-{sensor.spec['revision'][:8]}-{MAX_LENGTH}-{name}-{INPUT_SHA256[name][:12]}.npy"
    )
    if path.exists():
        return np.load(path)
    scores = sensor.score([row["text"] for row in rows])
    np.save(path, scores)
    return scores


def chunked_scores(rows: list[dict], sensor: Sensor) -> np.ndarray:
    chunks: list[str] = []
    spans = []
    for row in rows:
        parts = [row["text"]] + [
            part
            for part in row["text"].split("\n\n")
            if len(part.strip()) >= 8 and part != row["text"]
        ]
        start = len(chunks)
        chunks.extend(parts)
        spans.append((start, len(chunks)))
    scores = sensor.score(chunks)
    return np.asarray([scores[start:end].max() for start, end in spans])


def threshold_on_validation(
    rows: list[dict], scores: np.ndarray, max_fpr: float
) -> tuple[float, dict]:
    validation = validation_mask(rows)
    selected_rows = [row for row, use in zip(rows, validation) if use]
    selected_scores = scores[validation]
    threshold = choose_threshold(
        [row["label"] for row in selected_rows], selected_scores, max_fpr=max_fpr
    )
    return threshold, _rates(selected_rows, selected_scores, threshold)


def measured_latency(sensor: Sensor, texts: list[str]) -> dict[str, float]:
    sample = texts[:100]
    for text in sample[:5]:
        sensor.score([text], batch_size=1)
    if sensor.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for text in sample:
        sensor.score([text], batch_size=1)
    if sensor.device.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - started) * 1_000 / len(sample)

    if sensor.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    sensor.score(sample, batch_size=32)
    if sensor.device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - started) * 1_000 / len(sample)
    return {"batch_1_ms_per_text": single_ms, "batch_32_ms_per_text": batch_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", choices=MODELS, default="protectai")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = {name: read_rows(name) for name in INPUT_SHA256}
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sensor = Sensor(args.batch_size, args.model, args.device)

    scoring_seconds = {}
    scores = {}
    for name in (
        "train",
        "toxic_chat",
        "prompt_injections",
        "tensor_trust_attack",
        "xstest",
        "notinject",
        "oasst1_chat",
        "do_not_answer",
        "harmbench",
        "multi_turn",
        "jailbreaks_over_time",
        "oasst1_position_stress",
        "indirect_train",
        "bipia_payload",
    ):
        step = time.perf_counter()
        scores[name] = cached_scores(name, rows[name], sensor)
        scoring_seconds[name] = time.perf_counter() - step

    direct_threshold, direct_validation = threshold_on_validation(
        rows["train"], scores["train"], max_fpr=0.001
    )
    direct_names = (
        "toxic_chat",
        "prompt_injections",
        "tensor_trust_attack",
        "xstest",
        "notinject",
        "oasst1_chat",
        "do_not_answer",
        "harmbench",
        "multi_turn",
        "jailbreaks_over_time",
        "oasst1_position_stress",
    )
    direct_sets = {
        name: _rates(rows[name], scores[name], direct_threshold)
        for name in direct_names
    }
    hard_names = (
        "xstest",
        "notinject",
        "oasst1_chat",
        "oasst1_position_stress",
        "do_not_answer",
        "harmbench",
    )
    direct_sets["hard_negative_aggregate"] = _rates(
        [row for name in hard_names for row in rows[name]],
        np.concatenate([scores[name] for name in hard_names]),
        direct_threshold,
    )
    validation = validation_mask(rows["train"])
    direct_validation_rows = [
        row for row, selected in zip(rows["train"], validation) if selected
    ]
    direct_evaluations = {name: (rows[name], scores[name]) for name in direct_names}
    direct_evaluations["hard_negative_aggregate"] = (
        [row for name in hard_names for row in rows[name]],
        np.concatenate([scores[name] for name in hard_names]),
    )
    direct_operating_points = operating_points(
        direct_validation_rows, scores["train"][validation], direct_evaluations
    )
    direct_precision_profiles = precision_profiles(
        direct_validation_rows, scores["train"][validation], direct_evaluations
    )

    indirect_threshold, indirect_validation = threshold_on_validation(
        rows["indirect_train"], scores["indirect_train"], max_fpr=0.0
    )
    indirect_sets = {
        "bipia_payload": _rates(
            rows["bipia_payload"], scores["bipia_payload"], indirect_threshold
        )
    }
    for name in ("bipia_context", "bipia_clean_context", "tensor_trust_context"):
        step = time.perf_counter()
        chunked = chunked_scores(rows[name], sensor)
        scoring_seconds[name + "_chunked"] = time.perf_counter() - step
        indirect_sets[name] = _rates(rows[name], chunked, indirect_threshold)

    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_sha256": INPUT_SHA256,
        "model": {
            **sensor.spec,
            "architecture": "DeBERTa-v3-base sequence classifier",
            "max_length": MAX_LENGTH,
            "dtype": "float16" if args.device == "cuda" else "float32",
            "training": "off-the-shelf; thresholds calibrated only on frozen validation groups",
        },
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if args.device == "cuda"
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20
                if args.device == "cuda"
                else None
            ),
        },
        "direct": {
            "threshold_target_fpr": 0.001,
            "threshold": direct_threshold,
            "validation": direct_validation,
            "sets": direct_sets,
            "operating_points": direct_operating_points,
            "precision_profiles": direct_precision_profiles,
            "default_precision_floor": DIRECT_REVIEW_PRECISION_FLOOR,
            "threshold_protocol": "grouped-validation diagnostics; not production calibration",
        },
        "indirect": {
            "threshold_target_fpr": 0.0,
            "threshold": indirect_threshold,
            "validation": indirect_validation,
            "scoring": "maximum score over whole document and blank-line paragraphs",
            "sets": indirect_sets,
        },
        "raw_threshold_0_5": {
            name: _rates(rows[name], scores[name], 0.5)
            for name in (
                "toxic_chat",
                "prompt_injections",
                "tensor_trust_attack",
                "multi_turn",
                "jailbreaks_over_time",
            )
        },
        "latency": measured_latency(
            sensor, [row["text"] for row in rows["oasst1_chat"]]
        ),
        "scoring_seconds": scoring_seconds,
        "wall_seconds": time.perf_counter() - started,
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    output = HERE / f"{args.model}_{args.device}_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
