#!/usr/bin/env python3
"""Freeze a text-free matched LogInject long-log panel without scoring it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from morgott.data import file_sha256
from morgott.models.mmbert.data import filter_small_training_sets
from morgott.models.mmbert.inference import verified_artifact_path
from morgott.normalization import strict_normalize
from morgott.sources.tasks import _sensitive_text_reasons

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "loginject_long_span_panel"
ZENODO_RECORD = 20_436_935
ARCHIVE_MD5 = "bf56698a2ab2dd2280189620e7654a6d"
ARCHIVE_SHA256 = "010118d6cfbb03440bb7178d431912a2207f602c9ed546f47667a37d03820a8f"
BATCH_SIZE = 50
SOURCE_FILES = {
    "data/adversarial/samples.jsonl": (
        "ab8eb971bdc99e1f236f50dc9f395b0a3fd63fde99eaf4de29b6e321e5135827",
        2_569,
    ),
    "data/benign/apache_access.jsonl": (
        "bea8a7cffc1651a7a678bc2936d3f233052585de9740db455ec7e6b9cf227783",
        4_521,
    ),
    "data/benign/json_api.jsonl": (
        "14e4ed79177c11629a69e476175ca0b2519e9095c4890cd0a2453e7ed45ffdcc",
        2_910,
    ),
    "data/benign/ssh_auth.jsonl": (
        "c5affd368d03634ef1ab34b52bbc638cf09f2790f8859253a3fca7804b7e7433",
        2_847,
    ),
}
PANEL_FIELDS = {
    "attack_level",
    "attack_objective",
    "clean_normalized_sha256",
    "clean_text_sha256",
    "clean_token_count",
    "fragment_count",
    "injection_vector",
    "pair_id",
    "sample_id",
    "span_count",
    "spans",
    "split_group_id",
    "text_chars",
    "text_normalized_sha256",
    "text_sha256",
    "token_count",
}
EXPERIMENT_FILES = (
    "experiments/loginject_long_span_panel/README.md",
    "experiments/loginject_long_span_panel/prepare.py",
    "experiments/loginject_long_span_panel/test_prepare.py",
    "experiments/force_bench_eval/run.py",
)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("loginject_force", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FORCE = _load_module(ROOT / "experiments" / "force_bench_eval" / "run.py")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source(source_root: Path) -> tuple[list[dict], list[dict], dict]:
    rows = {}
    contract = {}
    for relative, (digest, count) in SOURCE_FILES.items():
        path = source_root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"LogInject source digest mismatch: {relative}")
        values = _read_jsonl(path)
        if len(values) != count:
            raise ValueError(f"LogInject source count mismatch: {relative}")
        rows[relative] = values
        contract[relative] = {"rows": count, "sha256": digest}
    adversarial = rows["data/adversarial/samples.jsonl"]
    ids = [row.get("sample_id") for row in adversarial]
    if len(ids) != len(set(ids)) or any(
        not isinstance(value, str) or not value for value in ids
    ):
        raise ValueError("LogInject sample IDs are invalid or duplicated")
    benign = [
        row
        for relative in SOURCE_FILES
        if "/benign/" in relative
        for row in rows[relative]
    ]
    if any(
        not isinstance(row.get("raw_log"), str) or not row["raw_log"] for row in benign
    ):
        raise ValueError("LogInject benign row has no raw log")
    return adversarial, benign, contract


def _attack_entries(row: dict) -> list[str]:
    if row.get("attack_level") == 2:
        fragments = row.get("fragments")
        if not isinstance(fragments, list) or len(fragments) != row.get(
            "fragment_count"
        ):
            raise ValueError("LogInject stitched sample has invalid fragments")
        values = [fragment.get("raw_log") for fragment in fragments]
    else:
        values = [row.get("raw_log")]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("LogInject attack row has no raw log")
    return values


def _formatted(entries: list[str], attacked: set[int]) -> tuple[str, list[dict]]:
    lines = [f"[Entry {index + 1}] {value}" for index, value in enumerate(entries)]
    normalized_lines = [strict_normalize(line) for line in lines]
    normalized = " ".join(normalized_lines)
    if strict_normalize("\n".join(lines)) != normalized:
        raise ValueError("LogInject normalized line assembly changed")
    spans = []
    offset = 0
    for index, line in enumerate(normalized_lines):
        if index in attacked:
            spans.append(
                {"entry_index": index, "start": offset, "end": offset + len(line)}
            )
        offset += len(line) + 1
    return "\n".join(lines), spans


def _pair(row: dict, benign: list[dict]) -> dict:
    sample_id = row["sample_id"]
    rng = random.Random(
        int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    )
    clean_entries = [entry["raw_log"] for entry in rng.sample(benign, BATCH_SIZE)]
    attack_entries = _attack_entries(row)
    start = rng.randrange(BATCH_SIZE - len(attack_entries) + 1)
    attacked_indices = set(range(start, start + len(attack_entries)))
    attacked_entries = clean_entries.copy()
    attacked_entries[start : start + len(attack_entries)] = attack_entries
    clean, clean_spans = _formatted(clean_entries, set())
    attacked, spans = _formatted(attacked_entries, attacked_indices)
    if clean_spans or len(spans) != len(attack_entries):
        raise ValueError("LogInject matched span construction failed")
    return {
        "pair_id": f"loginject:{sample_id}",
        "sample_id": sample_id,
        "split_group_id": f"loginject:{sample_id}",
        "attack_level": row["attack_level"],
        "attack_objective": row["attack_objective"],
        "injection_vector": row["injection_vector"],
        "fragment_count": len(attack_entries),
        "clean_text": clean,
        "text": attacked,
        "spans": spans,
    }


def _tokenizer() -> tuple[Tokenizer, dict]:
    registry_path = ROOT / "model-artifacts.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    spec = registry["models"]["mmbert-lora-full-s42"]["serving"]["tokenizer"]
    path = verified_artifact_path(ROOT, spec, name="registered tokenizer")
    return Tokenizer.from_file(str(path)), {
        "registry_sha256": file_sha256(registry_path),
        "tokenizer_sha256": spec["sha256"],
    }


def _panel_row(pair: dict, tokenizer: Tokenizer) -> dict:
    clean = pair["clean_text"]
    attacked = pair["text"]
    return {key: pair[key] for key in PANEL_FIELDS & pair.keys()} | {
        "clean_text_sha256": hashlib.sha256(clean.encode()).hexdigest(),
        "clean_normalized_sha256": hashlib.sha256(
            strict_normalize(clean).encode()
        ).hexdigest(),
        "clean_token_count": len(tokenizer.encode(strict_normalize(clean)).ids),
        "text_sha256": hashlib.sha256(attacked.encode()).hexdigest(),
        "text_normalized_sha256": hashlib.sha256(
            strict_normalize(attacked).encode()
        ).hexdigest(),
        "text_chars": len(attacked),
        "token_count": len(tokenizer.encode(strict_normalize(attacked)).ids),
        "span_count": len(pair["spans"]),
        "spans": pair["spans"],
    }


def _prepare(source_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("LogInject panel is write-once; use a fresh output")
    adversarial, benign, source_contract = _source(source_root)
    pairs = [_pair(row, benign) for row in adversarial]
    candidates = {
        "loginject": [
            {
                "id": f"{pair['pair_id']}:clean",
                "text": pair["clean_text"],
                "source": "loginject",
                "label": 0,
            }
            for pair in pairs
        ]
        + [
            {
                "id": f"{pair['pair_id']}:attack",
                "text": pair["text"],
                "source": "loginject",
                "label": 1,
            }
            for pair in pairs
        ]
        + [
            {
                "id": f"{pair['pair_id']}:span:{index}",
                "text": strict_normalize(pair["text"])[span["start"] : span["end"]],
                "source": "loginject",
                "label": 1,
            }
            for pair in pairs
            for index, span in enumerate(pair["spans"])
        ]
    }
    reference_counts = Counter()
    kept, removed = filter_small_training_sets(
        candidates,
        FORCE._fit_references(reference_counts),
    )
    kept_ids = {row["id"] for row in kept["loginject"]}
    retained = [
        pair
        for pair in pairs
        if all(
            value in kept_ids
            for value in (
                f"{pair['pair_id']}:clean",
                f"{pair['pair_id']}:attack",
                *(
                    f"{pair['pair_id']}:span:{index}"
                    for index in range(len(pair["spans"]))
                ),
            )
        )
    ]
    tokenizer, tokenizer_contract = _tokenizer()
    panel = [_panel_row(pair, tokenizer) for pair in retained]
    panel.sort(key=lambda row: row["pair_id"])
    if not panel or any(set(row) != PANEL_FIELDS for row in panel):
        raise ValueError("LogInject panel is empty or malformed")
    panel_spec = FORCE._write_gzip_jsonl(output / "panel.jsonl.gz", panel)
    privacy = Counter(
        reason
        for row in adversarial
        for reason in _sensitive_text_reasons("\n".join(_attack_entries(row)))
    )
    manifest = {
        "schema_version": 1,
        "purpose": "sealed source-heldout matched known-span long generated-log evaluation",
        "source": {
            "zenodo_record": ZENODO_RECORD,
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": ARCHIVE_SHA256,
            "files": source_contract,
            "generated_rows_only": True,
            "raw_text_retained_in_artifacts": False,
        },
        "construction": {
            "batch_size": BATCH_SIZE,
            "clean": "50 deterministic generated benign log entries",
            "attack": "replace one or more contiguous clean entries with the source attack entry or ordered fragments",
            "known_span": "strict-normalized offsets of every complete replaced entry",
            "input_channel": "untrusted_content",
        },
        "population": {
            "source_attacks": len(pairs),
            "retained_pairs": len(panel),
            "repositories_or_natural_documents": 0,
            "token_count": {
                "at_least_512": sum(row["token_count"] >= 512 for row in panel),
                "at_least_1024": sum(row["token_count"] >= 1_024 for row in panel),
                "maximum": max(row["token_count"] for row in panel),
            },
            "by_level": dict(
                sorted(Counter(str(row["attack_level"]) for row in panel).items())
            ),
            "by_vector": dict(
                sorted(Counter(row["injection_vector"] for row in panel).items())
            ),
        },
        "selection": {
            "fit_reference_rows": dict(sorted(reference_counts.items())),
            "removed_components": removed["loginject"],
            "pair_rule": "remove the whole pair when its clean batch, attack batch, or any span overlaps fit data",
            "synthetic_sensitive_pattern_matches": dict(sorted(privacy.items())),
        },
        "tokenizer_contract": tokenizer_contract,
        "experiment_contract": FORCE._file_contract(EXPERIMENT_FILES),
        "evaluation_contract": {
            "status": "sealed_unscored",
            "use": "evaluate exactly one frozen candidate and its unchanged incumbent after architecture and operating-point selection",
            "prohibited": [
                "fitting",
                "threshold selection",
                "prompt selection",
                "architecture selection",
            ],
            "report": [
                "paired ordering",
                "attack recall",
                "clean restriction load",
                "span-window admission",
                "level and vector slices",
            ],
        },
        "panel": panel_spec,
        "limitations": [
            "Every released benign and adversarial log row is generated.",
            "A log batch is not a natural document.",
            "The attack generator uses a small template system.",
            "Complete replaced-entry spans are broader than payload-only spans.",
        ],
    }
    FORCE._write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = _prepare(args.source_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
