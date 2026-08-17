"""One-time local replay of the frozen LogInject matched panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.loginject_long_span_panel import prepare
from experiments.pipeline_benchmark import local

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "artifacts" / "loginject_long_span_panel" / "panel.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "artifacts" / "pipeline_benchmark" / "20260816"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rehydrate(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    adversarial, benign, _ = prepare._source(source_root)
    frozen = {row["pair_id"]: row for row in local.load_jsonl(FROZEN)}
    rows = []
    texts = {}
    for source in adversarial:
        pair = prepare._pair(source, benign)
        expected = frozen.get(pair["pair_id"])
        if expected is None:
            continue
        for variant, text, label, digest_field in (
            ("clean", pair["clean_text"], 0, "clean_text_sha256"),
            ("attack", pair["text"], 1, "text_sha256"),
        ):
            if _sha256(text) != expected[digest_field]:
                raise ValueError(
                    f"LogInject text mismatch: {pair['pair_id']}:{variant}"
                )
            artifact_id = f"{pair['pair_id']}:{variant}"
            rows.append(
                {
                    "panel_id": artifact_id,
                    "dataset": "loginject",
                    "source": "loginject",
                    "input_channel": "untrusted_content",
                    "label": label,
                    "text_sha256": expected[digest_field],
                    "pair_id": pair["pair_id"],
                    "variant": variant,
                    "attack_level": expected["attack_level"],
                    "injection_vector": expected["injection_vector"],
                }
            )
            texts[artifact_id] = text
    if len(rows) != 2 * len(frozen):
        raise ValueError("LogInject retained population did not rehydrate exactly")
    return rows, texts


def summarize(
    records: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    by_id = {row["artifact_id"]: row for row in records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["artifact_id"].rsplit(":", 1)[0]].append(row)
    profiles = {}
    for name, selected in selection["logprob"].items():
        low = float(selected["thresholds"]["untrusted_low"])
        high = float(selected["thresholds"]["local_high"])
        clean = [row for row in records if row["label"] == 0]
        attack = [row for row in records if row["label"] == 1]
        profiles[name] = {
            "thresholds": {"untrusted_low": low, "local_high": high},
            "clean_local_high_rate": sum(row["local_score"] >= high for row in clean)
            / len(clean),
            "clean_review_or_high_rate": sum(row["local_score"] >= low for row in clean)
            / len(clean),
            "attack_local_high_recall": sum(
                row["local_score"] >= high for row in attack
            )
            / len(attack),
            "attack_review_or_high_recall": sum(
                row["local_score"] >= low for row in attack
            )
            / len(attack),
        }
    movements = []
    for pair_id in grouped:
        clean = by_id[f"{pair_id}:clean"]["local_score"]
        attack = by_id[f"{pair_id}:attack"]["local_score"]
        movements.append(attack - clean)
    movements.sort()
    return {
        "schema_version": 1,
        "pairs": len(grouped),
        "profiles": profiles,
        "paired_score_movement": {
            "mean": sum(movements) / len(movements),
            "p50": movements[len(movements) // 2],
            "positive_rate": sum(value > 0 for value in movements) / len(movements),
        },
        "remote_cascade_status": "pending_provider_winner",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result_path = args.output / "loginject_local_scores.jsonl.gz"
    if result_path.exists():
        raise FileExistsError("LogInject is sealed and has already been scored")
    rows, texts = rehydrate(args.source_root)
    records, runtime = local.score_cuda(rows, texts)
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(result_path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    result = summarize(records, selection) | {
        "runtime": runtime,
        "result_sha256": local.file_sha256(result_path),
        "source_archive_sha256": prepare.ARCHIVE_SHA256,
    }
    (args.output / "loginject_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["profiles"], sort_keys=True))


if __name__ == "__main__":
    main()
