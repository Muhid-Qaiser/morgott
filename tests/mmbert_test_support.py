from __future__ import annotations

import hashlib
import json
from pathlib import Path

from morgott.models.mmbert import data as mmbert_data


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(row_id: str, split: str, label: int, source: str) -> dict:
    return {
        "schema_version": 5,
        "id": row_id,
        "text": f"{row_id} unique sample",
        "routing_label": label,
        "injection_label": label,
        "routing_training_eligible": True,
        "security_label": "benign" if label == 0 else "direct_prompt_injection",
        "security_tags": ["benign"] if label == 0 else ["direct_prompt_injection"],
        "label_basis": "source_supported",
        "data_role": split,
        "source": source,
        "input_channel": "direct_user",
        "split_group_id": f"group:{row_id}",
        "origins": [{"label_basis": "source_supported"}],
    }


def _training_data(**overrides) -> mmbert_data.TrainingData:
    values = {
        "views": {},
        "data_manifest_sha256": "a",
        "external_manifest_sha256": "b",
        "promptshield": [],
        "promptshield_validation": [],
        "pairs": [],
        "checkpoint": [],
        "calibration": [],
        "validation_partition": {},
        "canonical_counts": {},
        "canonical_group_counts": {},
        "canonical_owners": {},
        "removed": {},
    }
    values.update(overrides)
    return mmbert_data.TrainingData(**values)
