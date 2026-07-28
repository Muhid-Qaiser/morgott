from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from morgott.models.mmbert import data as mmbert_data
from morgott.models.mmbert import external_data
from morgott.models.mmbert.train import prepare_training_data
from morgott.normalization import strict_normalize


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


class MmbertDataTests(unittest.TestCase):
    def test_model_target_accepts_only_declared_positive_tags(self):
        for tag in mmbert_data.INSTRUCTION_SUBVERSION_TAGS:
            row = _canonical(f"positive:{tag}", "train", 1, "source")
            row["security_tags"] = [tag]
            self.assertTrue(mmbert_data._model_eligible(row))
        unsupported = _canonical("positive:unsupported", "train", 1, "source")
        unsupported["security_tags"] = ["future_subversion_type"]
        self.assertFalse(mmbert_data._model_eligible(unsupported))

    def test_external_projection_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "invalid PromptShield row"):
            external_data._promptshield_rows("train", [{"prompt": "", "label": 0}])

    def test_candidate_reference_does_not_exclude_itself(self):
        row = {
            "id": "validation:1",
            "text": "heldout validation example",
            "label": 0,
            "source": "promptshield",
            "input_channel": "direct_user",
        }
        peer = {**row, "id": "validation:2"}
        duplicate = {**row, "id": "pair:1", "source": "matched_pairs"}
        kept, removed = mmbert_data.filter_small_training_sets(
            {
                "promptshield_validation": [row, peer],
                "pairs": [duplicate],
            },
            [
                {
                    **candidate,
                    "_candidate_dataset": "promptshield_validation",
                }
                for candidate in (row, peer)
            ],
        )
        self.assertEqual(kept["promptshield_validation"], [row, peer])
        self.assertEqual(kept["pairs"], [])
        self.assertEqual(removed["pairs"], {"normalized_exact": 1})

    def test_stream_helpers_are_deterministic_and_preserve_rows(self):
        rows = [{"id": str(index)} for index in range(20)]
        first = list(mmbert_data.shuffled(rows, seed=42, buffer_size=4))
        second = list(mmbert_data.shuffled(rows, seed=42, buffer_size=4))
        self.assertEqual(first, second)
        self.assertEqual(
            sorted(row["id"] for row in first), sorted(row["id"] for row in rows)
        )
        self.assertEqual(
            list(mmbert_data.batches(rows, 7)),
            [rows[:7], rows[7:14], rows[14:]],
        )

    def test_source_label_weights_have_population_mean_one(self):
        rows = [
            {"id": "a", "text": "alpha", "source": "large", "label": 0},
            {"id": "b", "text": "beta", "source": "large", "label": 0},
            {"id": "c", "text": "gamma", "source": "small", "label": 1},
        ]
        counts = Counter({("large", 0): 2, ("small", 1): 1})
        owners = {
            hashlib.sha256(strict_normalize(row["text"]).encode()).hexdigest(): (
                row["id"],
                row["source"],
                row["label"],
            )
            for row in rows
        }
        weighted = list(mmbert_data.training_rows(rows, counts, owners))
        self.assertAlmostEqual(sum(row["weight"] for row in weighted) / 3, 1.0)

    def test_full_preflight_reads_canonical_external_and_pair_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            views_dir = data_dir / "views" / "routing"
            external_dir = root / "external"
            views_dir.mkdir(parents=True)
            external_dir.mkdir()

            populations = {
                "train": [
                    _canonical("train-benign", "train", 0, "source-a"),
                    _canonical("train-attack", "train", 1, "source-a"),
                    _canonical("train-benign-b", "train", 0, "source-b"),
                    _canonical("train-attack-b", "train", 1, "source-b"),
                    {
                        **_canonical("harmful-negative", "train", 0, "source-c"),
                        "security_label": "harmful_non_injection",
                        "security_tags": ["harmful_non_injection"],
                    },
                    {
                        **_canonical("harmful-not-injection", "train", 1, "source-c"),
                        "injection_label": None,
                        "security_label": "harmful_non_injection",
                    },
                    {
                        **_canonical("model-output", "train", 0, "source-c"),
                        "input_channel": "model_output",
                    },
                ],
                "validation": [
                    _canonical("validation-benign", "validation", 0, "source-a"),
                    _canonical("validation-attack", "validation", 1, "source-a"),
                ],
                "dev_test": [
                    _canonical("dev-benign", "dev_test", 0, "source-a"),
                    _canonical("dev-attack", "dev_test", 1, "source-a"),
                ],
            }
            routing = {}
            for split, rows in populations.items():
                relative = Path("views") / "routing" / f"{split}.jsonl"
                digest = _write_jsonl(data_dir / relative, rows)
                routing[split] = {
                    "path": str(relative),
                    "sha256": digest,
                    "rows": len(rows),
                }
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "canonical_row_schema_version": 5,
                        "routing_views": routing,
                    }
                )
            )

            external = {
                "promptshield_train": [
                    {
                        "id": "ps-train-0",
                        "text": "promptshield clean",
                        "label": 0,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                    {
                        "id": "ps-train-1",
                        "text": "promptshield attack",
                        "label": 1,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                ],
                "promptshield_validation": [
                    {
                        "id": "ps-validation-0",
                        "text": "heldout clean",
                        "label": 0,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                    {
                        "id": "ps-validation-1",
                        "text": "heldout attack",
                        "label": 1,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                ],
                "promptshield_test": [
                    {
                        "id": "ps-test-0",
                        "text": "test clean",
                        "label": 0,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                    {
                        "id": "ps-test-1",
                        "text": "test attack",
                        "label": 1,
                        "source": "promptshield",
                        "input_channel": "direct_user",
                    },
                ],
                "sep": [
                    {
                        "id": "sep-0",
                        "text": "sep clean",
                        "label": 0,
                        "source": "sep",
                        "input_channel": "untrusted_content",
                    },
                    {
                        "id": "sep-1",
                        "text": "sep attack",
                        "label": 1,
                        "source": "sep",
                        "input_channel": "untrusted_content",
                    },
                ],
            }
            for rows in external.values():
                for row in rows:
                    row.update(source_revision="test-revision", license="test-license")
            outputs = {}
            for name, rows in external.items():
                path = external_dir / f"{name}.jsonl"
                outputs[name] = {
                    "path": path.name,
                    "sha256": _write_jsonl(path, rows),
                    "rows": len(rows),
                }
            (external_dir / "manifest.json").write_text(
                json.dumps({"schema_version": 1, "outputs": outputs})
            )

            pair_path = root / "pairs.jsonl.gz"
            raw_pairs = [
                {
                    "benign": "paired clean sample",
                    "attack": "paired attack sample",
                    "channel": "direct_user",
                },
                {
                    "benign": "train-benign unique sample",
                    "attack": "overlapping pair attack",
                    "channel": "direct_user",
                },
            ]
            with gzip.open(pair_path, "wb") as handle:
                for raw_pair in raw_pairs:
                    handle.write(json.dumps(raw_pair).encode() + b"\n")
            with gzip.open(pair_path, "rb") as handle:
                content_hash = hashlib.sha256(handle.read()).hexdigest()
            archive_hash = hashlib.sha256(pair_path.read_bytes()).hexdigest()

            with (
                patch.object(mmbert_data, "PAIR_ARCHIVE_SHA256", archive_hash),
                patch.object(mmbert_data, "PAIR_CONTENT_SHA256", content_hash),
                patch.object(mmbert_data, "is_checkpoint_group", return_value=True),
            ):
                prepared = prepare_training_data(data_dir, external_dir, pair_path)

            self.assertEqual(sum(prepared.canonical_counts.values()), 5)
            self.assertEqual(len(prepared.promptshield), 2)
            self.assertEqual(len(prepared.pairs), 1)
            self.assertEqual(len(prepared.checkpoint), 2)
            self.assertEqual(
                prepared.removed["pairs_against_canonical_train"],
                {"normalized_exact": 1},
            )
            self.assertEqual(prepared.removed["pair_atoms"], 1)


if __name__ == "__main__":
    unittest.main()
