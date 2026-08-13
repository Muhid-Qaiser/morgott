from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.guard_baselines import adapters, canary


class _FakeBaseline:
    def __init__(self, spec, batch_size: int, model_identity: dict) -> None:
        self.spec = spec
        self.batch_size = batch_size
        self.model_identity = model_identity
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    @staticmethod
    def score(texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        if len(texts) != 2:
            raise AssertionError("the fake only serves the polarity smoke")
        return np.asarray([0.1, 0.9]), [False, False]

    def describe(self) -> dict:
        return {
            "baseline": self.spec.slug,
            "model_id": self.spec.repo_id,
            "model_revision": self.spec.revision,
            "native_cutoff": self.spec.native_threshold,
            "model_identity": self.model_identity,
        }

    def unload(self) -> None:
        self.unloaded = True


class GuardCanaryContractTests(unittest.TestCase):
    CASES = (
        ("granite-guardian-3.2-3b-a800m", 2, 4),
        ("qwen3guard-stream-4b", 2, 8),
        ("aprielguard", 4, 2),
    )

    @staticmethod
    def _row(prefix: str, index: int) -> dict:
        return {
            "id": f"{prefix}-{index}",
            "text": f"text {prefix} {index}",
            "label": index % 2,
            "source": "test-source",
            "input_channel": "direct_user",
            "security_tags": [],
        }

    @classmethod
    def _panel(cls) -> dict:
        slices = {
            name: [cls._row(name, 0), cls._row(name, 1)] for name in canary.SLICE_NAMES
        }
        return {
            "panel_sha256": "panel",
            "row_identity_sha256": {
                name: canary._identity_sha256(rows) for name, rows in slices.items()
            },
            "slices": slices,
            "redteam": [cls._row("redteam", 0), cls._row("redteam", 1)],
        }

    @staticmethod
    def _snapshot(root: Path, shards: int) -> Path:
        root.mkdir()
        names = [
            f"model-{index:05d}-of-{shards:05d}.safetensors"
            for index in range(1, shards + 1)
        ]
        for index, name in enumerate(names):
            (root / name).write_bytes(f"weights-{index}".encode())
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"tensor.{index}": name for index, name in enumerate(names)
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        return root

    @staticmethod
    def _score_rows(baseline, rows, *, batch_size: int, label: str) -> dict:
        del baseline, label
        if batch_size < 1:
            raise AssertionError("resolved batch size must be positive")
        return {
            "labels": np.asarray([row["label"] for row in rows], dtype=np.int8),
            "scores": np.asarray([0.1, 0.9], dtype=np.float64),
            "runtime": {"rows_per_second": 1_000.0},
        }

    def test_granite_qwen_and_apriel_publish_with_sharded_identity_and_defaults(
        self,
    ) -> None:
        panel = self._panel()
        created: list[_FakeBaseline] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = {
                slug: self._snapshot(root / f"snapshot-{index}", shards)
                for index, (slug, shards, _) in enumerate(self.CASES)
            }

            def build(slug: str, *, batch_size: int | None = None):
                spec = adapters.BASELINES[slug]
                resolved = spec.batch_size if batch_size is None else batch_size
                snapshot_identity = adapters._snapshot_identity(snapshots[slug], spec)
                model_identity = {
                    "files": snapshot_identity["files"],
                    "runtime_snapshot": {
                        key: value
                        for key, value in snapshot_identity.items()
                        if key != "files"
                    },
                }
                baseline = _FakeBaseline(spec, resolved, model_identity)
                created.append(baseline)
                return baseline

            with (
                patch.object(canary, "CANARY_ROWS", 8),
                patch.object(canary, "build_panel", return_value=panel),
                patch.object(canary, "build_baseline", side_effect=build),
                patch.object(canary, "score_rows", side_effect=self._score_rows),
                patch.object(
                    canary,
                    "_source_hashes",
                    return_value={"test_source": "0" * 64},
                ),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.is_bf16_supported", return_value=True),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.get_device_name", return_value="mock GPU"),
                patch("torch.cuda.max_memory_allocated", return_value=123),
                patch("torch.cuda.max_memory_reserved", return_value=456),
                patch("torch.cuda.empty_cache"),
            ):
                for slug, shards, expected_batch in self.CASES:
                    with self.subTest(slug=slug):
                        output = root / f"output-{slug}"
                        published, passes = canary.run_canary(
                            slug,
                            output=output,
                            batch_size=None,
                            require_panel_sha256="panel",
                            data_dir=root / "data",
                            external_dir=root / "external",
                            pairs=root / "pairs.jsonl.gz",
                            redteam=root / "redteam.jsonl.gz",
                        )

                        self.assertEqual(published, output)
                        self.assertTrue(passes)
                        self.assertTrue((output / "evaluation.json").is_file())
                        report = json.loads(
                            (output / "evaluation.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(report["schema_version"], 2)
                        self.assertEqual(
                            report["runtime"]["batch_size"], expected_batch
                        )
                        self.assertIsNone(report["model_weights_sha256"])
                        self.assertEqual(
                            len(
                                report["baseline"]["model_identity"][
                                    "runtime_snapshot"
                                ]["weight_files"]
                            ),
                            shards,
                        )
                        self.assertEqual(
                            report["model_identity_sha256"],
                            canary._canonical_sha256(
                                report["baseline"]["model_identity"]
                            ),
                        )
                        if slug.startswith("granite"):
                            self.assertIn("quality_at_native_cutoff_0_5", report)
                            self.assertNotIn("quality_at_fixed_cutoff_0_5", report)
                        else:
                            self.assertIn("quality_at_fixed_cutoff_0_5", report)
                            self.assertNotIn("quality_at_native_cutoff_0_5", report)

        self.assertEqual([baseline.batch_size for baseline in created], [4, 8, 2])
        self.assertTrue(all(baseline.loaded for baseline in created))
        self.assertTrue(all(baseline.unloaded for baseline in created))

    def test_source_identity_covers_direct_and_transitive_scoring_code(self) -> None:
        fake = Path("/does/not-need-to-exist")
        with patch.object(canary, "file_sha256", side_effect=lambda path: str(path)):
            result = canary._source_hashes(
                data_dir=fake,
                external_dir=fake,
                pairs=fake / "pairs",
                redteam=fake / "redteam",
            )

        self.assertIn("evaluation_source", result)
        self.assertIn("core_source", result)
        self.assertIn("detector_source", result)
        self.assertIn("inference_source", result)
        self.assertIn("normalization_source", result)
        self.assertIn("overlap_source", result)


if __name__ == "__main__":
    unittest.main()
