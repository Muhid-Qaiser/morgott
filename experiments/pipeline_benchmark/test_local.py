from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.pipeline_benchmark import local


class LocalBenchmarkTests(unittest.TestCase):
    def test_load_frozen_texts_uses_exact_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "sources"
            source.mkdir(parents=True)
            text = "safe text"
            (source / "sample.jsonl").write_text(
                json.dumps({"id": "one", "text": text}) + "\n",
                encoding="utf-8",
            )
            panel = [
                {
                    "panel_id": "canonical:one",
                    "dataset": "canonical",
                    "source": "sample",
                    "row_id": "one",
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            ]
            with mock.patch.object(local, "external_rows", return_value=({}, {})):
                self.assertEqual(
                    local.load_frozen_texts(panel, root=root),
                    {"canonical:one": text},
                )

    def test_load_frozen_texts_rejects_changed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "sources"
            source.mkdir(parents=True)
            (source / "sample.jsonl").write_text(
                json.dumps({"id": "one", "text": "changed"}) + "\n",
                encoding="utf-8",
            )
            panel = [
                {
                    "panel_id": "canonical:one",
                    "dataset": "canonical",
                    "source": "sample",
                    "row_id": "one",
                    "text_sha256": "0" * 64,
                }
            ]
            with (
                mock.patch.object(local, "external_rows", return_value=({}, {})),
                self.assertRaisesRegex(ValueError, "frozen row changed"),
            ):
                local.load_frozen_texts(panel, root=root)

    def test_sigmoid_is_stable(self):
        values = local._sigmoid(np.asarray([-1_000.0, 0.0, 1_000.0]))

        self.assertEqual(values.tolist(), [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
