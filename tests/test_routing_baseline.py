import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from morgott.models import routing_baseline
from morgott.models.routing_baseline import (
    MAX_BATCH_CHARACTERS,
    SEED,
    _cap_rows,
    _evaluate,
    _fit,
    _is_weak_label,
    _metrics,
    _row_batches,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(source: str, label: int, index: int) -> dict:
    return {
        "text": f"sample {index}",
        "label": label,
        "source": source,
        "group": f"{source}:{index}",
        "hash": f"{index:064x}",
    }


class RoutingBaselineTests(unittest.TestCase):
    def test_versioned_report_tracks_the_canonical_manifest(self):
        manifest = (ROOT / "data" / "manifest.json").read_bytes()
        report = json.loads((ROOT / "reports" / "routing-baseline.json").read_text())

        self.assertEqual(
            report["data_manifest_sha256"], hashlib.sha256(manifest).hexdigest()
        )

    def test_recipe_excludes_only_all_weak_origins(self):
        self.assertFalse(
            _is_weak_label(
                {
                    "routing_label": 0,
                    "origins": [
                        {
                            "source": "false_reject",
                            "routing_label": 0,
                            "label_basis": "multi_agent_generated_benign_weak_label",
                        },
                        {
                            "source": "banking77",
                            "routing_label": 0,
                            "label_basis": "banking_assistant_intent_collection",
                        },
                    ],
                }
            )
        )
        self.assertTrue(
            _is_weak_label(
                {
                    "routing_label": 0,
                    "label_basis": "automated_weak_benign",
                }
            )
        )

    def test_cap_is_deterministic_per_source_and_label(self):
        rows = [
            _row(source, label, label * 100 + index + 1)
            for source, label in (("benign", 0), ("attack", 1))
            for index in range(5)
        ]
        first, stats = _cap_rows(rows, 3)
        second, _ = _cap_rows(reversed(rows), 3)
        self.assertEqual(
            [row["hash"] for row in first], [row["hash"] for row in second]
        )
        self.assertEqual(stats["selected_rows"], 6)

    def test_batches_do_not_truncate_long_text(self):
        rows = [
            {"text": "a" * (MAX_BATCH_CHARACTERS + 1)},
            {"text": "short"},
            {"text": "also short"},
        ]
        self.assertEqual(
            [len(batch) for batch in _row_batches(rows)],
            [1, 2],
        )

    def test_fit_coefficients_match_per_batch_transforms(self):
        from sklearn.base import clone

        rng = np.random.default_rng(7)
        words = ["alpha", "beta", "ignore", "previous", "instructions", "please"]
        rows = [
            {
                "text": " ".join(rng.choice(words, size=int(rng.integers(3, 12)))),
                "label": index % 2,
            }
            for index in range(48)
        ]

        # A small batch size forces several partial_fit calls per epoch, so a
        # reordering or boundary slip in _fit would change the coefficients.
        with mock.patch.object(routing_baseline, "BATCH_SIZE", 8):
            vectorizer, fitted = _fit(rows, epochs=2)
            reference = clone(fitted)
            for epoch in range(2):
                order = np.random.default_rng(SEED + epoch).permutation(len(rows))
                shuffled = (rows[int(index)] for index in order)
                for batch in _row_batches(shuffled):
                    reference.partial_fit(
                        vectorizer.transform([row["text"] for row in batch]),
                        np.asarray([row["label"] for row in batch], dtype=np.int8),
                        classes=np.asarray([0, 1]),
                    )

        self.assertEqual(fitted.t_, reference.t_)
        self.assertTrue(np.array_equal(fitted.coef_, reference.coef_))
        self.assertTrue(np.array_equal(fitted.intercept_, reference.intercept_))

    def test_metrics_use_the_untouched_cutoff(self):
        metrics = _metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.6, 0.4, 0.9]),
        )
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertEqual(
            {
                key: metrics[key]
                for key in (
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                )
            },
            {
                "true_positive": 1,
                "false_positive": 1,
                "true_negative": 1,
                "false_negative": 1,
            },
        )
        self.assertAlmostEqual(
            metrics["expected_precision_at_attack_prevalence"]["0.1%"],
            0.001,
        )

    def test_single_class_metrics_keep_prevalence_report_shape(self):
        metrics = _metrics(np.asarray([0, 0]), np.asarray([0.1, 0.2]))

        self.assertEqual(
            metrics["expected_precision_at_attack_prevalence"],
            {"0.1%": None, "1%": None, "5%": None},
        )

    def test_source_metrics_include_every_exact_merge_membership(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Classifier:
            def predict_proba(self, texts):
                return np.asarray([[0.1, 0.9] for _ in texts])

        result = _evaluate(
            [
                {
                    "text": "merged attack",
                    "label": 1,
                    "source": "representative",
                    "sources": ("representative", "second_source"),
                }
            ],
            Vectorizer(),
            Classifier(),
        )

        self.assertEqual(result["all"]["rows"], 1)
        self.assertEqual(result["by_source"]["representative"]["rows"], 1)
        self.assertEqual(result["by_source"]["second_source"]["rows"], 1)
        self.assertEqual(result["by_normalized_character_length"]["0-256"]["rows"], 1)

    def test_length_slices_keep_boundaries_disjoint(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Classifier:
            def predict_proba(self, texts):
                return np.asarray([[0.1, 0.9] for _ in texts])

        result = _evaluate(
            [
                {"text": "a" * 256, "label": 1, "source": "source"},
                {"text": "a" * 257, "label": 1, "source": "source"},
                {"text": "a" * 4_097, "label": 1, "source": "source"},
            ],
            Vectorizer(),
            Classifier(),
        )

        self.assertEqual(
            {
                name: metrics["rows"]
                for name, metrics in result["by_normalized_character_length"].items()
            },
            {"0-256": 1, "257-1024": 1, "1025-4096": 0, "4097+": 1},
        )


if __name__ == "__main__":
    unittest.main()
