import unittest

import numpy as np

from morgott.models.routing_baseline import (
    MAX_BATCH_CHARACTERS,
    _cap_rows,
    _direct_strength,
    _metrics,
    _row_batches,
)


def _row(source: str, label: int, index: int) -> dict:
    return {
        "text": f"sample {index}",
        "label": label,
        "source": source,
        "group": f"{source}:{index}",
        "hash": f"{index:064x}",
        "strength": "strong",
    }


class RoutingBaselineTests(unittest.TestCase):
    def test_recipe_excludes_only_all_weak_origins(self):
        self.assertEqual(
            _direct_strength(
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
            ),
            "strong",
        )
        self.assertEqual(
            _direct_strength(
                {
                    "routing_label": 0,
                    "label_basis": "automated_weak_benign",
                }
            ),
            "weak_label",
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

    def test_metrics_use_the_untouched_cutoff(self):
        metrics = _metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.6, 0.4, 0.9]),
        )
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)


if __name__ == "__main__":
    unittest.main()
