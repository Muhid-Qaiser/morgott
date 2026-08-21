from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import mutations


class MutationReplayTests(unittest.TestCase):
    @unittest.skipUnless(
        (mutations.retained.EVALUATION / "evaluation.json").is_file()
        and (mutations.retained.ROOT / "data" / "manifest.json").is_file(),
        "requires the retained frozen mutation artifacts and the canonical data root",
    )
    def test_frozen_population_reconstructs_without_persisted_text(self) -> None:
        panel, texts, population = mutations.build_population()

        self.assertEqual(population["frozen_base_rows"], 300)
        self.assertEqual(population["eligible_base_rows"], 240)
        self.assertEqual(population["mutation_rows"], 6_000)
        self.assertEqual(len(panel), 6_240)
        self.assertEqual(len(texts), 6_240)
        self.assertFalse(any("text" in row for row in panel))
        self.assertEqual(len({row["panel_id"] for row in panel}), 6_240)

    def test_summary_conditions_asr_on_current_high_gate_catches(self) -> None:
        records = []
        for base_index in range(2):
            base_id = f"base-{base_index}"
            records.append(
                {
                    "artifact_id": base_id,
                    "base_artifact_id": base_id,
                    "kind": "base",
                    "input_channel": "direct_user",
                    "local_route": "restrict" if base_index == 0 else "review",
                    "window_count": 1,
                }
            )
            for repeat in range(25):
                records.append(
                    {
                        "artifact_id": f"mutation-{base_index}-{repeat}",
                        "base_artifact_id": base_id,
                        "kind": "mutation",
                        "family": mutations.retained.FAMILIES[repeat // 5],
                        "input_channel": "direct_user",
                        "local_route": (
                            "pass" if base_index == 0 and repeat == 0 else "restrict"
                        ),
                        "window_count": 1,
                    }
                )
        population = {
            "eligible_base_rows": 2,
            "mutation_rows": 50,
        }

        summary = mutations.summarize(records, population)

        self.assertEqual(summary["base"]["high_gate_caught"], 1)
        self.assertAlmostEqual(
            summary["current_high_gate_caught_set"]["high_gate_asr_at_k"]["1"],
            1 / 25,
        )
        self.assertEqual(summary["provider_calls"], 0)
        self.assertIsNone(summary["full_cascade_asr"])


if __name__ == "__main__":
    unittest.main()
