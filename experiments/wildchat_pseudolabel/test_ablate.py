import unittest

from ablate import select, validate_accepted, weights


def accepted_row(sample_id="a", audited=False):
    stages = ["primary_a", "primary_b"] + (["third"] if audited else [])
    return {
        "sample_id": sample_id,
        "text": "ordinary user text",
        "label": 0,
        "weak_label": True,
        "label_basis": "cross_family_model_weak_label",
        "third_audited": audited,
        "judge_provenance": [
            {"stage": stage, "label": "benign", "confidence": "high"}
            for stage in stages
        ],
    }


class AblationTests(unittest.TestCase):
    def test_validation_requires_unanimous_high_benign(self):
        validate_accepted([accepted_row()])
        invalid = accepted_row()
        invalid["judge_provenance"][0]["confidence"] = "medium"
        with self.assertRaises(ValueError):
            validate_accepted([invalid])

    def test_selection_is_deterministic(self):
        rows = [accepted_row(str(index)) for index in range(20)]
        self.assertEqual(select(rows, 5), select(list(reversed(rows)), 5))

    def test_weak_rows_cannot_swamp_original_classes(self):
        value = weights(311, 35_601, 50_000)
        self.assertAlmostEqual(value["total_mass"], 35_912)
        self.assertAlmostEqual(value["base_positive_total_mass"], 17_956)
        self.assertAlmostEqual(value["base_negative_total_mass"], 16_160.4)
        self.assertAlmostEqual(value["weak_negative_total_mass"], 1_795.6)
        self.assertAlmostEqual(
            value["weak_negative_total_mass"]
            / (value["base_negative_total_mass"] + value["weak_negative_total_mass"]),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
