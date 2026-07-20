import unittest

from evaluate import (
    DIRECT_ATTACK_SETS,
    DIRECT_PRECISION_FLOORS,
    INDIRECT_ATTACK_SETS,
    NORMAL_SETS,
    scale_assessment,
)


def result(name, recall, indirect_recall, false_positive):
    points = []
    for floor in DIRECT_PRECISION_FLOORS:
        direct = {
            set_name: {"recall": recall, "false_positive": 0}
            for set_name in DIRECT_ATTACK_SETS
        }
        direct.update(
            {
                set_name: {"recall": None, "false_positive": false_positive}
                for set_name in NORMAL_SETS
            }
        )
        points.append(
            {
                "selection_kind": "min_validation_precision",
                "selection_value": floor,
                "direct_sets": direct,
                "untrusted_content_combined": {
                    set_name: {"recall": indirect_recall}
                    for set_name in INDIRECT_ATTACK_SETS
                },
            }
        )
    return {"name": name, "profiles": points}


class ScaleGateTests(unittest.TestCase):
    def test_scales_only_when_recall_improves_without_more_false_positives(self):
        baseline = result("zero", 0.5, 0.5, 2)
        candidate = result("pilot", 0.6, 0.5, 2)
        self.assertEqual(scale_assessment([baseline, candidate])["decision"], "scale")

        candidate["profiles"][1]["direct_sets"][NORMAL_SETS[0]]["false_positive"] = 3
        self.assertEqual(scale_assessment([baseline, candidate])["decision"], "stop")

    def test_equal_recall_stops_scaling(self):
        baseline = result("zero", 0.5, 0.5, 2)
        candidate = result("pilot", 0.5, 0.5, 1)
        self.assertEqual(scale_assessment([baseline, candidate])["decision"], "stop")


if __name__ == "__main__":
    unittest.main()
