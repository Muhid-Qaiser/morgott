import math
import unittest

from morgott.models.downstream import route, subversion_probability


class DownstreamRouteTests(unittest.TestCase):
    def test_low_mmbert_score_passes_without_deepseek(self):
        result = route(0.199)

        self.assertEqual((result.route, result.reason), ("pass", "mmbert_low"))

    def test_high_mmbert_score_restricts_without_deepseek(self):
        result = route(0.999)

        self.assertEqual((result.route, result.reason), ("restrict", "mmbert_high"))

    def test_middle_mmbert_score_requests_deepseek_review(self):
        result = route(0.2)

        self.assertEqual(
            (result.route, result.reason),
            ("review", "deepseek_required"),
        )

    def test_middle_deepseek_probability_below_threshold_passes(self):
        result = route(0.5, llm_probability=0.899)

        self.assertEqual((result.route, result.reason), ("pass", "deepseek_clear"))

    def test_middle_deepseek_probability_at_threshold_restricts(self):
        result = route(0.5, llm_probability=0.9)

        self.assertEqual(
            (result.route, result.reason),
            ("restrict", "deepseek_flag"),
        )

    def test_decision_logprobs_become_a_stable_binary_probability(self):
        self.assertAlmostEqual(subversion_probability(0.0, 0.0), 0.5)
        self.assertAlmostEqual(
            subversion_probability(math.log(0.1), math.log(0.9)),
            0.9,
        )
        self.assertEqual(subversion_probability(0.0, -1000.0), 0.0)
        self.assertEqual(subversion_probability(-1000.0, 0.0), 1.0)

    def test_exhausted_middle_deepseek_failure_restricts(self):
        result = route(0.5, llm_failed=True)

        self.assertEqual(
            (result.route, result.reason),
            ("restrict", "deepseek_failed"),
        )

    def test_invalid_mmbert_scores_are_rejected(self):
        for score in (-0.1, 1.1, math.nan, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                route(score)

    def test_invalid_deepseek_results_are_rejected(self):
        invalid = (
            {"llm_probability": math.nan},
            {"llm_probability": -0.1},
            {"llm_probability": 1.1},
            {"llm_probability": 0.5, "llm_failed": True},
            {"llm_failed": "yes"},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                route(0.5, **arguments)

    def test_deepseek_result_is_only_valid_in_middle_zone(self):
        for score in (0.1, 0.999):
            with self.subTest(score=score), self.assertRaises(ValueError):
                route(score, llm_probability=0.5)


if __name__ == "__main__":
    unittest.main()
