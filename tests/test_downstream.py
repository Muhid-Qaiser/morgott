import math
import unittest

from morgott.models.downstream import (
    LLM_FLAG_PROBABILITY,
    MMBERT_HIGH,
    PIPELINE_PROFILE,
    THRESHOLD_SHA256,
    route,
    subversion_probability,
)


class DownstreamRouteTests(unittest.TestCase):
    def test_promoted_balanced_profile(self):
        self.assertEqual(PIPELINE_PROFILE, "balanced-20260816")
        self.assertEqual(LLM_FLAG_PROBABILITY, 0.5)
        self.assertEqual(MMBERT_HIGH, 0.9999)
        self.assertEqual(
            THRESHOLD_SHA256,
            "e9b375b079667fd7c82bc5439fcde772ad3d3b04b7fb44277b8f180478558353",
        )

    def test_untrusted_content_uses_a_lower_review_floor(self):
        direct = route(0.025, input_channel="direct_user")
        untrusted = route(0.025, input_channel="untrusted_content")

        self.assertEqual((direct.route, direct.reason), ("pass", "mmbert_low"))
        self.assertEqual(
            (untrusted.route, untrusted.reason),
            ("review", "deepseek_required"),
        )

    def test_low_mmbert_score_passes_without_deepseek(self):
        result = route(0.199, input_channel="direct_user")

        self.assertEqual((result.route, result.reason), ("pass", "mmbert_low"))

    def test_high_mmbert_score_restricts_without_deepseek(self):
        result = route(0.9999, input_channel="direct_user")

        self.assertEqual((result.route, result.reason), ("restrict", "mmbert_high"))

    def test_middle_mmbert_score_requests_deepseek_review(self):
        for score in (0.2, math.nextafter(0.9999, 0.0)):
            with self.subTest(score=score):
                result = route(score, input_channel="direct_user")

                self.assertEqual(
                    (result.route, result.reason),
                    ("review", "deepseek_required"),
                )

    def test_middle_deepseek_probability_below_threshold_passes(self):
        result = route(
            0.5,
            input_channel="direct_user",
            llm_probability=math.nextafter(LLM_FLAG_PROBABILITY, 0.0),
        )

        self.assertEqual((result.route, result.reason), ("pass", "deepseek_clear"))

    def test_middle_deepseek_probability_at_threshold_restricts(self):
        result = route(
            0.5,
            input_channel="direct_user",
            llm_probability=LLM_FLAG_PROBABILITY,
        )

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
        result = route(0.5, input_channel="direct_user", llm_failed=True)

        self.assertEqual(
            (result.route, result.reason),
            ("restrict", "deepseek_failed"),
        )

    def test_invalid_mmbert_scores_are_rejected(self):
        for score in (-0.1, 1.1, math.nan, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                route(score, input_channel="direct_user")

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
                route(0.5, input_channel="direct_user", **arguments)

    def test_deepseek_result_is_only_valid_in_middle_zone(self):
        for score in (0.025, 0.9999):
            with self.subTest(score=score), self.assertRaises(ValueError):
                route(
                    score,
                    input_channel="direct_user",
                    llm_probability=0.5,
                )

    def test_middle_zone_floor_is_channel_specific(self):
        # 0.15 is below the 0.2 direct_user floor but inside the 0.025
        # untrusted_content middle zone.
        result = route(
            0.15,
            input_channel="untrusted_content",
            llm_probability=math.nextafter(LLM_FLAG_PROBABILITY, 0.0),
        )
        self.assertEqual(result.route, "pass")
        with self.assertRaises(ValueError):
            route(0.15, input_channel="direct_user", llm_probability=0.5)
        with self.assertRaises(ValueError):
            route(0.01, input_channel="untrusted_content", llm_probability=0.5)

    def test_invalid_input_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            route(0.5, input_channel="attacker_controlled")


if __name__ == "__main__":
    unittest.main()
