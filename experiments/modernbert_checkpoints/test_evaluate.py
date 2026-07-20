import json
from pathlib import Path

from evaluate import DEFAULT_DATASETS, HARD_NEGATIVE_DATASETS, OPERATING_FPR_BUDGETS


def test_saved_checkpoint_curves_cover_the_common_suite() -> None:
    here = Path(__file__).parent
    for filename in ("siberiancat_cuda_results.json", "wolf_small_cuda_results.json"):
        result = json.loads((here / filename).read_text())
        assert set(HARD_NEGATIVE_DATASETS) <= set(result["sets"])
        assert "multi_turn" in result["sets"]
        assert [
            point["validation_fpr_budget"]
            for point in result["direct_operating_points"]
        ] == list(OPERATING_FPR_BUDGETS)
        assert result["default_precision_floor"] == 0.85
        assert [
            profile.get(
                "min_validation_precision", profile.get("validation_precision_floor")
            )
            for profile in result["direct_precision_profiles"]
        ] == [0.8, 0.85, 0.9, 0.95]
        assert not any(
            profile["attained"] for profile in result["direct_precision_profiles"]
        )
        assert set(DEFAULT_DATASETS) <= set(result["sets"])
