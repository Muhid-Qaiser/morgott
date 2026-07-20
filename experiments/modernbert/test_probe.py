import json
from pathlib import Path

import numpy as np

from run_probe import (
    DIRECT_SETS,
    INDIRECT_SETS,
    aggregate_metrics,
    read_rows,
    validation_mask,
)


def test_group_split_and_deduplicated_aggregate() -> None:
    rows = [
        {"id": "1", "text": "same", "label": 0, "group_id": "g"},
        {"id": "2", "text": "same", "label": 0, "group_id": "g"},
    ]
    assert validation_mask(rows).tolist() in ([False, False], [True, True])
    metrics = aggregate_metrics(
        ("a", "b"),
        {"a": rows[:1], "b": rows[1:]},
        {"a": np.array([0.9]), "b": np.array([0.1])},
        0.5,
    )
    assert metrics["rows"] == 1
    assert metrics["cross_dataset_duplicates_removed"] == 1


def test_tensor_trust_is_locked_and_channel_scoped() -> None:
    direct = read_rows("tensor_trust_attack")
    indirect = read_rows("tensor_trust_context")
    assert "tensor_trust_attack" in DIRECT_SETS
    assert "tensor_trust_context" in INDIRECT_SETS
    assert len(direct) == 908 and {row["input_channel"] for row in direct} == {
        "direct_user"
    }
    assert len(indirect) == 1_346 and {row["input_channel"] for row in indirect} == {
        "untrusted_content"
    }


def test_result_routes_tensor_trust_through_both_poolings() -> None:
    result = json.loads((Path(__file__).parent / "results_512_cuda.json").read_text())
    assert result["evaluation_channels"]["direct_user"][-1] == "tensor_trust_attack"
    assert (
        result["evaluation_channels"]["untrusted_content"][-1] == "tensor_trust_context"
    )
    for pooling in ("masked_mean", "cls"):
        metrics = result["poolings"][pooling]
        assert metrics["direct"]["sets"]["tensor_trust_attack"]["rows"] == 908
        assert metrics["indirect"]["sets"]["tensor_trust_context"]["rows"] == 1_346
