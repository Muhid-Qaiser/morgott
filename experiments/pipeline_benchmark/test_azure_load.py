import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.pipeline_benchmark.run import (
    _atomic_json,
    _azure_checkpoint_cells,
    _azure_cost_estimate,
    _azure_route_counts,
    _azure_text,
    _verify_azure_identity,
)
from morgott.models import downstream


class AzureLoadTests(unittest.TestCase):
    def test_measured_texts_are_unique_at_the_exact_byte_target(self):
        values = [
            _azure_text("review", 256, nonce=f"cell:{index}") for index in range(4)
        ]

        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(len(value.encode()) == 256 for value in values))
        self.assertEqual(
            _azure_text("review", 256, nonce="short").find("This document"),
            _azure_text("review", 256, nonce="a-much-longer-cell-id").find(
                "This document"
            ),
        )

    def test_deployment_identity_includes_registered_onnx(self):
        status = {
            "model_key": "mmbert-lora-full-ctx1024-u17000-s42",
            "onnx_sha256": "expected",
            "pipeline_profile": downstream.PIPELINE_PROFILE,
            "threshold_sha256": downstream.THRESHOLD_SHA256,
            "policy_sha256": "policy",
            "context_length": 1024,
            "window_overlap": 128,
        }
        _verify_azure_identity(status, "expected", "policy")
        status["onnx_sha256"] = "other"
        with self.assertRaises(RuntimeError):
            _verify_azure_identity(status, "expected", "policy")
        status["onnx_sha256"] = "expected"
        status["threshold_sha256"] = "old"
        with self.assertRaises(RuntimeError):
            _verify_azure_identity(status, "expected", "policy")
        status["threshold_sha256"] = downstream.THRESHOLD_SHA256
        status["policy_sha256"] = "other"
        with self.assertRaises(RuntimeError):
            _verify_azure_identity(status, "expected", "policy")

    def test_existing_azure_result_refuses_before_any_azure_lookup(self):
        from experiments.pipeline_benchmark import run

        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "azure_load.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.object(run, "_azure_value") as azure_value:
                with self.assertRaises(FileExistsError):
                    import asyncio

                    asyncio.run(run.azure_load(output, requests_per_cell=100))
        azure_value.assert_not_called()

    def test_remote_cost_ceiling_blocks_the_original_azure_matrix(self):
        cost = _azure_cost_estimate(
            [256, 6800, 27000, 61440],
            [64, 1024, 4096, 15360],
            requests_per_cell=100,
            input_per_million=Decimal("0.14"),
            output_per_million=Decimal("0.28"),
        )
        ninety_five = _azure_cost_estimate(
            [256, 6800, 27000, 61440],
            [64, 1024, 4096, 15360],
            requests_per_cell=95,
            input_per_million=Decimal("0.14"),
            output_per_million=Decimal("0.28"),
        )
        self.assertGreater(cost, 0)
        self.assertEqual(cost, ninety_five * Decimal(105) / Decimal(100))
        self.assertGreater(cost, Decimal("24"))

    def test_missing_route_survives_sorted_atomic_json(self):
        routes = _azure_route_counts([{"route": None}, {"route": "review"}])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            _atomic_json(path, {"routes": routes})
            rendered = path.read_text(encoding="utf-8")

        self.assertEqual(routes, {"missing": 1, "review": 1})
        self.assertIn('"missing": 1', rendered)

    def test_checkpoint_binds_identity_and_exact_cell_ids(self):
        identity = {"revision": "r1", "git_commit": "a" * 40}
        checkpoint = {
            "schema_version": 1,
            "benchmark_identity": identity,
            "cells": [{"cell_id": "allow:64:c1"}],
        }
        self.assertEqual(
            _azure_checkpoint_cells(checkpoint, identity, {"allow:64:c1"}),
            checkpoint["cells"],
        )
        checkpoint["cells"].append({"cell_id": "allow:64:c1"})
        with self.assertRaises(RuntimeError):
            _azure_checkpoint_cells(checkpoint, identity, {"allow:64:c1"})


if __name__ == "__main__":
    unittest.main()
