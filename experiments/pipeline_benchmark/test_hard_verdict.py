from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.pipeline_benchmark import hard_verdict


def _summary(
    provider: str,
    transport: str,
    *,
    recall: float,
    worst: float,
    p95: float,
    valid: float = 1.0,
    fpr: float = 0.01,
) -> dict:
    return {
        "provider": provider,
        "transport": transport,
        "valid_output_rate": valid,
        "quality": {"aggregate": {"recall": recall, "fpr": fpr}},
        "slice_recall": {"dataset=a": worst, "input_channel=direct_user": recall},
        "worst_slice_recall": worst,
        "latency_seconds": {"p95": p95},
        "cost_usd": "0.01",
    }


class HardVerdictAnalysisTests(unittest.TestCase):
    def test_window_ledger_filters_preselection_provider_superset(self) -> None:
        artifact_sha = hashlib.sha256(b"artifact").hexdigest()
        window_hashes = [
            hashlib.sha256(f"window-{i}".encode()).hexdigest() for i in range(2)
        ]
        common = {
            "stage": hard_verdict.WINDOW_STAGE,
            "row_id": "artifact",
            "artifact_id": "artifact",
            "artifact_text_sha256": artifact_sha,
            "requested_model": hard_verdict.providers.MODEL,
            "attempts": 1,
            "client_seconds": 0.1,
            "status": "ok",
            "returned_model": hard_verdict.providers.MODEL,
        }
        rows = []
        for provider, transport, index, verdict in (
            ("decart", "strict_hard_verdict", 1, 1),
            ("decart", "strict_hard_verdict", 0, 0),
            ("cloudflare", "strict_logprob", 1, 1),
        ):
            rows.append(
                common
                | {
                    "job_id": hashlib.sha256(
                        f"{provider}:{transport}:{index}".encode()
                    ).hexdigest(),
                    "requested_provider": provider,
                    "returned_provider": provider,
                    "transport": transport,
                    "window_index": index,
                    "char_start": index * 10,
                    "char_end": index * 10 + 10,
                    "text_sha256": window_hashes[index],
                    "window_text_sha256": window_hashes[index],
                    "local_score": [0.01, 0.2][index],
                    "window_local_score": [0.01, 0.2][index],
                    "verdict": verdict,
                }
            )
        with tempfile.TemporaryDirectory(dir=hard_verdict.ROOT) as directory:
            output = Path(directory)
            manifest_path = output / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            scores_path = output / "morgott_1024_scores.jsonl.gz"
            scores_path.write_bytes(b"scores")
            summary_path = output / "provider_summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            result_path = output / hard_verdict.WINDOW_RESULTS_NAME
            result_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            run_path = output / hard_verdict.WINDOW_RUN_NAME
            run_path.write_text(
                json.dumps(
                    {
                        "stage": hard_verdict.WINDOW_STAGE,
                        "providers": ["cloudflare", "decart"],
                        "model": hard_verdict.providers.MODEL,
                        "jobs": len(rows),
                        "result_path": str(result_path.relative_to(hard_verdict.ROOT)),
                        "result_sha256": hard_verdict._sha256(result_path),
                        "inputs": {
                            "manifest_sha256": hard_verdict._sha256(manifest_path),
                            "local_scores_sha256": hard_verdict._sha256(scores_path),
                            "provider_summary_sha256": hard_verdict._sha256(
                                summary_path
                            ),
                            "metrics_code_sha256": hard_verdict._sha256(
                                Path(hard_verdict.metrics.__file__)
                            ),
                            "model_key": hard_verdict.local.MODEL_KEY,
                            "max_tokens": hard_verdict.local.MAX_TOKENS,
                            "window_overlap": hard_verdict.local.WINDOW_OVERLAP,
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = hard_verdict._complete_window_ledger(
                output,
                panel={"artifact": {"text_sha256": artifact_sha}},
                scores={"artifact": {"window_scores": [0.01, 0.2]}},
            )

        self.assertIsNotNone(completed)
        values, _, _ = completed
        self.assertEqual(
            values[("decart", "strict_hard_verdict")],
            {("artifact", 0): False, ("artifact", 1): True},
        )
        self.assertEqual(
            values[("cloudflare", "strict_logprob")],
            {("artifact", 1): True},
        )

    def test_quality_rules_choose_strict_without_mixing_alternate_transport(
        self,
    ) -> None:
        summaries = [
            _summary(
                "strict-fast",
                "strict_hard_verdict",
                recall=0.90,
                worst=0.90,
                p95=1.0,
            ),
            _summary(
                "strict-slow",
                "strict_hard_verdict",
                recall=0.90,
                worst=0.90,
                p95=2.0,
            ),
            _summary(
                "tool-perfect",
                "forced_tool",
                recall=1.0,
                worst=1.0,
                p95=0.1,
            ),
        ]

        strict = hard_verdict.choose_provider(
            summaries, frozenset({"strict_hard_verdict"})
        )
        alternate = hard_verdict.choose_provider(
            summaries, hard_verdict.ALTERNATE_TRANSPORTS
        )

        self.assertEqual(strict["provider"], "strict-fast")
        self.assertEqual(alternate["provider"], "tool-perfect")

    def test_quality_and_reliability_gates_are_fail_closed(self) -> None:
        summaries = [
            _summary(
                "unreliable",
                "strict_hard_verdict",
                recall=1.0,
                worst=1.0,
                p95=0.1,
                valid=0.994,
            ),
            _summary(
                "slice-regression",
                "strict_hard_verdict",
                recall=0.99,
                worst=0.80,
                p95=0.2,
            ),
            _summary(
                "stable",
                "strict_hard_verdict",
                recall=0.98,
                worst=0.98,
                p95=1.0,
            ),
        ]

        selected = hard_verdict.choose_provider(
            summaries, frozenset({"strict_hard_verdict"})
        )

        self.assertEqual(selected["provider"], "stable")
        values, failures = hard_verdict._provider_values(
            ["ok", "failed"],
            [
                {"row_id": "ok", "status": "ok", "verdict": 0},
                {"row_id": "failed", "status": "failed", "verdict": None},
            ],
        )
        self.assertEqual(values, [False, None])
        self.assertEqual(failures, 1)

    def test_joint_selection_uses_exact_balanced_profile_across_providers(
        self,
    ) -> None:
        def candidate(
            provider: str,
            *,
            recall: float,
            slice_recall: dict[str, float],
            valid: float = 1.0,
        ) -> dict:
            balanced = {
                "metrics": {"recall": recall},
                "slice_recall": slice_recall,
                "worst_slice_recall": min(slice_recall.values()),
                "call_rate": 0.2,
            }
            return {
                "provider": provider,
                "valid_output_rate": valid,
                "latency_seconds": {"p95": 1.0},
                "cost_usd": "0.01",
                "profiles": {
                    "conservative": balanced,
                    "balanced": balanced,
                    "high_recall": balanced,
                },
            }

        selected = hard_verdict.choose_joint_provider(
            [
                candidate(
                    "raw-recall-winner",
                    recall=0.95,
                    slice_recall={"dataset=a": 0.95, "dataset=b": 0.80},
                ),
                candidate(
                    "exact-cascade-winner",
                    recall=0.94,
                    slice_recall={"dataset=a": 0.94, "dataset=b": 0.94},
                ),
                candidate(
                    "unreliable",
                    recall=1.0,
                    slice_recall={"dataset=a": 1.0, "dataset=b": 1.0},
                    valid=0.994,
                ),
            ]
        )

        self.assertEqual(selected["provider"], "exact-cascade-winner")

    def test_joint_selection_preserves_infeasible_conservative_profile(self) -> None:
        profile = {
            "metrics": {"recall": 0.9},
            "slice_recall": {"dataset=a": 0.9},
            "worst_slice_recall": 0.9,
            "call_rate": 0.2,
        }
        selected = hard_verdict.choose_joint_provider(
            [
                {
                    "provider": "strict",
                    "valid_output_rate": 1.0,
                    "latency_seconds": {"p95": 1.0},
                    "cost_usd": "0.01",
                    "profiles": {
                        "conservative": None,
                        "balanced": profile,
                        "high_recall": profile,
                    },
                }
            ]
        )

        self.assertEqual(selected["provider"], "strict")
        self.assertIsNone(selected["profiles"]["conservative"])

    def test_joint_selection_returns_none_when_every_provider_loses_a_source_slice(
        self,
    ) -> None:
        def candidate(provider: str, source_a: float, source_b: float) -> dict:
            profile = {
                "metrics": {"recall": 0.9},
                "slice_recall": {
                    "source=a": source_a,
                    "source=b": source_b,
                },
                "worst_slice_recall": min(source_a, source_b),
                "call_rate": 0.2,
            }
            return {
                "provider": provider,
                "valid_output_rate": 1.0,
                "latency_seconds": {"p95": 1.0},
                "cost_usd": "0.01",
                "profiles": {
                    "conservative": None,
                    "balanced": profile,
                    "high_recall": profile,
                },
            }

        selected = hard_verdict.choose_joint_provider(
            [candidate("a", 0.90, 0.80), candidate("b", 0.80, 0.90)]
        )

        self.assertIsNone(selected)

    def test_no_eligible_selection_finishes_without_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(dir=hard_verdict.ROOT) as directory:
            output = Path(directory)
            (output / hard_verdict.SELECTION_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "selection_status": "no_eligible_provider",
                        "provider": None,
                    }
                ),
                encoding="utf-8",
            )

            status = hard_verdict.analyze(output)

        self.assertEqual(status, "complete: no eligible strict hard-verdict provider")

    def test_exact_replay_preserves_full_context_and_batched_windows(self) -> None:
        rows = [
            {"artifact_id": "single", "input_channel": "direct_user"},
            {"artifact_id": "untrusted", "input_channel": "untrusted_content"},
            {"artifact_id": "all-low", "input_channel": "untrusted_content"},
            {"artifact_id": "direct", "input_channel": "direct_user"},
        ]
        score_records = {
            "single": {"window_scores": [0.2]},
            "untrusted": {"window_scores": [0.01, 0.2]},
            "all-low": {"window_scores": [0.01, 0.02]},
            "direct": {"window_scores": [0.2, 0.2, 0.2, 0.2, 0.2]},
        }
        artifact_verdicts = {
            "single": False,
            "untrusted": False,
            "all-low": False,
            "direct": True,
        }
        window_verdicts = {
            ("untrusted", 1): True,
            ("direct", 0): True,
            ("direct", 1): None,
            ("direct", 2): False,
            ("direct", 3): False,
        }
        selection = {
            "thresholds": {
                "direct_low": 0.1,
                "untrusted_low": 0.1,
                "local_high": 0.99,
            }
        }

        replay = hard_verdict.exact_cascade_predictions(
            rows,
            score_records,
            artifact_verdicts,
            window_verdicts,
            selection,
        )

        self.assertEqual(replay["predictions"].tolist(), [False, True, False, True])
        self.assertEqual(replay["artifact_calls"].tolist(), [1, 1, 1, 0])
        self.assertEqual(replay["window_calls"].tolist(), [0, 1, 0, 4])
        self.assertEqual(replay["invalid_reviews"].tolist(), [0, 0, 0, 1])

    @mock.patch.object(hard_verdict, "_complete_window_ledger", return_value=None)
    @mock.patch.object(hard_verdict, "_complete_ledger")
    @mock.patch.object(hard_verdict, "_selection_inputs")
    def test_evaluation_is_pending_without_text_free_window_ledger(
        self,
        selection_inputs: mock.Mock,
        complete_ledger: mock.Mock,
        _complete_window: mock.Mock,
    ) -> None:
        panel = {
            "single": {
                "panel_id": "single",
                "label": 0,
                "input_channel": "direct_user",
                "dataset": "d",
                "source": "s",
            },
            "multi": {
                "panel_id": "multi",
                "label": 1,
                "input_channel": "untrusted_content",
                "dataset": "d",
                "source": "s",
            },
        }
        scores = {
            "single": {"window_scores": [0.2]},
            "multi": {"window_scores": [0.01, 0.2]},
        }
        selection_inputs.return_value = (
            {"roles": {"provider_safe_evaluation_panel_ids": ["single", "multi"]}},
            panel,
            scores,
            [],
            {},
        )
        complete_ledger.return_value = (
            [
                {
                    "row_id": row_id,
                    "requested_provider": "strict",
                    "transport": "strict_hard_verdict",
                    "status": "ok",
                    "verdict": 0,
                }
                for row_id in ("single", "multi")
            ],
            {"sha256": "provider-evaluation"},
        )
        selection = {
            "schema_version": 3,
            "inputs": {},
            "profile_semantics": {
                "threshold_selection": "maintained_multi_window_exact"
            },
            "provider": {
                "name": "strict",
                "transport": "strict_hard_verdict",
            },
            "profiles": {
                "conservative": None,
                "balanced": {
                    "thresholds": {
                        "direct_low": 0.1,
                        "untrusted_low": 0.1,
                        "local_high": 0.99,
                    }
                },
            },
        }

        result, status = hard_verdict._evaluate(Path("unused"), selection)

        self.assertIsNone(result)
        self.assertIn("provider_cascade_windows_results", status)
        self.assertIn("1 required", status)

    @mock.patch.object(hard_verdict, "_exact_metrics", return_value={"exact": True})
    @mock.patch.object(hard_verdict, "_complete_window_ledger")
    @mock.patch.object(hard_verdict, "_complete_ledger")
    @mock.patch.object(hard_verdict, "_selection_inputs")
    def test_evaluation_keeps_infeasible_profile_null(
        self,
        selection_inputs: mock.Mock,
        complete_ledger: mock.Mock,
        complete_windows: mock.Mock,
        exact_metrics: mock.Mock,
    ) -> None:
        panel = {
            "row": {
                "panel_id": "row",
                "label": 0,
                "input_channel": "direct_user",
                "dataset": "d",
                "source": "s",
            }
        }
        selection_inputs.return_value = (
            {"roles": {"provider_safe_evaluation_panel_ids": ["row"]}},
            panel,
            {"row": {"window_scores": [0.2]}},
            [],
            {},
        )
        complete_ledger.return_value = (
            [
                {
                    "row_id": "row",
                    "requested_provider": "strict",
                    "transport": "strict_hard_verdict",
                    "status": "ok",
                    "verdict": 0,
                }
            ],
            {"sha256": "evaluation"},
        )
        window_identity = {"sha256": "windows"}
        complete_windows.return_value = (
            {("strict", "strict_hard_verdict"): {}},
            {},
            window_identity,
        )
        thresholds = {
            "direct_low": 0.1,
            "untrusted_low": 0.1,
            "local_high": 0.99,
        }
        selection = {
            "schema_version": 3,
            "inputs": {"provider_cascade_windows": window_identity},
            "profile_semantics": {
                "threshold_selection": "maintained_multi_window_exact"
            },
            "provider": {
                "name": "strict",
                "transport": "strict_hard_verdict",
            },
            "profiles": {
                "conservative": None,
                "balanced": {"thresholds": thresholds},
            },
        }
        with tempfile.TemporaryDirectory(dir=hard_verdict.ROOT) as directory:
            output = Path(directory)
            (output / hard_verdict.SELECTION_NAME).write_text("{}\n", encoding="utf-8")
            result, status = hard_verdict._evaluate(output, selection)

        self.assertEqual(status, "evaluation complete")
        self.assertIsNone(result["profiles"]["conservative"])
        self.assertEqual(result["profiles"]["balanced"], {"exact": True})
        exact_metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
