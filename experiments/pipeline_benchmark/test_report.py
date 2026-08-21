import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.pipeline_benchmark import report

_PANEL_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "loginject_long_span_panel"
    / "manifest.json"
)


class ReportTest(unittest.TestCase):
    # Anchored at the artifact, not the constant, so a typo'd constant fails
    # here instead of silently dropping the loginject context from reports.
    @unittest.skipUnless(
        _PANEL_MANIFEST.is_file(),
        "requires the retained loginject panel artifact",
    )
    def test_loginject_panel_manifest_constant_is_wired(self):
        self.assertEqual(report.LOGINJECT_PANEL_MANIFEST, _PANEL_MANIFEST)
        self.assertIsInstance(report._read_json(_PANEL_MANIFEST), dict)
    def test_failed_source_slice_gate_removes_hard_provider_winner(self):
        summary = {
            "winners": {
                "hard_verdict": {
                    "provider": "decart",
                    "transport": "strict_hard_verdict",
                }
            },
            "providers": {
                "decart:strict_hard_verdict": {
                    "provider": "decart",
                    "transport": "strict_hard_verdict",
                    "rows": 1,
                    "valid_output_rate": 1.0,
                    "quality": {"aggregate": {"recall": 1.0, "fpr": 0.0}},
                    "latency_seconds": {"p50": 1.0, "p95": 1.0},
                    "cost_usd": "0.01",
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            rows = report._provider_panel_rows(
                Path(temporary),
                summary,
                {"selection_status": "no_eligible_provider"},
            )

        self.assertIsNone(rows[0]["winner"])

    @staticmethod
    def _exact_metrics(rows=12):
        aggregate = {
            "rows": rows,
            "recall": 0.75,
            "recall_95": [0.5, 0.9],
            "fpr": 0.01,
            "fpr_95": [0.0, 0.05],
            "precision": 0.9,
            "precision_95": [0.7, 0.98],
            "restriction_rate": 0.3,
        }
        return {
            "aggregate": aggregate,
            "by_slice": {"dataset": {}, "input_channel": {}},
            "artifact_review_units": 3,
            "window_review_units": 2,
            "provider_review_units": 5,
            "artifacts_with_provider_review": 4,
            "invalid_called_reviews": 0,
            "prevalence_projections": {},
        }

    def test_missing_stages_are_explicit_and_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.md"
            tables_path = root / "tables.json"
            result = report.generate(root, report_path, tables_path)

            rendered = report_path.read_text(encoding="utf-8")
            tables = json.loads(tables_path.read_text(encoding="utf-8"))
            self.assertEqual(
                tables["evidence_status"]["azure_load"]["status"], "pending"
            )
            self.assertEqual(
                tables["evidence_status"]["production_traffic"]["status"],
                "unavailable",
            )
            self.assertEqual(
                tables["evidence_status"]["maintained_promotion"]["status"],
                "promoted_advisory_default",
            )
            self.assertEqual(tables["profiles"], [])
            self.assertIn("maintained advisory default", rendered)
            self.assertIn("no provider or quantization winner is reported", rendered)
            self.assertIn(
                "## Real-world defense layers and excluded comparators", rendered
            )
            self.assertIn(
                "https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall",
                rendered,
            )
            self.assertIn(
                "https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak",
                rendered,
            )
            self.assertNotIn("\N{EM DASH}", rendered)
            self.assertEqual(result["report_sha256"], report._sha256(report_path))

    def test_empty_provider_summary_is_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provider_summary.json").write_text(
                '{"providers": {}, "winners": {"hard_verdict": null, "logprob": null}}\n',
                encoding="utf-8",
            )

            tables = report.build_tables(root)

            self.assertEqual(
                tables["evidence_status"]["provider_panel"]["status"], "pending"
            )
            self.assertIsNone(tables["provider_panel"])

    def test_failed_channel_split_screen_is_reported_as_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = {
                "aggregate": {
                    "recall": 0.7,
                    "fpr": 0.02,
                    "precision": 0.9,
                    "tp": 70,
                    "positives": 100,
                    "fp": 2,
                    "negatives": 100,
                },
                "provider_calls": 20,
                "valid_outputs": 256,
                "slices": {"source": {"sep": {"recall": 0.5}}},
            }
            candidate = json.loads(json.dumps(arm))
            candidate["aggregate"].update(recall=0.5, fpr=0.01, tp=50, fp=1)
            candidate["slices"]["source"]["sep"]["recall"] = 0.2
            (root / "reviewer_channel_split_screen_summary.json").write_text(
                json.dumps(
                    {
                        "selection_eligible": False,
                        "rows": 256,
                        "recommendation": "do_not_proceed",
                        "current": arm,
                        "candidate": candidate,
                        "delta_candidate_minus_current": {
                            "recall": -0.2,
                            "fpr": -0.01,
                            "attack_detections": -20,
                            "false_restrictions": -1,
                            "source_recall": {"llmail": -0.01},
                        },
                    }
                ),
                encoding="utf-8",
            )

            rendered = report.render_report(report.build_tables(root))

            self.assertIn("rejected for further confirmation", rendered)

    def test_local_robustness_evidence_does_not_complete_remote_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mutation_1024_summary.json").write_text("{}\n", encoding="utf-8")
            (root / "loginject_summary.json").write_text(
                '{"remote_cascade_status": "pending_provider_winner"}\n',
                encoding="utf-8",
            )
            (root / "morgott_1024_openvino_runtime.json").write_text(
                '{"artifacts": 20000}\n', encoding="utf-8"
            )
            (root / "morgott_1024_openvino_scores.jsonl.gz").write_bytes(b"scores")
            (root / "gpt_oss_native_summary.json").write_text(
                '{"summary": {}}\n', encoding="utf-8"
            )

            tables = report.build_tables(root)

            self.assertEqual(
                tables["evidence_status"]["mutation_1024"]["status"], "measured"
            )
            self.assertEqual(
                tables["evidence_status"]["loginject_local_routing"]["status"],
                "measured_sealed_once",
            )
            self.assertEqual(
                tables["evidence_status"]["loginject_remote_cascade"]["status"],
                "pending",
            )
            self.assertEqual(
                tables["evidence_status"]["openvino_full_quality"]["status"],
                "measured_score_ledger_analysis_pending",
            )
            self.assertEqual(
                tables["evidence_status"]["gpt_oss_native_screen"]["status"],
                "measured_supplementary_256_row_screen",
            )

    def test_traffic_mix_projections_cover_all_mixes_and_prevalences(self):
        summary = {
            "current_1024_logprob_cascade": {
                "balanced": {
                    "by_input_channel": {
                        "direct_user": {"recall": 0.8, "fpr": 0.01},
                        "untrusted_content": {"recall": 0.6, "fpr": 0.03},
                    }
                }
            }
        }

        rows = report._traffic_mix_projections(summary)

        self.assertEqual(
            [row["traffic_mix"] for row in rows], ["90/10", "50/50", "20/80"]
        )
        self.assertTrue(
            all(
                list(row["prevalence_projections"])
                == ["0.0001", "0.001", "0.01", "0.05"]
                for row in rows
            )
        )
        self.assertAlmostEqual(rows[0]["mixed_recall"], 0.78)
        self.assertAlmostEqual(rows[0]["mixed_fpr"], 0.012)

    def test_loginject_posthoc_points_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"label": 0, "local_score": 0.1},
                {"label": 0, "local_score": 0.8},
                {"label": 1, "local_score": 0.4},
                {"label": 1, "local_score": 0.9},
            ]
            with gzip.open(
                root / "loginject_local_scores.jsonl.gz", "wt", encoding="utf-8"
            ) as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            diagnostic = report._loginject_posthoc_diagnostic(root)

            self.assertEqual(len(diagnostic), 6)
            self.assertTrue(all(not row["selection_eligible"] for row in diagnostic))
            self.assertTrue(
                all(
                    row["status"] == "sealed_post_hoc_diagnostic_only"
                    for row in diagnostic
                )
            )
            profile = {
                "clean_local_high_rate": 0.0,
                "clean_review_or_high_rate": 1.0,
                "attack_local_high_recall": 0.5,
                "attack_review_or_high_recall": 1.0,
            }
            (root / "loginject_summary.json").write_text(
                json.dumps(
                    {
                        "pairs": 2,
                        "paired_score_movement": {
                            "mean": 0.1,
                            "p50": 0.1,
                            "positive_rate": 0.5,
                        },
                        "profiles": {
                            "conservative": profile,
                            "balanced": profile,
                            "high_recall": profile,
                        },
                    }
                ),
                encoding="utf-8",
            )

            # The panel manifest is read from a repo-level retained artifact,
            # so point it at a fixture to keep this test hermetic.
            panel_manifest = root / "loginject_panel_manifest.json"
            panel_manifest.write_text(
                json.dumps(
                    {
                        "population": {"by_vector": {"complete_entry": 1}},
                        "selection": {
                            "fit_reference_rows": {
                                "canonical_train": 1,
                                "matched_pairs": 1,
                                "promptshield_train": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                report, "LOGINJECT_PANEL_MANIFEST", panel_manifest
            ):
                rendered = report.render_report(report.build_tables(root))

            self.assertIn("source-held-out out-of-distribution", rendered)
            self.assertIn("cannot tune, select, or revise", rendered)
            self.assertIn("only on that calibration role", rendered)
            self.assertIn("transport it unchanged", rendered)

    def test_openvino_quality_keeps_runtime_thresholds_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_ids = ["cal-n1", "cal-n2", "cal-p1", "cal-p2"]
            (root / "manifest.json").write_text(
                json.dumps({"roles": {"calibration_panel_ids": calibration_ids}}),
                encoding="utf-8",
            )
            records = [
                {"artifact_id": "cal-n1", "label": 0, "local_score": 0.1},
                {"artifact_id": "cal-n2", "label": 0, "local_score": 0.2},
                {"artifact_id": "cal-p1", "label": 1, "local_score": 0.8},
                {"artifact_id": "cal-p2", "label": 1, "local_score": 0.9},
                {"artifact_id": "eval-n1", "label": 0, "local_score": 0.15},
                {"artifact_id": "eval-n2", "label": 0, "local_score": 0.25},
                {"artifact_id": "eval-p1", "label": 1, "local_score": 0.75},
                {"artifact_id": "eval-p2", "label": 1, "local_score": 0.85},
            ]
            with gzip.open(
                root / "morgott_1024_openvino_scores.jsonl.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            cuda_evaluation = {
                "recall": 0.5,
                "fpr": 0.0,
                "precision": 1.0,
                "restriction_rate": 0.25,
                "auroc": 0.75,
                "average_precision": 0.75,
            }
            summary = {
                "current_1024_standalone": {
                    "0.01": {
                        "threshold": 0.7,
                        "evaluation": cuda_evaluation,
                    }
                }
            }

            rows = report._openvino_fixed_fpr(root, summary)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cuda"]["threshold"], 0.7)
            self.assertNotEqual(
                rows[0]["cuda"]["threshold"], rows[0]["openvino"]["threshold"]
            )
            self.assertAlmostEqual(
                rows[0]["delta_openvino_minus_cuda"]["recall"],
                rows[0]["openvino"]["evaluation"]["recall"] - 0.5,
            )

    def test_exact_evaluation_is_required_and_null_hard_profile_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thresholds = {
                "direct_low": 0.2,
                "untrusted_low": 0.025,
                "local_high": 0.9999,
                "reviewer": 0.5,
            }
            selected = {"thresholds": thresholds}
            (root / "logprob_exact_selection.json").write_text(
                json.dumps(
                    {
                        "advisory_only": True,
                        "profile_semantics": "maintained_multi_window_exact",
                        "profiles": {
                            "conservative": selected,
                            "balanced": selected,
                            "high_recall": selected,
                        },
                    }
                ),
                encoding="utf-8",
            )

            pending = report.build_tables(root)

            self.assertEqual(
                pending["evidence_status"]["exact_logprob_cascade"]["status"],
                "selection_frozen_evaluation_pending",
            )
            self.assertEqual(pending["exact_logprob_profiles"], [])

            exact = self._exact_metrics()
            (root / "logprob_exact_evaluation.json").write_text(
                json.dumps(
                    {
                        "advisory_only": True,
                        "evaluation_semantics": "maintained_multi_window_exact",
                        "rows": 12,
                        "profiles": {
                            "conservative": exact,
                            "balanced": exact,
                            "high_recall": exact,
                        },
                    }
                ),
                encoding="utf-8",
            )
            hard_selected = {
                "thresholds": thresholds | {"reviewer": None},
            }
            (root / "hard_verdict_selection.json").write_text(
                json.dumps(
                    {
                        "advisory_only": True,
                        "profile_semantics": {
                            "threshold_selection": "maintained_multi_window_exact",
                            "end_to_end_exact": True,
                        },
                        "profiles": {
                            "conservative": None,
                            "balanced": hard_selected,
                            "high_recall": hard_selected,
                        },
                        "selected_profile_infeasibility": {
                            "conservative": {
                                "minimum_observed_fpr": 0.012,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "hard_verdict_evaluation.json").write_text(
                json.dumps(
                    {
                        "advisory_only": True,
                        "evaluation_semantics": "maintained_multi_window_exact",
                        "rows": 12,
                        "profiles": {
                            "conservative": None,
                            "balanced": exact,
                            "high_recall": exact,
                        },
                    }
                ),
                encoding="utf-8",
            )

            tables = report.build_tables(root)
            rendered = report.render_report(tables)

            self.assertEqual(
                tables["evidence_status"]["exact_logprob_cascade"]["status"],
                "measured_maintained_multi_window_exact",
            )
            self.assertEqual(
                tables["exact_hard_verdict_profiles"][0]["status"], "infeasible"
            )
            self.assertIn("12 provider-safe evaluation artifacts", rendered)
            self.assertIn("minimum observed FPR 1.20%", rendered)

    def test_completed_load_remote_and_azure_artifacts_expose_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provider_load_confounded_length_band_old.json").write_text(
                '{"cells": [{"requests": 99}]}', encoding="utf-8"
            )
            pending = report.build_tables(root)
            self.assertEqual(
                pending["evidence_status"]["provider_load"]["status"], "pending"
            )

            load_cell = {
                "cell_id": "cloudflare:strict_logprob:c4",
                "concurrency": 4,
                "requests": 16,
                "terminal_failures": 0,
                "terminal_failure_rate": 0.0,
                "requests_per_second": 2.0,
                "input_tokens_per_second": 2048.0,
                "length_bands": {
                    "<1024": 4,
                    "1024-4095": 4,
                    "4096-15999": 4,
                    ">=16000": 4,
                },
                "latency_seconds": {"p50": 0.5, "p95": 1.0, "p99": 1.2},
                "artifact_review_units": 0,
                "cost_usd": "0.01",
            }
            (root / "provider_load.json").write_text(
                json.dumps(
                    {
                        "samples_are_unique_across_cells": True,
                        "cells": [load_cell],
                    }
                ),
                encoding="utf-8",
            )
            remote_profile = {
                "pairs": 10,
                "paired_clean_false_restrictions": {"count": 1, "rate": 0.1},
                "attack_recall": {"detected": 8, "total": 10, "recall": 0.8},
                "paired_outcomes": {
                    "attack_restricted_clean_clear": 7,
                    "both_restricted": 1,
                    "both_clear": 2,
                    "attack_clear_clean_restricted": 0,
                },
                "calls": 20,
                "failures": 0,
            }
            (root / "loginject_remote_summary.json").write_text(
                json.dumps(
                    {
                        "sealed_once": True,
                        "pairs": 10,
                        "profiles": {
                            "balanced": remote_profile,
                            "incumbent": remote_profile,
                        },
                        "unique_provider_calls": 20,
                        "terminal_failures": 0,
                    }
                ),
                encoding="utf-8",
            )
            azure_cell = {
                "cell_id": "review:4096:c4",
                "kind": "review",
                "input_channel": "untrusted_content",
                "input_tokens": {
                    "minimum": 1020,
                    "mean": 1024.5,
                    "maximum": 1029,
                    "total": 102450,
                },
                "input_bytes": 4096,
                "concurrency": 4,
                "requests": 100,
                "successes": 99,
                "requests_per_second": 3.0,
                "input_tokens_per_second": 3072.0,
                "latency_seconds": {"p50": 0.5, "p95": 1.0, "p99": 1.5},
                "routes": {"deepseek_review": 99},
                "deepseek_calls": 99,
            }
            saturation_cells = [
                {
                    **azure_cell,
                    "cell_id": "allow:61440:c16",
                    "kind": "allow",
                    "input_channel": "direct_user",
                    "input_bytes": 61440,
                    "successes": 100,
                    "routes": {"pass": 10, "restrict": 90},
                    "deepseek_calls": 110,
                },
                {
                    **azure_cell,
                    "cell_id": "high:61440:c16",
                    "kind": "high",
                    "input_bytes": 61440,
                    "successes": 15,
                    "routes": {"missing": 85, "restrict": 15},
                    "deepseek_calls": 0,
                },
                {
                    **azure_cell,
                    "cell_id": "review:61440:c16",
                    "input_bytes": 61440,
                    "successes": 99,
                    "routes": {"missing": 1, "pass": 99},
                    "deepseek_calls": 1089,
                },
            ]
            (root / "azure_load.json").write_text(
                json.dumps(
                    {
                        "status": {
                            "model_key": "registered-1024",
                            "context_length": 1024,
                            "window_overlap": 128,
                            "onnx_sha256": "abc",
                        },
                        "estimated_remote_cost_usd": "0.5",
                        "prior_failed_azure_estimate_usd": "0.4",
                        "cells": [azure_cell, *saturation_cells],
                        "resource_metrics": {
                            "value": [
                                {
                                    "name": {"value": "WorkingSetBytes"},
                                    "unit": "Bytes",
                                    "timeseries": [
                                        {
                                            "data": [
                                                {
                                                    "maximum": 100.0,
                                                    "average": 80.0,
                                                    "total": 200.0,
                                                },
                                                {
                                                    "maximum": 90.0,
                                                    "average": 70.0,
                                                    "total": 180.0,
                                                },
                                            ]
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            tables = report.build_tables(root)
            rendered = report.render_report(tables)

            self.assertEqual(
                tables["provider_load_rows"][0]["length_bands"][">=16000"], 4
            )
            self.assertEqual(
                tables["evidence_status"]["loginject_remote_cascade"]["status"],
                "measured_sealed_once",
            )
            self.assertEqual(tables["azure_resource_metrics"][0]["maximum"], 100.0)
            self.assertNotIn("resource_metrics", tables["azure_load"])
            self.assertEqual(
                tables["azure_resource_metrics"][0]["maximum_total"], 200.0
            )
            self.assertIn("99/100", rendered)
            self.assertIn("1,020/1024.5/1,029", rendered)
            self.assertIn("102,450", rendered)
            self.assertIn("queueing-inclusive burst latencies", rendered)
            self.assertIn("not strict upper bounds", rendered)
            self.assertIn("cannot support a scaling", rendered)
            self.assertIn("did not measure a true local-allow path", rendered)
            self.assertIn("not evidence that two replicas ran concurrently", rendered)
            self.assertIn("attack restricted and clean clear", rendered.lower())
            self.assertIn("concurrency-8 execution failed", rendered)
            self.assertIn("resumed at concurrency 4", rendered)
            self.assertIn("terminal failures fail closed", rendered)


if __name__ == "__main__":
    unittest.main()
