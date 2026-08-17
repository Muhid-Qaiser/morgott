from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from experiments.pipeline_benchmark import loginject_remote, providers, run


def _review(artifact_id: str, index: int | None) -> loginject_remote.Review:
    review_index = 0 if index is None else index + 1
    return loginject_remote.Review(
        job_id=f"{artifact_id}:{review_index}",
        artifact_id=artifact_id,
        pair_id=artifact_id.rsplit(":", 1)[0],
        variant=artifact_id.rsplit(":", 1)[1],
        attack_level=1,
        injection_vector="complete_entry",
        review_index=review_index,
        window_index=index,
        char_start=0 if index is None else index * 10,
        char_end=100 if index is None else index * 10 + 20,
        artifact_text_sha256="a" * 64,
        review_text_sha256=f"{review_index:064x}",
        estimated_input_tokens=100,
        text="not persisted",
    )


def _artifact(artifact_id: str, scores: tuple[float, ...]) -> loginject_remote.Artifact:
    return loginject_remote.Artifact(
        row={
            "panel_id": artifact_id,
            "pair_id": artifact_id.rsplit(":", 1)[0],
            "variant": artifact_id.rsplit(":", 1)[1],
            "attack_level": 1,
            "injection_vector": "complete_entry",
        },
        scores=scores,
        full=_review(artifact_id, None),
        windows=tuple(_review(artifact_id, index) for index in range(len(scores))),
    )


def _ok(probability: float) -> dict:
    return {"status": "ok", "probability": probability}


class LogInjectRemoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = (
            loginject_remote.Profile("balanced", 0.1, 0.05, 0.99999, 0.6),
            loginject_remote.Profile("incumbent", 0.2, 0.1, 0.99999, 0.6),
        )

    def test_execute_rejects_unbounded_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "concurrency"):
            asyncio.run(loginject_remote.execute(Path("."), Path("."), concurrency=2))

    def test_profiles_use_balanced_selection_and_exact_incumbent(self) -> None:
        selection = {
            "schema_version": 1,
            "advisory_only": True,
            "frozen_from": "provider-safe 1024-row calibration panel and windows only",
            "profile_semantics": "maintained_multi_window_exact",
            "provider": {
                "name": "cloudflare",
                "transport": "strict_logprob",
            },
            "profiles": {
                "balanced": {
                    "semantics": "maintained_multi_window_exact",
                    "metrics": {"rows": 1024},
                    "thresholds": {
                        "direct_low": 0.2,
                        "untrusted_low": 0.025,
                        "local_high": 0.9999,
                        "reviewer": 0.5,
                    },
                }
            },
        }
        profiles = loginject_remote._profiles(selection)

        self.assertEqual(profiles[0].name, "balanced")
        self.assertEqual(profiles[0].reviewer, 0.5)
        self.assertEqual(profiles[0].local_high, 0.9999)
        self.assertEqual(profiles[1].name, "incumbent")
        self.assertEqual(profiles[1].reviewer, 0.6224593312018547)

        with self.assertRaisesRegex(ValueError, "unavailable or pending"):
            loginject_remote._profiles(selection | {"profile_semantics": "approximate"})
        pending = selection | {"profiles": {"balanced": None}}
        with self.assertRaisesRegex(ValueError, "unavailable or pending"):
            loginject_remote._profiles(pending)

    def test_shared_reviews_preserve_each_profiles_ordered_batch(self) -> None:
        artifact = _artifact("pair:attack", (0.07, 0.2, 0.2, 0.2, 0.2))
        outcomes, needed = loginject_remote.resolve(artifact, self.profiles, {})
        self.assertEqual(outcomes, {})
        self.assertEqual(needed, (artifact.full,))

        records = {artifact.full.job_id: _ok(0.1)}
        outcomes, needed = loginject_remote.resolve(artifact, self.profiles, records)
        self.assertEqual(outcomes, {})
        self.assertEqual(
            {review.window_index for review in needed},
            {0, 1, 2, 3, 4},
        )

        records.update(
            {
                review.job_id: _ok(0.7 if review.window_index == 0 else 0.1)
                for review in needed
            }
        )
        outcomes, needed = loginject_remote.resolve(artifact, self.profiles, records)
        self.assertFalse(needed)
        self.assertEqual(outcomes["balanced"], loginject_remote.Outcome(True, 5, 0))
        self.assertEqual(outcomes["incumbent"], loginject_remote.Outcome(False, 5, 0))

    def test_failure_and_local_high_resolve_conservatively_without_extra_calls(
        self,
    ) -> None:
        artifact = _artifact("pair:clean", (0.2, 0.3))
        failed = {
            artifact.full.job_id: {
                "status": "failed",
                "probability": None,
            }
        }
        outcomes, needed = loginject_remote.resolve(artifact, self.profiles, failed)
        self.assertFalse(needed)
        self.assertTrue(all(value.restricted for value in outcomes.values()))
        self.assertTrue(all(value.failures == 1 for value in outcomes.values()))

        high = _artifact("pair:attack", (1.0, 0.01))
        outcomes, needed = loginject_remote.resolve(high, self.profiles, {})
        self.assertFalse(needed)
        self.assertTrue(
            all(
                value == loginject_remote.Outcome(True, 0, 0)
                for value in outcomes.values()
            )
        )

    def test_shared_budget_includes_loginject_remote_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / loginject_remote.RESULT_NAME).write_text(
                json.dumps(
                    {
                        "stage": loginject_remote.STAGE,
                        "cost_usd": "0.75",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output / "gpt_oss_native_results.jsonl").write_text(
                json.dumps({"cost_usd": "0.25"}) + "\n", encoding="utf-8"
            )

            self.assertEqual(run._recorded_spend(output), Decimal("1.00"))

    def test_parsed_ledger_rejects_raw_provider_content(self) -> None:
        review = _review("pair:attack", 0)
        endpoint = providers.Endpoint(
            provider="cloudflare",
            name="Cloudflare",
            tag="cloudflare",
            model=providers.MODEL,
            quantization="fp8",
            uptime_percent=100,
            supported_parameters=frozenset(
                {"response_format", "structured_outputs", "logprobs", "top_logprobs"}
            ),
            input_per_million_usd=Decimal("0.14"),
            output_per_million_usd=Decimal("0.28"),
            cache_read_per_million_usd=Decimal("0.028"),
        )
        record = {
            "stage": loginject_remote.STAGE,
            "job_id": review.job_id,
            "row_id": review.artifact_id,
            "pair_id": review.pair_id,
            "variant": review.variant,
            "attack_level": review.attack_level,
            "injection_vector": review.injection_vector,
            "review_index": review.review_index,
            "review_kind": "window",
            "window_index": review.window_index,
            "char_start": review.char_start,
            "char_end": review.char_end,
            "artifact_text_sha256": review.artifact_text_sha256,
            "text_sha256": review.review_text_sha256,
            "review_text_sha256": review.review_text_sha256,
            "estimated_input_tokens": review.estimated_input_tokens,
            "requested_provider": "cloudflare",
            "returned_provider": "Cloudflare",
            "requested_model": providers.MODEL,
            "returned_model": providers.MODEL,
            "transport": "strict_logprob",
            "endpoint_tag": endpoint.tag,
            "endpoint_quantization": endpoint.quantization,
            "attempts": 1,
            "client_seconds": 0.5,
            "status": "ok",
            "verdict": 0,
            "probability": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / loginject_remote.RESULT_NAME
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            parsed = loginject_remote._validate_records(
                path, {review.job_id: review}, endpoint=endpoint
            )
            self.assertEqual(set(parsed), {review.job_id})

            path.write_text(
                json.dumps(record | {"content": "raw"}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                loginject_remote._validate_records(
                    path, {review.job_id: review}, endpoint=endpoint
                )

    def test_summary_reports_paired_and_attack_slice_denominators(self) -> None:
        clean = _artifact("pair:clean", (0.2, 0.3))
        attack = _artifact("pair:attack", (0.2, 0.3))
        outcomes = {
            "balanced": {
                "pair:clean": loginject_remote.Outcome(False, 3, 0),
                "pair:attack": loginject_remote.Outcome(True, 2, 1),
            },
            "incumbent": {
                "pair:clean": loginject_remote.Outcome(True, 1, 1),
                "pair:attack": loginject_remote.Outcome(True, 1, 0),
            },
        }

        result = loginject_remote._metrics([clean, attack], outcomes, self.profiles)

        self.assertEqual(
            result["balanced"]["paired_clean_false_restrictions"],
            {"count": 0, "rate": 0.0},
        )
        self.assertEqual(result["balanced"]["attack_recall"]["recall"], 1.0)
        self.assertEqual(result["balanced"]["calls"], 5)
        self.assertEqual(result["balanced"]["by_attack_level"]["1"]["rows"], 1)
        self.assertEqual(
            result["balanced"]["by_injection_vector"]["complete_entry"]["failures"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
