from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp

from experiments.pipeline_benchmark import providers
from experiments.pipeline_benchmark import run as benchmark


def _snapshot() -> dict:
    return {
        "data": {
            "id": providers.MODEL,
            "endpoints": [
                {
                    "name": "Cloudflare",
                    "tag": "cloudflare",
                    "quantization": "unknown",
                    "uptime_last_30m": 0.9999,
                    "supported_parameters": [
                        "response_format",
                        "structured_outputs",
                        "logprobs",
                        "top_logprobs",
                    ],
                    "pricing": {
                        "prompt": "0.00000014",
                        "completion": "0.00000028",
                        "input_cache_read": "0.000000028",
                    },
                },
                {
                    "name": "CoreWeave",
                    "tag": "coreweave/fp8",
                    "quantization": "fp8",
                    "uptime_last_30m": 99.95,
                    "supported_parameters": [
                        "tools",
                        "tool_choice",
                        "logprobs",
                    ],
                    "pricing": {},
                },
                {
                    "name": "DigitalOcean",
                    "tag": "digitalocean/fp8",
                    "quantization": "fp8",
                    "uptime_last_30m": 96.7,
                    "supported_parameters": ["response_format", "logprobs"],
                    "pricing": {},
                },
            ],
        }
    }


def _payload(content: str = '{"subversion":1}') -> dict:
    return {
        "model": providers.MODEL,
        "provider": "Cloudflare",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
                "logprobs": {
                    "content": [
                        {
                            "token": "1",
                            "bytes": [49],
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "1", "bytes": [49], "logprob": -0.1},
                                {"token": "0", "bytes": [48], "logprob": -2.1},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 10},
            "cost": 0.00001,
        },
    }


class ProviderBenchmarkTest(unittest.TestCase):
    def test_benchmark_prepare_rejects_uncommitted_source(self) -> None:
        dirty = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="?? experiments/pipeline_benchmark/run.py\n"
        )
        with patch.object(benchmark.subprocess, "run", return_value=dirty):
            with self.assertRaisesRegex(RuntimeError, "committed benchmark source"):
                benchmark._require_committed_benchmark_source()

    def test_budget_reservations_are_durable_and_preserve_the_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            benchmark._reserve_budget(output, "phase-a", Decimal("20"))
            benchmark._reserve_budget(output, "phase-a", Decimal("19"))
            benchmark._reserve_budget(output, "phase-b", Decimal("4"))
            with self.assertRaisesRegex(RuntimeError, "reserved budget"):
                benchmark._reserve_budget(output, "phase-c", Decimal("0.01"))

            state = json.loads(
                (output / benchmark.BUDGET_STATE_NAME).read_text(encoding="utf-8")
            )

        self.assertEqual(state["reservations"], {"phase-a": "20", "phase-b": "4"})

    def test_legacy_remote_ledger_cannot_bypass_budget_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "old_results.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "without a durable budget"):
                benchmark._reserve_budget(output, "phase", Decimal("1"))

    def test_snapshot_parsing_and_capability_tiers(self) -> None:
        endpoints = providers.parse_endpoint_snapshot(_snapshot())
        by_provider = {endpoint.provider: endpoint for endpoint in endpoints}

        self.assertEqual(
            by_provider["cloudflare"].input_per_million_usd, Decimal("0.14")
        )
        self.assertEqual(by_provider["cloudflare"].uptime_percent, 99.99)
        self.assertEqual(
            by_provider["cloudflare"].transports,
            frozenset({"strict_logprob", "strict_hard_verdict", "relaxed_json"}),
        )
        self.assertEqual(
            by_provider["coreweave"].transports, frozenset({"forced_tool"})
        )
        self.assertEqual(
            by_provider["digitalocean"].transports, frozenset({"relaxed_json"})
        )
        tiers = providers.capability_tiers(endpoints)
        self.assertEqual(
            [row.provider for row in tiers["strict_logprob"]], ["cloudflare"]
        )

        duplicate = _snapshot()
        duplicate["data"]["endpoints"].append(
            copy.deepcopy(duplicate["data"]["endpoints"][0])
        )
        with self.assertRaisesRegex(ValueError, "unique endpoint tags"):
            providers.parse_endpoint_snapshot(duplicate)

    def test_request_transports_preserve_the_frozen_contract(self) -> None:
        strict = providers.build_request(
            "strict_logprob",
            provider="cloudflare",
            text="classify me",
            input_channel="untrusted_content",
        )
        hard = providers.build_request(
            "strict_hard_verdict",
            provider="deepinfra",
            text="classify me",
            input_channel="untrusted_content",
        )
        tool = providers.build_request(
            "forced_tool",
            provider="coreweave",
            text="classify me",
            input_channel="untrusted_content",
        )
        relaxed = providers.build_request(
            "relaxed_json",
            provider="digitalocean",
            text="classify me",
            input_channel="untrusted_content",
        )
        reasoning = providers.build_request(
            "strict_hard_verdict",
            provider="cloudflare",
            text="classify me",
            input_channel="untrusted_content",
            system_prompt="custom policy",
            reasoning_effort="high",
        )

        self.assertEqual(strict["messages"], hard["messages"])
        self.assertEqual(strict["response_format"]["type"], "json_schema")
        self.assertTrue(strict["logprobs"])
        self.assertNotIn("logprobs", hard)
        self.assertEqual(tool["tool_choice"]["function"]["name"], providers._TOOL_NAME)
        self.assertNotIn("response_format", tool)
        self.assertEqual(relaxed["response_format"], {"type": "json_object"})
        self.assertEqual(reasoning["messages"][0]["content"], "custom policy")
        self.assertEqual(reasoning["reasoning"]["effort"], "high")
        self.assertEqual(reasoning["max_tokens"], 1_024)
        self.assertEqual(
            strict["provider"],
            {
                "order": ["cloudflare"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )

    def test_exact_parsing_identity_hook_and_safe_record(self) -> None:
        record = providers.parse_result(
            _payload(),
            row_id="row-1",
            transport="strict_logprob",
            requested_provider="cloudflare",
            client_seconds=0.25,
        )

        self.assertEqual(record.verdict, 1)
        self.assertAlmostEqual(record.probability, 0.8807970779778823)
        self.assertEqual(record.cost_usd, Decimal("0.00001"))
        self.assertFalse(hasattr(record, "raw_response"))
        with self.assertRaises(FrozenInstanceError):
            record.verdict = 0

        wrong = _payload()
        wrong["provider"] = "fireworks"
        with self.assertRaisesRegex(ValueError, "pinned provider"):
            providers.parse_result(
                wrong,
                row_id="row-1",
                transport="strict_logprob",
                requested_provider="cloudflare",
            )

        called = []
        providers.parse_result(
            wrong,
            row_id="row-1",
            transport="strict_logprob",
            requested_provider="cloudflare",
            identity_validator=lambda *identity: called.append(identity),
        )
        self.assertEqual(called[0][2], "fireworks")

        invalid = _payload('{"subversion":true}')
        with self.assertRaisesRegex(ValueError, "frozen integer schema"):
            providers.parse_result(
                invalid,
                row_id="row-1",
                transport="strict_hard_verdict",
                requested_provider="cloudflare",
            )

    def test_tool_parsing_budget_and_expansion_gates(self) -> None:
        payload = _payload()
        payload["provider"] = "CoreWeave"
        payload["choices"][0]["finish_reason"] = "tool_calls"
        payload["choices"][0]["message"] = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": providers._TOOL_NAME,
                        "arguments": '{"subversion":0}',
                    },
                }
            ]
        }
        records = tuple(
            providers.parse_result(
                payload,
                row_id=f"row-{index}",
                transport="forced_tool",
                requested_provider="coreweave",
            )
            for index in range(16)
        )
        self.assertTrue(providers.can_expand_canary(records))
        self.assertFalse(
            providers.can_expand_canary(
                records[:-1]
                + (
                    providers.failed_result(
                        row_id="row-15",
                        transport="forced_tool",
                        requested_provider="coreweave",
                        failure_code="timeout",
                    ),
                )
            )
        )

        ledger = providers.BudgetLedger()
        self.assertEqual(ledger.remaining_usd, Decimal("24"))
        self.assertTrue(ledger.allows("24"))
        self.assertFalse(ledger.allows("24.01"))
        ceiling = providers.request_cost_ceiling(
            providers.parse_endpoint_snapshot(_snapshot())[0], input_bytes=1_000
        )
        self.assertGreater(ceiling, Decimal("0.0007"))

        payload["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "finish reason"):
            providers.parse_result(
                payload,
                row_id="row-length",
                transport="forced_tool",
                requested_provider="coreweave",
            )

    def test_concurrency_eight_requires_reliability_and_more_throughput(self) -> None:
        four = providers.ConcurrencyObservation(4, 1000, 5, 12.0)
        eight = providers.ConcurrencyObservation(8, 1000, 5, 13.0)
        self.assertTrue(providers.may_probe_concurrency_eight(four))
        self.assertTrue(providers.accept_concurrency_eight(four, eight))
        self.assertFalse(
            providers.accept_concurrency_eight(
                four,
                providers.ConcurrencyObservation(8, 1000, 5, 11.0),
            )
        )
        self.assertFalse(
            providers.may_probe_concurrency_eight(
                providers.ConcurrencyObservation(4, 1000, 6, 20.0)
            )
        )

    def test_shared_provider_experiment_runner_writes_and_resumes(self) -> None:
        endpoint = providers.parse_endpoint_snapshot(_snapshot())[0]
        jobs = [
            {
                "job_id": f"job-{index}",
                "row": {
                    "panel_id": f"row-{index}",
                    "text_sha256": str(index) * 64,
                    "input_channel": "direct_user",
                },
                "text": f"text {index}",
                "prompt": "classify",
            }
            for index in range(2)
        ]

        async def run_once(output: Path):
            return await benchmark._run_provider_experiment(
                output=output,
                jobs=jobs,
                manifest={"schema_version": 1},
                manifest_path=output / "manifest.json",
                result_path=output / "results.jsonl",
                endpoint=endpoint,
                transport="strict_hard_verdict",
                stage="test",
                concurrency=1,
                make_record=lambda job, response: {
                    **response,
                    "job_id": job["job_id"],
                },
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            call = AsyncMock(return_value={"status": "ok"})
            with (
                patch.object(benchmark, "_api_key", return_value="secret"),
                patch.object(benchmark, "_call_provider", call),
            ):
                records, pending, _ = asyncio.run(run_once(output))
                resumed, resumed_pending, _ = asyncio.run(run_once(output))

        self.assertEqual(pending, 2)
        self.assertEqual(resumed_pending, 0)
        self.assertEqual(resumed, records)
        self.assertEqual(call.await_count, 2)

    def test_provider_load_cells_are_disjoint_and_summary_is_bounded(self) -> None:
        endpoints = providers.parse_endpoint_snapshot(_snapshot())
        winners = [
            (endpoints[0], next(iter(endpoints[0].transports))),
            (endpoints[1], next(iter(endpoints[1].transports))),
        ]
        lengths = [100 + index for index in range(70)]
        lengths += [1024 + index for index in range(20)]
        lengths += [4096 + index for index in range(10)]
        lengths += [16000, 96434]
        rows = [
            {
                "panel_id": f"row-{index}",
                "dataset": "canonical",
                "label": index % 2,
                "input_channel": "direct_user",
                "text_chars": length,
            }
            for index, length in enumerate(lengths)
        ]
        cells = benchmark._provider_load_cells(
            rows,
            winners,
            requests_per_cell=16,
        )
        ids = [row["panel_id"] for values in cells.values() for row in values]
        self.assertEqual(len(cells), 6)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            sum(
                benchmark._provider_length_band(row["text_chars"]) == ">=16000"
                for values in cells.values()
                for row in values
            ),
            2,
        )
        self.assertLessEqual(
            max(
                sum(
                    benchmark._provider_length_band(row["text_chars"]) == ">=16000"
                    for row in values
                )
                for values in cells.values()
            ),
            1,
        )
        self.assertTrue(all(len(values) == 16 for values in cells.values()))
        for band in benchmark.PROVIDER_LENGTH_BANDS:
            counts = [
                sum(
                    benchmark._provider_length_band(row["text_chars"]) == band
                    for row in values
                )
                for values in cells.values()
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)
        repeated = benchmark._provider_load_cells(
            rows,
            winners,
            requests_per_cell=16,
        )
        self.assertEqual(
            {key: [row["panel_id"] for row in values] for key, values in cells.items()},
            {
                key: [row["panel_id"] for row in values]
                for key, values in repeated.items()
            },
        )

        summary, observation = benchmark._provider_load_cell_summary(
            [
                {
                    "status": "ok",
                    "client_seconds": 0.5,
                    "prompt_tokens": 100,
                    "attempts": 1,
                    "cost_usd": "0.001",
                    "input_length_band": "<1024",
                },
                {
                    "status": "failed",
                    "client_seconds": 1.0,
                    "prompt_tokens": None,
                    "attempts": 2,
                    "cost_usd": None,
                    "input_length_band": ">=16000",
                },
            ],
            concurrency=4,
            wall_seconds=2.0,
        )
        self.assertEqual(summary["requests_per_second"], 1.0)
        self.assertEqual(summary["input_tokens_per_second"], 50.0)
        self.assertEqual(summary["cost_usd"], "0.001")
        self.assertEqual(summary["length_bands"], {"<1024": 1, ">=16000": 1})
        self.assertEqual(observation.terminal_failure_rate, 0.5)

    def test_provider_load_real_inventory_is_balanced_across_six_cells(self) -> None:
        inventory = {
            "<1024": (11412, 100),
            "1024-4095": (855, 2000),
            "4096-15999": (83, 8000),
            ">=16000": (2, 20000),
        }
        targets = benchmark._balanced_band_targets(
            {name: count for name, (count, _) in inventory.items()}, 6 * 32
        )

        self.assertEqual(
            targets,
            {
                "<1024": 64,
                "1024-4095": 63,
                "4096-15999": 63,
                ">=16000": 2,
            },
        )
        endpoints = providers.parse_endpoint_snapshot(_snapshot())
        winners = [
            (endpoints[0], next(iter(endpoints[0].transports))),
            (endpoints[1], next(iter(endpoints[1].transports))),
        ]
        rows = [
            {
                "panel_id": f"{band}-{index}",
                "text_chars": length,
            }
            for band, (count, length) in inventory.items()
            for index in range(count)
        ]
        cells = benchmark._provider_load_cells(rows, winners, requests_per_cell=32)
        self.assertTrue(all(len(values) == 32 for values in cells.values()))
        for band in benchmark.PROVIDER_LENGTH_BANDS:
            counts = [
                sum(
                    benchmark._provider_length_band(row["text_chars"]) == band
                    for row in values
                )
                for values in cells.values()
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_evaluation_uses_authoritative_hard_verdict_winner(self) -> None:
        keys = benchmark._evaluation_winner_keys(
            {
                "winners": {
                    "logprob": {
                        "provider": "cloudflare",
                        "transport": "strict_logprob",
                    },
                    "hard_verdict": None,
                }
            },
            {
                "provider": {
                    "name": "decart",
                    "transport": "strict_hard_verdict",
                }
            },
        )

        self.assertEqual(
            keys,
            {
                ("cloudflare", "strict_logprob"),
                ("decart", "strict_hard_verdict"),
            },
        )

    def test_evaluation_does_not_substitute_an_ineligible_hard_provider(self) -> None:
        keys = benchmark._evaluation_winner_keys(
            {
                "winners": {
                    "logprob": {
                        "provider": "cloudflare",
                        "transport": "strict_logprob",
                    }
                }
            },
            {"selection_status": "no_eligible_provider", "provider": None},
        )

        self.assertEqual(keys, {("cloudflare", "strict_logprob")})

    def test_generation_dns_and_timeout_fail_closed(self) -> None:
        class BrokenSession:
            def __init__(self, error: Exception) -> None:
                self.error = error

            def get(self, *_args, **_kwargs):
                raise self.error

        for error, expected in (
            (aiohttp.ClientConnectionError("dns"), "connection_error"),
            (TimeoutError(), "timeout"),
        ):
            metadata, failure = asyncio.run(
                benchmark._lookup_generation(BrokenSession(error), "token", "gen-1")
            )
            self.assertEqual(metadata, {})
            self.assertEqual(failure, expected)

    def test_missing_generation_metadata_defers_to_completion_identity(self) -> None:
        class MissingResponse:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def read(self):
                return b""

        class MissingSession:
            def get(self, *_args, **_kwargs):
                return MissingResponse()

        with patch("experiments.pipeline_benchmark.run.asyncio.sleep", new=AsyncMock()):
            metadata, failure = asyncio.run(
                benchmark._lookup_generation(MissingSession(), "token", "gen-1")
            )
        self.assertEqual(metadata, {})
        self.assertIsNone(failure)

    def test_complete_completion_metadata_skips_generation_lookup(self) -> None:
        payload = _payload()
        self.assertTrue(benchmark._completion_metadata_is_complete(payload))

        for key in ("provider", "model"):
            incomplete = copy.deepcopy(payload)
            incomplete.pop(key)
            self.assertFalse(benchmark._completion_metadata_is_complete(incomplete))
        for key in ("prompt_tokens", "completion_tokens", "cost"):
            incomplete = copy.deepcopy(payload)
            incomplete["usage"].pop(key)
            self.assertFalse(benchmark._completion_metadata_is_complete(incomplete))
        incomplete = copy.deepcopy(payload)
        incomplete["usage"]["prompt_tokens_details"].pop("cached_tokens")
        self.assertFalse(benchmark._completion_metadata_is_complete(incomplete))

    def test_provider_stage_ledgers_are_disjoint_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for stage in benchmark.PROVIDER_STAGES:
                benchmark._provider_result_path(output, stage).write_text(
                    json.dumps({"stage": stage, "cost_usd": "0.01"}) + "\n",
                    encoding="utf-8",
                )
            self.assertEqual(
                len(benchmark._all_provider_records(output)),
                len(benchmark.PROVIDER_STAGES),
            )
            self.assertEqual(benchmark._recorded_spend(output), Decimal("0.05"))

            benchmark._provider_result_path(output, "panel").write_text(
                json.dumps({"stage": "canary"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "wrong stage ledger"):
                benchmark._all_provider_records(output)


if __name__ == "__main__":
    unittest.main()
