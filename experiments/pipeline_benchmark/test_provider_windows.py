from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

from experiments.pipeline_benchmark import provider_windows, providers


def _endpoint(provider: str, transport: providers.Transport) -> providers.Endpoint:
    parameters = {
        "strict_logprob": frozenset(
            {"response_format", "structured_outputs", "logprobs", "top_logprobs"}
        ),
        "strict_hard_verdict": frozenset({"response_format", "structured_outputs"}),
    }[transport]
    return providers.Endpoint(
        provider=provider,
        name=provider,
        tag=provider,
        model=providers.MODEL,
        quantization=None,
        uptime_percent=100,
        supported_parameters=parameters,
        input_per_million_usd=None,
        output_per_million_usd=None,
        cache_read_per_million_usd=None,
    )


class _Preprocessor:
    def prepare(self, text: str):
        return SimpleNamespace(
            normalized_text=text,
            token_count=42,
            windows=(
                SimpleNamespace(char_start=0, char_end=4),
                SimpleNamespace(char_start=3, char_end=8),
                SimpleNamespace(char_start=7, char_end=len(text)),
            ),
        )


class ProviderWindowTest(unittest.TestCase):
    def test_approved_thresholds_cover_the_complete_local_grid(self) -> None:
        thresholds = provider_windows._approved_thresholds()
        self.assertEqual(
            len(thresholds),
            len(provider_windows.metrics.DIRECT_LOW_GRID)
            * len(provider_windows.metrics.UNTRUSTED_LOW_GRID)
            * len(provider_windows.metrics.LOCAL_HIGH_GRID),
        )
        self.assertEqual(
            {value["direct_low"] for value in thresholds},
            set(provider_windows.metrics.DIRECT_LOW_GRID),
        )

    def test_targets_include_all_completed_strict_hard_providers_only(self) -> None:
        summary = {
            "winners": {
                "logprob": {
                    "provider": "cloudflare",
                    "transport": "strict_logprob",
                },
                "hard_verdict": None,
            },
            "providers": {
                "baidu": {
                    "provider": "baidu",
                    "transport": "strict_hard_verdict",
                    "rows": 1024,
                },
                "deepinfra": {
                    "provider": "deepinfra",
                    "transport": "strict_hard_verdict",
                    "rows": 1024,
                },
                "alternate": {
                    "provider": "digitalocean",
                    "transport": "relaxed_json",
                    "rows": 1024,
                },
                "partial": {
                    "provider": "decart",
                    "transport": "strict_hard_verdict",
                    "rows": 1000,
                },
            },
        }
        self.assertEqual(
            provider_windows._target_keys(summary, expected_panel_rows=1024),
            (
                ("cloudflare", "strict_logprob"),
                ("baidu", "strict_hard_verdict"),
                ("deepinfra", "strict_hard_verdict"),
            ),
        )

    def test_reachable_union_respects_artifact_high_short_circuit(self) -> None:
        profiles = (
            {"direct_low": 0.05, "untrusted_low": 0.025, "local_high": 0.99},
            {"direct_low": 0.2, "untrusted_low": 0.1, "local_high": 0.99999},
        )
        self.assertEqual(
            provider_windows._reachable_window_indices(
                [0.01, 0.1, 0.98], "direct_user", profiles
            ),
            {1, 2},
        )
        self.assertEqual(
            provider_windows._reachable_window_indices(
                [0.1, 0.999999], "untrusted_content", profiles
            ),
            set(),
        )

    def test_jobs_keep_only_safe_multiwindow_middle_zones_without_text_output(
        self,
    ) -> None:
        text = "abcdefghijk"
        digest = hashlib.sha256(text.encode()).hexdigest()
        panel = [
            {
                "panel_id": "safe",
                "input_channel": "untrusted_content",
                "text_sha256": digest,
            },
            {
                "panel_id": "unsafe",
                "input_channel": "direct_user",
                "text_sha256": digest,
            },
        ]
        scores = {
            name: {
                "window_count": 3,
                "token_count": 42,
                "window_scores": [0.01, 0.1, 0.98],
                "text_sha256": digest,
            }
            for name in ("safe", "unsafe")
        }
        winners = (
            (_endpoint("cloudflare", "strict_logprob"), "strict_logprob"),
            (
                _endpoint("deepinfra", "strict_hard_verdict"),
                "strict_hard_verdict",
            ),
        )
        jobs = provider_windows.build_jobs(
            panel,
            scores,
            {"safe": text, "unsafe": text},
            safe_ids={"safe"},
            thresholds=(
                {
                    "direct_low": 0.05,
                    "untrusted_low": 0.025,
                    "local_high": 0.99,
                },
            ),
            winners=winners,
            preprocessor=_Preprocessor(),
        )

        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.artifact_id for job in jobs}, {"safe"})
        self.assertEqual({job.artifact_text_sha256 for job in jobs}, {digest})
        self.assertEqual({job.window_index for job in jobs}, {1, 2})
        self.assertEqual(len({job.job_id for job in jobs}), 4)
        self.assertEqual(
            {job.text_sha256 for job in jobs},
            {
                hashlib.sha256(text[3:8].encode()).hexdigest(),
                hashlib.sha256(text[7:].encode()).hexdigest(),
            },
        )
        persisted_fields = {
            "artifact_id",
            "artifact_text_sha256",
            "window_index",
            "char_start",
            "char_end",
            "text_sha256",
            "local_score",
            "window_text_sha256",
            "window_local_score",
        }
        self.assertNotIn("text", persisted_fields)


if __name__ == "__main__":
    unittest.main()
