from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from morgott import azure_app


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: dict, status: int = 200):
        super().__init__(json.dumps(payload).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class AzureSmokeBehaviorTests(unittest.TestCase):
    """The local smoke must issue its checks in the frozen order.

    These behavioral checks replace raw source-text assertions: they run
    smoke_local against a stubbed HTTP layer and assert the auth probe, the
    bounds probe, the exact routed probe, and the 30 local-pass requests in
    order, with the routed-probe identity and timeout intact.
    """

    def _run_smoke(self):
        calls = []

        status_payload = {
            "ready": True,
            "requested_precision": "auto",
            "precision": "fp32",
            "context_length": 1024,
        }
        probe_payload = {
            "advisory_route": "restrict",
            "reason": "deepseek_flag",
            "decision": "allow",
            "retrieval_status": "ok",
            "retrieval_packet_sha256": "f" * 64,
        }
        local_payload = {"decision": "allow", "advisory_route": "pass"}

        def fake_urlopen(request, timeout=None):
            index = len(calls)
            calls.append(
                {
                    "url": request.full_url,
                    "body": None if request.data is None else request.data,
                    "authorization": request.get_header("Authorization"),
                    "timeout": timeout,
                }
            )
            if index == 0:
                raise urllib.error.HTTPError(
                    request.full_url, 401, "unauthorized", None, None
                )
            if index == 1:
                return _FakeResponse(status_payload)
            if index == 2:
                raise urllib.error.HTTPError(
                    request.full_url, 422, "too large", None, None
                )
            if index == 3:
                return _FakeResponse(probe_payload)
            return _FakeResponse(local_payload)

        environment = {
            "MORGOTT_API_KEY": "company-preview-key-with-at-least-32-characters",
            "MORGOTT_INFERENCE_PRECISION": "auto",
        }
        with (
            patch.dict("os.environ", environment),
            patch("morgott.azure_app.urllib.request.urlopen", fake_urlopen),
        ):
            result = azure_app.smoke_local()
        return calls, result

    def test_the_smoke_checks_run_in_the_frozen_order(self):
        calls, result = self._run_smoke()

        self.assertEqual(len(calls), 34)
        self.assertTrue(calls[0]["url"].endswith("/v1/status"))
        self.assertIsNone(calls[0]["authorization"])
        self.assertTrue(calls[1]["url"].endswith("/v1/status"))
        self.assertIsNotNone(calls[1]["authorization"])
        oversized = json.loads(calls[2]["body"])
        self.assertGreater(len(oversized["text"]), azure_app.MAX_TEXT_BYTES)
        self.assertEqual(oversized["input_channel"], "direct_user")
        self.assertEqual(result["routed_probe"]["retrieval_packet_sha256"], "f" * 64)
        self.assertEqual(result["status"]["ready"], True)

    def test_the_routed_probe_precedes_every_local_pass_request(self):
        calls, _ = self._run_smoke()

        probe = json.loads(calls[3]["body"])
        self.assertEqual(
            probe,
            {
                "text": azure_app.ROUTED_PROBE_TEXT,
                "input_channel": "untrusted_content",
            },
        )
        self.assertEqual(calls[3]["timeout"], 90)
        local = calls[4:]
        self.assertEqual(len(local), 30)
        for call in local:
            body = json.loads(call["body"])
            self.assertEqual(body["input_channel"], "direct_user")
            self.assertNotEqual(body["text"], azure_app.ROUTED_PROBE_TEXT)


class AzureDeployScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = Path("scripts/deploy-azure.sh").read_text(encoding="utf-8")
        cls.azure_app = Path("src/morgott/azure_app.py").read_text(encoding="utf-8")
        cls.bicep = Path("infra/main.bicep").read_text(encoding="utf-8")
        cls.dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        cls.dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
        cls.pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    def test_rejects_unknown_candidate_size_before_azure_calls(self) -> None:
        result = subprocess.run(
            ["scripts/deploy-azure.sh", "--candidate-size", "8cpu-16gi"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("2cpu-4gi or 4cpu-8gi", result.stderr)

    def test_rejects_promotion_until_the_replacement_gate_exists(self) -> None:
        result = subprocess.run(
            ["scripts/deploy-azure.sh", "--promote"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("promotion is blocked pending", result.stderr.lower())

    def test_private_bundle_is_downloaded_verified_and_baked(self) -> None:
        policy = json.loads(
            Path(
                "artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/promotion-retrieval.json"
            ).read_text(encoding="utf-8")
        )
        evidence = policy["evidence"]["path"]
        required = (
            (self.script, '"morgott-lineage-hybrid-v1"'),
            (self.script, "az storage blob download"),
            (self.script, "--auth-mode login"),
            (self.script, "stat -c '%s'"),
            (self.script, 'sha256sum "$target"'),
            (self.script, "bundle_root / pure_path"),
            (self.script, '"$build_context"'),
            (self.script, 'policy_target="$build_context/$CASCADE_POLICY_PATH"'),
            (
                self.script,
                "from morgott.models.cascade import _verify_registered_policy",
            ),
            (self.script, evidence),
            (self.script, ".source.data_manifest_sha256"),
            (self.script, "retrieval bank was built from a different data manifest"),
            (self.dockerfile, evidence),
            (self.dockerignore, f"!{evidence}"),
            (self.script, "download_retrieval_file"),
            (self.script, 'cp --reflink=auto "$blob_path" "$target"'),
        )
        for document, expected in required:
            with self.subTest(expected=expected):
                self.assertIn(expected, document)

    def test_runtime_user_owns_the_model_registry(self) -> None:
        self.assertIn(
            "COPY --chown=morgott:morgott model-artifacts.json ./",
            self.dockerfile,
        )

    def test_acr_digest_lookup_retries_after_a_transient_data_plane_failure(
        self,
    ) -> None:
        lookup = self.script.index("for _ in {1..6}", self.script.index("image_digest"))
        failure = self.script.index("ACR returned an invalid image digest", lookup)

        self.assertIn("az acr repository show", self.script[lookup:failure])
        self.assertIn("sleep 5", self.script[lookup:failure])

    def test_acr_image_is_built_from_the_current_context(self) -> None:
        build = self.script.index("az acr build")
        digest = self.script.index('image_digest=""', build)

        self.assertNotIn("az acr repository show-tags", self.script)
        self.assertIn('"$build_context"', self.script[build:digest])
        self.assertIn(
            'image_tag="lineage-hybrid-${image_fingerprint:0:16}"', self.script
        )

    def test_revision_poll_retries_transient_azure_read_failures(self) -> None:
        poll = self.script.index("for _ in {1..36}")
        failure = self.script.index(
            "Container App revision did not reach Running", poll
        )
        body = self.script[poll:failure]

        self.assertIn("if ! revision_name=$(az containerapp show", body)
        self.assertIn('revision_name == "$previous_revision"', body)
        self.assertIn("az containerapp revision activate", body)
        self.assertIn("if ! revision_details=$(az containerapp revision show", body)
        self.assertIn('jq -r .active <<<"$revision_details"', body)
        self.assertEqual(body.count("continue"), 5)

    def test_revision_smoke_parser_accepts_the_nested_result(self) -> None:
        function = self.script.index("revision_smoke()")
        start = self.script.index("import json", function)
        end = self.script.index("\nPY\n}", start)
        parser = self.script[start:end]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "smoke.raw"
            raw.write_text(
                'terminal noise\r\n{"routed_probe":{"status":"ok"},"status":{"ready":true}}\r\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-", str(raw)],
                input=parser,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"routed_probe"', result.stdout)

    def test_clean_tree_is_required_before_azure_calls(self) -> None:
        clean_gate = self.script.index("git status --porcelain")
        azure_account = self.script.index("az account show")

        self.assertLess(clean_gate, azure_account)
        self.assertIn("A clean Git worktree is required", self.script)

    def test_zero_traffic_validation_only_syncs_consumed_secrets(self) -> None:
        self.assertIn("ensure_secret_matches()", self.script)
        self.assertIn('if [[ $existing != "$value" ]]', self.script)
        self.assertIn("rotate it separately", self.script)
        self.assertIn(
            'ensure_secret_matches openrouter-api-key "$openrouter_key"', self.script
        )
        for unused in ("hf-token", "morgott-sas-url", "openai-api-key"):
            with self.subTest(unused=unused):
                self.assertNotIn(unused, self.script)

    def test_candidate_is_always_retained_at_zero_traffic(self) -> None:
        gate = self.script.index('<<<"$candidate_smoke" >/dev/null')
        retention = self.script.index("candidate_retained=true", gate)

        self.assertGreater(retention, gate)
        self.assertIn("Validated candidate retained at zero traffic", self.script)
        self.assertNotIn("Publishing 100 percent", self.script)
        self.assertNotIn('revision_smoke "$previous_revision"', self.script)
        self.assertIn("Candidate is not a distinct zero-traffic revision", self.script)
        self.assertIn(".revisionName == $candidate and .weight > 0", self.script)

    def test_candidate_resources_have_only_the_two_supported_shapes(self) -> None:
        self.assertIn("param candidateSize string = '2cpu-4gi'", self.bicep)
        self.assertIn("candidateSize == '4cpu-8gi' ? 4 : 2", self.bicep)
        self.assertIn("candidateSize == '4cpu-8gi' ? '8Gi' : '4Gi'", self.bicep)
        self.assertIn("maxReplicas: 1\n        minReplicas: 1", self.bicep)
        self.assertNotIn("concurrentRequests", self.bicep)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"name: '{name}'\n              value: '1'", self.bicep)

    def test_zero_traffic_validation_does_not_manage_canary_infrastructure(
        self,
    ) -> None:
        self.assertNotIn("morgott-daily-canary", self.bicep)
        self.assertNotIn("direct-canary", self.bicep)
        self.assertNotIn("scheduledQueryRules", self.bicep)
        self.assertNotIn("az containerapp job", self.script)
        self.assertNotIn("azure-servicebus", self.pyproject)

    def test_probe_identity_is_the_single_consistent_source(self) -> None:
        import hashlib

        from morgott.probe_identity import (
            ROUTED_PROBE_PACKET_SHA256,
            ROUTED_PROBE_SCORE_RANGE,
            ROUTED_PROBE_SHA256,
            ROUTED_PROBE_TEXT,
        )

        self.assertEqual(
            hashlib.sha256(ROUTED_PROBE_TEXT.encode()).hexdigest(),
            ROUTED_PROBE_SHA256,
        )
        self.assertRegex(ROUTED_PROBE_PACKET_SHA256, r"^[0-9a-f]{64}$")
        low, high = ROUTED_PROBE_SCORE_RANGE
        self.assertLess(0.0, low)
        self.assertLess(low, high)
        self.assertLess(high, 1.0)
        # Both consumers read the one canonical module, so a promotion
        # updates src/morgott/probe_identity.py and nothing else.
        self.assertIn("from morgott.probe_identity import", self.script)
        self.assertNotIn("ROUTED_PROBE_TEXT = ", self.azure_app)
        self.assertIn("from .probe_identity import ROUTED_PROBE_TEXT", self.azure_app)

    def test_validation_requires_one_exact_routed_probe_and_memory_headroom(
        self,
    ) -> None:
        required = (
            (
                self.script,
                '(result.route, result.reason) != ("review", "deepseek_required")',
            ),
            (self.script, ".routed_probe.artifact_sha256 == $probe"),
            (self.script, '.routed_probe.advisory_route == "restrict"'),
            (self.script, '.routed_probe.reason == "deepseek_flag"'),
            (self.script, ".routed_probe.deepseek_calls >= 1"),
            (self.script, '.routed_probe.retrieval_status == "ok"'),
            (self.script, ".routed_probe.retrieval_packet_sha256 == $packet"),
            (
                self.script,
                ".routed_probe.embedding_request_sha256 == $embedding_request",
            ),
            (self.script, ".routed_probe.prompt_sha256 == $prompt"),
            (self.script, ".routed_probe.provider == $provider"),
            (self.script, ".routed_probe.provider_request_sha256 == $provider_request"),
            (self.script, "($limit - $peak) >= 536870912"),
        )
        for document, expected in required:
            with self.subTest(expected=expected):
                self.assertIn(expected, document)

        # Probe-before-local-pass ordering, the probe payload, and its 90s
        # timeout are asserted behaviorally in AzureSmokeBehaviorTests.
        for removed in (
            "ABBA",
            "CANARY_PAIRS",
            "STABLE_PROFILE",
            "azure-canary-report",
            "same-input randomized",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.script)


if __name__ == "__main__":
    unittest.main()
