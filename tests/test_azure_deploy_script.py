from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
        evidence = "reports/retrieval-lineage-hybrid-parity-20260820.json"
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

    def test_validation_requires_one_exact_routed_probe_and_memory_headroom(
        self,
    ) -> None:
        required = (
            (
                self.azure_app,
                '"a33b3e9299fd7d1c590413c2a8551fc4f6829c37bdb4c0ccfb6307c9fe668806"',
            ),
            (
                self.azure_app,
                '"843b52b4873b24f23417135e8e2244895cbe64b8c9eb84eee28570103f952e1d"',
            ),
            (self.azure_app, '"text": ROUTED_PROBE_TEXT'),
            (self.azure_app, "timeout=90"),
            (
                self.script,
                '(result.route, result.reason) != ("review", "deepseek_required")',
            ),
            (self.script, ".routed_probe.artifact_sha256 == $probe"),
            (self.script, '.routed_probe.advisory_route == "pass"'),
            (self.script, '.routed_probe.reason == "deepseek_clear"'),
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

        routed_probe = self.azure_app.index("_, routed_probe = request")
        local_pass = self.azure_app.index("for _ in range(30)")
        self.assertLess(routed_probe, local_pass)
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
