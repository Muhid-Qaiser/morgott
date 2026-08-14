import hashlib
import json
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from morgott.azure_app import (
    AzureSettings,
    _consume_canaries,
    _ConsumerHealth,
    _nearest_rank_p95,
    _run_canaries,
    _validate_command,
    create_app,
)


@dataclass(frozen=True)
class _Assessment:
    decision: str = "allow"
    advisory_only: bool = True
    advisory_route: str = "pass"
    artifact_sha256: str = "a" * 64
    complete: bool = True
    deepseek_calls: int = 1
    total_latency_ms: float = 1.0


class _Scanner:
    runtime_identity = SimpleNamespace(
        model_key="mmbert-lora-full-ctx1024-u17000-s42",
        onnx_sha256="b" * 64,
        max_tokens=1024,
        window_overlap=128,
        runtime="openvino-test-cpu-fp32",
        openvino="test",
        requested_inference_precision="auto",
        inference_precision="fp32",
        reported_inference_precision="f32",
        compile_seconds=1.25,
        cpu_capabilities=("BF16", "FP32"),
    )

    def __init__(self):
        self.closed = False

    async def assess_text(self, text, *, input_channel):
        del text, input_channel
        return _Assessment()

    async def aclose(self):
        self.closed = True


class AzureAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scanner = _Scanner()
        self.settings = AzureSettings(
            api_key="company-preview-key-with-at-least-32-characters",
            servicebus_fqdn="example.servicebus.windows.net",
            queue_name="daily-canary",
            storage_account_url="https://example.blob.core.windows.net",
            storage_container="morgott",
            manifest_blob="data/manifest.json",
            manifest_sha256="c" * 64,
            model_manifest=Path("model-artifacts.json"),
        )

    async def test_api_auth_bounds_status_and_advisory_response(self):
        app = create_app(
            settings=self.settings,
            scanner=self.scanner,
            start_consumer=False,
        )
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                self.assertEqual(
                    (await client.get("/healthz")).json(),
                    {"status": "ready"},
                )
                self.assertEqual((await client.get("/v1/status")).status_code, 401)
                self.assertEqual(
                    (
                        await client.get(
                            "/v1/status",
                            headers=[(b"authorization", b"Bearer \xff")],
                        )
                    ).status_code,
                    401,
                )
                status = (await client.get("/v1/status", headers=headers)).json()
                self.assertEqual(status["context_length"], 1024)
                self.assertEqual(status["requested_precision"], "auto")
                self.assertEqual(status["precision"], "fp32")

                text = "ordinary private input"
                response = await client.post(
                    "/v1/assess",
                    headers=headers,
                    json={"text": text, "input_channel": "direct_user"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["decision"], "allow")
                self.assertNotIn(text, json.dumps(response.json()))

                too_large = "€" * ((64 * 1024 // 3) + 1)
                response = await client.post(
                    "/v1/assess",
                    headers=headers,
                    json={"text": too_large, "input_channel": "direct_user"},
                )
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("€", response.text)

                app.state.consumer_health.consecutive_failures = 3
                self.assertEqual((await client.get("/healthz")).status_code, 503)
                status = await client.get("/v1/status", headers=headers)
                self.assertEqual(status.status_code, 200)
                self.assertIs(status.json()["ready"], False)
        self.assertTrue(self.scanner.closed)

    async def test_consumer_health_recovers_after_a_successful_receive(self):
        health = _ConsumerHealth()
        sleeps = 0

        async def stop_after_three_failures(delay):
            nonlocal sleeps
            self.assertEqual(delay, 10)
            sleeps += 1
            self.assertIs(health.ready, sleeps < 3)
            if sleeps == 3:
                raise RuntimeError("stop test consumer")

        with (
            patch(
                "morgott.azure_app._credential",
                side_effect=ConnectionError("identity unavailable"),
            ),
            patch(
                "morgott.azure_app.asyncio.sleep", side_effect=stop_after_three_failures
            ),
            self.assertRaisesRegex(RuntimeError, "stop test consumer"),
        ):
            await _consume_canaries(self.scanner, self.settings, health)

        self.assertFalse(health.ready)
        health.received()
        self.assertTrue(health.ready)

    async def test_daily_canary_rejects_an_incomplete_provider_review(self):
        manifest = b"{}"
        settings = replace(
            self.settings,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )

        class AsyncContext:
            def __init__(self, value):
                self.value = value

            async def __aenter__(self):
                return self.value

            async def __aexit__(self, *args):
                return None

        class Blob:
            async def download_blob(self, **kwargs):
                self.download = kwargs
                return self

            async def readall(self):
                return manifest

        class BlobService:
            def __init__(self):
                self.blob = Blob()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def get_blob_client(self, *args):
                self.target = args
                return self.blob

        scanner = _Scanner()

        async def incomplete(*args, **kwargs):
            return _Assessment(complete=False)

        scanner.assess_text = incomplete
        with (
            patch("morgott.azure_app._credential", return_value=AsyncContext(object())),
            patch("morgott.azure_app.BlobServiceClient", return_value=BlobService()),
            self.assertRaisesRegex(ValueError, "canary assessment incomplete"),
        ):
            await _run_canaries(scanner, settings)

    def test_canary_commands_are_versioned_and_recent(self):
        from datetime import date

        command = {
            "schema_version": 2,
            "command": "daily_canary",
            "command_id": "deployment-123",
            "issued_for": date.today().isoformat(),
        }
        self.assertEqual(_validate_command(json.dumps(command)), command)
        command["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "invalid canary command"):
            _validate_command(json.dumps(command))

    def test_nearest_rank_p95_uses_the_twenty_ninth_of_thirty_samples(self):
        self.assertEqual(_nearest_rank_p95(list(range(1, 31))), 29)


if __name__ == "__main__":
    unittest.main()
