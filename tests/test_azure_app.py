from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx

from morgott.azure_app import AzureSettings, create_app
from morgott.models.downstream import THRESHOLD_SHA256


@dataclass(frozen=True)
class _Assessment:
    decision: str = "allow"
    advisory_only: bool = True
    advisory_route: str = "pass"
    artifact_sha256: str = "a" * 64
    complete: bool = True
    deepseek_calls: int = 1
    retrieval_status: str = "ok"
    selected_example_count: int = 4
    retrieval_packet_sha256: str = (
        "843b52b4873b24f23417135e8e2244895cbe64b8c9eb84eee28570103f952e1d"
    )
    total_latency_ms: float = 1.0


class _Scanner:
    policy_sha256 = "d" * 64
    retrieval_enabled = True
    retrieval_manifest_sha256 = "e" * 64
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
        del text
        return _Assessment(
            advisory_route=(
                "restrict" if input_channel == "untrusted_content" else "pass"
            )
        )

    async def aclose(self):
        self.closed = True


class AzureAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scanner = _Scanner()
        self.settings = AzureSettings(
            api_key="company-preview-key-with-at-least-32-characters",
            model_manifest=Path("model-artifacts.json"),
        )

    async def test_api_auth_bounds_status_and_advisory_response(self):
        app = create_app(settings=self.settings, scanner=self.scanner)
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
                self.assertEqual(
                    status["pipeline_profile"], "balanced-retrieval-20260819"
                )
                self.assertEqual(status["policy_sha256"], "d" * 64)
                self.assertTrue(status["retrieval_enabled"])
                self.assertEqual(status["retrieval_manifest_sha256"], "e" * 64)
                self.assertEqual(status["threshold_sha256"], THRESHOLD_SHA256)
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
        self.assertTrue(self.scanner.closed)


if __name__ == "__main__":
    unittest.main()
