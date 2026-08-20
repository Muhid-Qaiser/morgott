"""Protected Azure preview API and local deployment smoke."""

import json
import os
import secrets
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .probe_identity import ROUTED_PROBE_TEXT

if TYPE_CHECKING:
    from .models.cascade import CascadeScanner

MAX_TEXT_BYTES = 64 * 1024


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class AzureSettings:
    api_key: str
    model_manifest: Path
    inference_precision: Literal["auto", "bf16", "fp32"] = "auto"

    @classmethod
    def from_env(cls) -> "AzureSettings":
        api_key = _required_env("MORGOTT_API_KEY")
        inference_precision = os.environ.get("MORGOTT_INFERENCE_PRECISION", "auto")
        if len(api_key) < 32:
            raise RuntimeError("MORGOTT_API_KEY must contain at least 32 characters")
        if inference_precision not in {"auto", "bf16", "fp32"}:
            raise RuntimeError(
                "MORGOTT_INFERENCE_PRECISION must be auto, bf16, or fp32"
            )
        return cls(
            api_key=api_key,
            model_manifest=Path(
                os.environ.get("MORGOTT_MODEL_MANIFEST", "model-artifacts.json")
            ),
            inference_precision=inference_precision,
        )


def create_app(
    *,
    settings: AzureSettings | None = None,
    scanner: "CascadeScanner | None" = None,
):
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, field_validator

    from .models.cascade import CascadeScanner
    from .models.downstream import PIPELINE_PROFILE, THRESHOLD_SHA256
    from .normalization import strict_normalize

    class AssessRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        text: str
        input_channel: Literal["direct_user", "untrusted_content"]

        @field_validator("text")
        @classmethod
        def validate_text(cls, value: str) -> str:
            try:
                size = len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("text must be valid UTF-8") from error
            if not value or size > MAX_TEXT_BYTES or not strict_normalize(value):
                raise ValueError("text must contain between 1 byte and 64 KiB")
            return value

    settings = settings or AzureSettings.from_env()
    scanner = scanner or CascadeScanner.from_artifacts(
        manifest_path=settings.model_manifest,
        inference_precision=settings.inference_precision,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            yield
        finally:
            await scanner.aclose()

    app = FastAPI(
        title="Morgott advisory preview",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        del request, error
        return JSONResponse({"detail": "Invalid request"}, status_code=422)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied.encode(), settings.api_key.encode()):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ready"}

    @app.get("/v1/status")
    async def status():
        identity = scanner.runtime_identity
        return {
            "ready": True,
            "pipeline_profile": PIPELINE_PROFILE,
            "policy_sha256": scanner.policy_sha256,
            "retrieval_enabled": scanner.retrieval_enabled,
            "retrieval_manifest_sha256": scanner.retrieval_manifest_sha256,
            "threshold_sha256": THRESHOLD_SHA256,
            "model_key": identity.model_key,
            "onnx_sha256": identity.onnx_sha256,
            "context_length": identity.max_tokens,
            "window_overlap": identity.window_overlap,
            "runtime": identity.runtime,
            "openvino": identity.openvino,
            "requested_precision": identity.requested_inference_precision,
            "precision": identity.inference_precision,
            "reported_inference_precision": identity.reported_inference_precision,
            "compile_seconds": identity.compile_seconds,
            "cpu_capabilities": list(identity.cpu_capabilities),
        }

    @app.post("/v1/assess")
    async def assess(request: AssessRequest):
        assessment = await scanner.assess_text(
            request.text,
            input_channel=request.input_channel,
        )
        if assessment.decision != "allow":
            raise RuntimeError("advisory-only invariant failed")
        return asdict(assessment)

    return app


def smoke_local() -> dict:
    key = _required_env("MORGOTT_API_KEY")
    requested_precision = os.environ.get("MORGOTT_INFERENCE_PRECISION", "auto")
    base = "http://127.0.0.1:8000"

    def request(
        path: str,
        *,
        body: dict | None = None,
        authenticated: bool = True,
        timeout: int = 30,
    ):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {key}"
        raw = None if body is None else json.dumps(body).encode()
        with urllib.request.urlopen(
            urllib.request.Request(base + path, data=raw, headers=headers),
            timeout=timeout,
        ) as response:
            return response.status, json.load(response)

    try:
        request("/v1/status", authenticated=False)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
    else:
        raise ValueError("protected status endpoint accepted no credential")

    status_code, status = request("/v1/status")
    if (
        status_code != 200
        or status.get("ready") is not True
        or status.get("requested_precision") != requested_precision
        or status.get("precision")
        not in (
            {"bf16", "fp32"} if requested_precision == "auto" else {requested_precision}
        )
        or status.get("context_length") != 1024
    ):
        raise ValueError("preview status contract failed")

    try:
        request(
            "/v1/assess",
            body={"text": "x" * (MAX_TEXT_BYTES + 1), "input_channel": "direct_user"},
        )
    except urllib.error.HTTPError as error:
        if error.code != 422:
            raise
    else:
        raise ValueError("oversized API text was accepted")

    _, routed_probe = request(
        "/v1/assess",
        body={"text": ROUTED_PROBE_TEXT, "input_channel": "untrusted_content"},
        timeout=90,
    )

    for _ in range(30):
        _, assessment = request(
            "/v1/assess",
            body={
                "text": "ordinary account support question",
                "input_channel": "direct_user",
            },
        )
        if (
            assessment.get("decision") != "allow"
            or assessment.get("advisory_route") != "pass"
        ):
            raise ValueError("local-pass smoke failed")
    peak_rss_kib = None
    status_path = Path("/proc/1/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                peak_rss_kib = int(line.split()[1])
                break

    def cgroup_bytes(name: str) -> int | None:
        path = Path("/sys/fs/cgroup") / name
        if not path.is_file():
            return None
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value.isdigit() else None

    return {
        "cgroup_memory_limit_bytes": cgroup_bytes("memory.max"),
        "cgroup_memory_peak_bytes": cgroup_bytes("memory.peak"),
        "peak_rss_kib": peak_rss_kib,
        "routed_probe": {
            key: routed_probe.get(key)
            for key in (
                "advisory_route",
                "artifact_sha256",
                "complete",
                "decision",
                "deepseek_calls",
                "deepseek_failures",
                "embedding_request_sha256",
                "high_windows",
                "low_windows",
                "max_mmbert_score",
                "middle_windows",
                "prompt_sha256",
                "provider",
                "provider_request_sha256",
                "reason",
                "retrieval_packet_sha256",
                "retrieval_status",
                "selected_example_count",
            )
        },
        "status": status,
    }


def main() -> None:
    print(json.dumps(smoke_local(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
