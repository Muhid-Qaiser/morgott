from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


OPEN_ONCE = urllib.request.build_opener(_NoRedirect()).open


def sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def build_request(
    body: dict, api_key: str, user_agent: str
) -> tuple[urllib.request.Request, str]:
    encoded = canonical(body)
    request = urllib.request.Request(
        ENDPOINT,
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Cache": "false",
            "X-OpenRouter-Metadata": "enabled",
            "User-Agent": user_agent,
        },
    )
    return request, sha256(encoded)


def request_once(
    request: urllib.request.Request,
    timeout: float,
    max_response_bytes: int,
    opener: Callable[..., object] = OPEN_ONCE,
) -> tuple[bytes | None, dict[str, object]]:
    """Make exactly one bounded HTTP attempt."""
    started = time.perf_counter()
    raw = None
    try:
        with opener(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            raw = response.read(max_response_bytes + 1)
        if len(raw) > max_response_bytes:
            result = {
                "unavailable_reason": "response_too_large",
                "response_sha256": sha256(raw),
            }
            raw = None
        else:
            result = {}
    except urllib.error.HTTPError as exc:
        exc.close()
        family = exc.code // 100 if isinstance(exc.code, int) else 0
        result = {
            "unavailable_reason": f"http_{family}xx"
            if family in {4, 5}
            else "http_error",
            "response_sha256": None,
        }
    except (TimeoutError, socket.timeout):
        result = {"unavailable_reason": "timeout", "response_sha256": None}
    except urllib.error.URLError:
        result = {"unavailable_reason": "network_error", "response_sha256": None}
    except Exception:
        result = {"unavailable_reason": "client_error", "response_sha256": None}
    result["latency_ms"] = round((time.perf_counter() - started) * 1_000, 3)
    return raw, result
