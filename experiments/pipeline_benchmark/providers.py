"""Pure provider-contract helpers for the pipeline benchmark.

This module deliberately performs no I/O. Callers fetch endpoint metadata and
provider responses, then pass those values through the fail-closed functions
below. Raw provider responses must not be written to benchmark artifacts.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Literal

from morgott.models.deepseek_nooa import (
    MODEL,
    PROMPT,
    _parse_logprobs,
    _parse_verdict,
)

Transport = Literal[
    "strict_logprob",
    "strict_hard_verdict",
    "forced_tool",
    "relaxed_json",
]
FailureCode = Literal[
    "http_error",
    "timeout",
    "connection_error",
    "invalid_response",
    "identity_mismatch",
]
IdentityValidator = Callable[[str, str, str, str], None]

MAX_COST_USD = Decimal("25")
RESERVE_USD = Decimal("1")
USABLE_COST_USD = MAX_COST_USD - RESERVE_USD
MIN_PROVIDER_UPTIME_PERCENT = 95.0
MAX_TERMINAL_FAILURE_RATE = 0.005
MAX_ATTEMPTS = 3
REQUEST_OVERHEAD_TOKEN_CEILING = 4_096

_SCHEMA = {
    "type": "object",
    "properties": {"subversion": {"type": "integer", "enum": [0, 1]}},
    "required": ["subversion"],
    "additionalProperties": False,
}
_STRICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "instruction_subversion",
        "strict": True,
        "schema": _SCHEMA,
    },
}
_TOOL_NAME = "classify_instruction_subversion"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Validated subset of one frozen OpenRouter endpoint record."""

    provider: str
    name: str
    tag: str
    model: str
    quantization: str | None
    uptime_percent: float
    supported_parameters: frozenset[str]
    input_per_million_usd: Decimal | None
    output_per_million_usd: Decimal | None
    cache_read_per_million_usd: Decimal | None

    @property
    def transports(self) -> frozenset[Transport]:
        parameters = self.supported_parameters
        result: set[Transport] = set()
        strict = {"response_format", "structured_outputs"} <= parameters
        if strict:
            result.add("strict_hard_verdict")
            if {"logprobs", "top_logprobs"} <= parameters:
                result.add("strict_logprob")
        if {"tools", "tool_choice"} <= parameters:
            result.add("forced_tool")
        if "response_format" in parameters:
            result.add("relaxed_json")
        return frozenset(result)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Immutable parsed-only record safe to persist as benchmark evidence."""

    row_id: str
    transport: Transport
    requested_provider: str
    requested_model: str
    returned_provider: str | None
    returned_model: str | None
    status: Literal["ok", "failed"]
    verdict: int | None
    probability: float | None
    log_odds: float | None
    attempts: int
    client_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: Decimal | None = None
    failure_code: FailureCode | None = None

    def __post_init__(self) -> None:
        if (
            not self.row_id
            or not self.requested_provider
            or not self.requested_model
            or self.attempts < 1
            or not math.isfinite(self.client_seconds)
            or self.client_seconds < 0
        ):
            raise ValueError("result metadata is invalid")
        if self.status == "failed":
            if self.failure_code is None or any(
                value is not None
                for value in (
                    self.returned_provider,
                    self.returned_model,
                    self.verdict,
                    self.probability,
                    self.log_odds,
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.cached_tokens,
                    self.cost_usd,
                )
            ):
                raise ValueError("failed records may contain only bounded metadata")
            return
        if (
            self.status != "ok"
            or self.failure_code is not None
            or not self.returned_provider
            or not self.returned_model
            or type(self.verdict) is not int
            or self.verdict not in (0, 1)
        ):
            raise ValueError("successful result is incomplete")
        if self.transport == "strict_logprob":
            if (
                not isinstance(self.probability, float)
                or not 0 <= self.probability <= 1
                or not isinstance(self.log_odds, float)
                or not math.isfinite(self.log_odds)
            ):
                raise ValueError("logprob result has no finite score")
        elif self.probability is not None or self.log_odds is not None:
            raise ValueError("hard-verdict result cannot contain a score")


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    """Functional cost ledger that always preserves the one-dollar reserve."""

    spent_usd: Decimal = Decimal("0")
    limit_usd: Decimal = MAX_COST_USD
    reserve_usd: Decimal = RESERVE_USD

    def __post_init__(self) -> None:
        if self.limit_usd <= 0 or self.reserve_usd < 0:
            raise ValueError("budget limit must be positive and reserve non-negative")
        if self.reserve_usd >= self.limit_usd:
            raise ValueError("budget reserve must be below the limit")
        if self.spent_usd < 0 or self.spent_usd > self.usable_usd:
            raise ValueError("recorded spend exceeds the usable budget")

    @property
    def usable_usd(self) -> Decimal:
        return self.limit_usd - self.reserve_usd

    @property
    def remaining_usd(self) -> Decimal:
        return self.usable_usd - self.spent_usd

    def allows(self, estimated_usd: Decimal | int | float | str) -> bool:
        estimate = _money(estimated_usd)
        return estimate >= 0 and estimate <= self.remaining_usd


@dataclass(frozen=True, slots=True)
class ConcurrencyObservation:
    concurrency: int
    requests: int
    terminal_failures: int
    requests_per_second: float

    def __post_init__(self) -> None:
        if self.concurrency < 1 or self.requests < 1:
            raise ValueError("concurrency and requests must be positive")
        if not 0 <= self.terminal_failures <= self.requests:
            raise ValueError("terminal failures must be within the request count")
        if not math.isfinite(self.requests_per_second) or self.requests_per_second < 0:
            raise ValueError("requests per second must be finite and non-negative")

    @property
    def terminal_failure_rate(self) -> float:
        return self.terminal_failures / self.requests


def request_cost_ceiling(
    endpoint: Endpoint,
    *,
    input_bytes: int,
    max_output_tokens: int = 16,
    attempts: int = MAX_ATTEMPTS,
) -> Decimal:
    """Conservatively reserve one request, including retries and chat framing."""

    if input_bytes < 0 or max_output_tokens < 1 or attempts < 1:
        raise ValueError("request cost inputs must be positive")
    input_tokens = input_bytes + REQUEST_OVERHEAD_TOKEN_CEILING
    return Decimal(attempts) * (
        Decimal(input_tokens)
        / Decimal(1_000_000)
        * (endpoint.input_per_million_usd or Decimal("1"))
        + Decimal(max_output_tokens)
        / Decimal(1_000_000)
        * (endpoint.output_per_million_usd or Decimal("1"))
    )


def parse_endpoint_snapshot(snapshot: dict) -> tuple[Endpoint, ...]:
    """Parse the public endpoint API while rejecting incomplete records."""

    if not isinstance(snapshot, dict):
        raise ValueError("endpoint snapshot must be an object")
    data = snapshot.get("data")
    if isinstance(data, dict):
        model = data.get("id") or snapshot.get("model")
        rows = data.get("endpoints")
    elif isinstance(data, list):
        model = snapshot.get("model")
        rows = data
    else:
        raise ValueError("endpoint snapshot has no data")
    if not isinstance(model, str) or not model or not isinstance(rows, list):
        raise ValueError("endpoint snapshot has no model or endpoint list")

    endpoints = tuple(_parse_endpoint(row, model) for row in rows)
    if not endpoints or len({endpoint.tag for endpoint in endpoints}) != len(endpoints):
        raise ValueError("endpoint snapshot must contain unique endpoint tags")
    return tuple(sorted(endpoints, key=lambda endpoint: endpoint.tag))


def capability_tiers(
    endpoints: tuple[Endpoint, ...],
    *,
    minimum_uptime_percent: float = MIN_PROVIDER_UPTIME_PERCENT,
) -> dict[Transport, tuple[Endpoint, ...]]:
    """Return uptime-qualified endpoints for every supported transport."""

    if not 0 <= minimum_uptime_percent <= 100:
        raise ValueError("minimum uptime must be between zero and one hundred")
    return {
        transport: tuple(
            endpoint
            for endpoint in endpoints
            if endpoint.uptime_percent >= minimum_uptime_percent
            and transport in endpoint.transports
        )
        for transport in (
            "strict_logprob",
            "strict_hard_verdict",
            "forced_tool",
            "relaxed_json",
        )
    }


def build_request(
    transport: Transport,
    *,
    provider: str,
    text: str,
    input_channel: Literal["direct_user", "untrusted_content"],
    model: str = MODEL,
    system_prompt: str | None = None,
    reasoning_effort: Literal["high", "max"] | None = None,
) -> dict:
    """Build one pinned OpenRouter request without performing any I/O."""

    if not provider or not text or not model:
        raise ValueError("provider, text, and model must be non-empty")
    if input_channel not in {"direct_user", "untrusted_content"}:
        raise ValueError("input channel is invalid")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or PROMPT.format(input_channel=input_channel),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 16,
        "reasoning": (
            {"effort": reasoning_effort, "exclude": True}
            if reasoning_effort
            else {"enabled": False, "exclude": True}
        ),
        "provider": {
            "order": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    if transport in {"strict_logprob", "strict_hard_verdict"}:
        body["response_format"] = copy.deepcopy(_STRICT_RESPONSE_FORMAT)
    elif transport == "forced_tool":
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": "Return the instruction-subversion verdict.",
                    "parameters": copy.deepcopy(_SCHEMA),
                    "strict": True,
                },
            }
        ]
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": _TOOL_NAME},
        }
    elif transport == "relaxed_json":
        body["response_format"] = {"type": "json_object"}
    else:
        raise ValueError(f"unsupported provider transport: {transport}")
    if transport == "strict_logprob":
        body["logprobs"] = True
        body["top_logprobs"] = 20
    if reasoning_effort:
        body["max_tokens"] = 1_024
    return body


def parse_result(
    payload: dict,
    *,
    row_id: str,
    transport: Transport,
    requested_provider: str,
    requested_model: str = MODEL,
    returned_provider: str | None = None,
    attempts: int = 1,
    client_seconds: float = 0.0,
    identity_validator: IdentityValidator | None = None,
) -> ProviderResult:
    """Validate a response and return only fields approved for persistence."""

    if not isinstance(payload, dict):
        raise ValueError("provider response must be an object")
    if not row_id or attempts < 1 or not math.isfinite(client_seconds):
        raise ValueError("result metadata is invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("provider response must contain exactly one choice")
    choice = choices[0]
    finish_reasons = {"stop", "tool_calls"} if transport == "forced_tool" else {"stop"}
    if (
        not isinstance(choice, dict)
        or choice.get("finish_reason") not in finish_reasons
    ):
        raise ValueError("provider response has an invalid finish reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("refusal"):
        raise ValueError("provider response has no usable message")

    verdict = _verdict_from_message(message, transport)
    returned_model = payload.get("model")
    if returned_provider is None and isinstance(payload.get("provider"), str):
        returned_provider = payload["provider"]
    validator = identity_validator or validate_exact_identity
    if not isinstance(returned_provider, str) or not isinstance(returned_model, str):
        raise ValueError("returned provider and model identity are required")
    validator(
        requested_provider,
        requested_model,
        returned_provider,
        returned_model,
    )

    probability = None
    log_odds = None
    if transport == "strict_logprob":
        logprob_0, logprob_1 = _parse_logprobs(choice, verdict)
        log_odds = logprob_1 - logprob_0
        probability = _sigmoid(log_odds)
    usage = _parse_usage(payload.get("usage"))
    return ProviderResult(
        row_id=row_id,
        transport=transport,
        requested_provider=requested_provider,
        requested_model=requested_model,
        returned_provider=returned_provider,
        returned_model=returned_model,
        status="ok",
        verdict=verdict,
        probability=probability,
        log_odds=log_odds,
        attempts=attempts,
        client_seconds=client_seconds,
        **usage,
    )


def failed_result(
    *,
    row_id: str,
    transport: Transport,
    requested_provider: str,
    failure_code: FailureCode,
    requested_model: str = MODEL,
    attempts: int = 1,
    client_seconds: float = 0.0,
) -> ProviderResult:
    """Create a bounded failure record without retaining provider details."""

    return ProviderResult(
        row_id=row_id,
        transport=transport,
        requested_provider=requested_provider,
        requested_model=requested_model,
        returned_provider=None,
        returned_model=None,
        status="failed",
        verdict=None,
        probability=None,
        log_odds=None,
        attempts=attempts,
        client_seconds=client_seconds,
        failure_code=failure_code,
    )


def validate_exact_identity(
    requested_provider: str,
    requested_model: str,
    returned_provider: str,
    returned_model: str,
) -> None:
    """Fail when OpenRouter generation metadata differs from the pinned route."""

    if _provider_key(requested_provider) != _provider_key(returned_provider):
        raise ValueError("returned provider does not match the pinned provider")
    if requested_model != returned_model:
        raise ValueError("returned model does not match the requested model")


def can_expand_canary(
    records: tuple[ProviderResult, ...], *, expected_rows: int = 16
) -> bool:
    """Require unique rows and perfect parsed, identity-checked canary results."""

    return (
        len(records) == expected_rows
        and len({record.row_id for record in records}) == expected_rows
        and all(record.status == "ok" for record in records)
    )


def may_probe_concurrency_eight(observation_at_four: ConcurrencyObservation) -> bool:
    """Allow an eight-way probe only after the four-way reliability gate."""

    return (
        observation_at_four.concurrency == 4
        and observation_at_four.terminal_failure_rate <= MAX_TERMINAL_FAILURE_RATE
    )


def accept_concurrency_eight(
    observation_at_four: ConcurrencyObservation,
    observation_at_eight: ConcurrencyObservation,
) -> bool:
    """Keep eight-way execution only when it is reliable and faster."""

    return (
        may_probe_concurrency_eight(observation_at_four)
        and observation_at_eight.concurrency == 8
        and observation_at_eight.terminal_failure_rate <= MAX_TERMINAL_FAILURE_RATE
        and observation_at_eight.requests_per_second
        > observation_at_four.requests_per_second
    )


def _parse_endpoint(row: object, model: str) -> Endpoint:
    if not isinstance(row, dict):
        raise ValueError("endpoint record must be an object")
    name = row.get("name") or row.get("provider_name")
    tag = row.get("tag")
    parameters = row.get("supported_parameters")
    uptime = row.get("uptime_last_30m", row.get("uptime"))
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(tag, str)
        or not tag
        or not isinstance(parameters, list)
        or not all(isinstance(value, str) for value in parameters)
    ):
        raise ValueError("endpoint record is missing identity or capabilities")
    uptime_percent = _uptime_percent(uptime)
    pricing = row.get("pricing")
    if not isinstance(pricing, dict):
        pricing = {}
    return Endpoint(
        provider=tag.split("/", 1)[0],
        name=name,
        tag=tag,
        model=model,
        quantization=(
            row["quantization"] if isinstance(row.get("quantization"), str) else None
        ),
        uptime_percent=uptime_percent,
        supported_parameters=frozenset(parameters),
        input_per_million_usd=_per_million(pricing.get("prompt")),
        output_per_million_usd=_per_million(pricing.get("completion")),
        cache_read_per_million_usd=_per_million(pricing.get("input_cache_read")),
    )


def _uptime_percent(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("endpoint uptime must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("endpoint uptime must be finite and non-negative")
    if result <= 1:
        result *= 100
    if result > 100:
        raise ValueError("endpoint uptime cannot exceed one hundred percent")
    return result


def _per_million(value: object) -> Decimal | None:
    if value is None:
        return None
    price = _money(value)
    if price < 0:
        raise ValueError("provider price cannot be negative")
    return price * Decimal(1_000_000)


def _money(value: Decimal | int | float | str | object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("money must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("money must be numeric") from error
    if not result.is_finite():
        raise ValueError("money must be finite")
    return result


def _provider_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _verdict_from_message(message: dict, transport: Transport) -> int:
    if transport == "forced_tool":
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("provider response must contain exactly one tool call")
        call = calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != _TOOL_NAME:
            raise ValueError("provider response called the wrong tool")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("provider tool arguments must be JSON text")
        return _parse_verdict(arguments)
    if transport not in {"strict_logprob", "strict_hard_verdict", "relaxed_json"}:
        raise ValueError(f"unsupported provider transport: {transport}")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("provider response has no content")
    return _parse_verdict(content)


def _parse_usage(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    prompt_details = value.get("prompt_tokens_details")
    cached = (
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, dict)
        else None
    )
    return {
        "prompt_tokens": _count(value.get("prompt_tokens")),
        "completion_tokens": _count(value.get("completion_tokens")),
        "cached_tokens": _count(cached),
        "cost_usd": _optional_money(value.get("cost")),
    }


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _optional_money(value: object) -> Decimal | None:
    if value is None:
        return None
    result = _money(value)
    if result < 0:
        raise ValueError("provider cost cannot be negative")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
