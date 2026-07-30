"""Async shadow cascade over the retained mmBERT and DeepSeek candidates."""

from __future__ import annotations

import asyncio
import hashlib
import math
import tempfile
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .deepseek_nooa import (
    PROMPT_SHA256,
    PROVIDER,
    REMOTE_CONCURRENCY,
    REQUEST_SHA256,
    DeepSeekReviewer,
    WindowReview,
)
from .downstream import (
    THRESHOLD_SHA256,
    route,
    subversion_probability,
)
from .mmbert.core import MODEL_ID, MODEL_REVISION
from .mmbert.serving import DEFAULT_MODEL_KEY, MmbertRuntime, PreparedText, Window

MAX_REMOTE_WINDOWS = 128
ALLOWED_CHANNELS = frozenset({"direct_user", "untrusted_content"})


def _validated_review(review: WindowReview) -> WindowReview:
    attempts = review.attempts
    latency_ms = review.latency_ms
    common_valid = (
        type(attempts) is int
        and attempts >= 1
        and isinstance(latency_ms, int | float)
        and not isinstance(latency_ms, bool)
        and math.isfinite(latency_ms)
        and latency_ms >= 0
    )
    if review.status == "ok":
        valid = (
            common_valid
            and isinstance(review.probability, int | float)
            and not isinstance(review.probability, bool)
            and math.isfinite(review.probability)
            and 0 <= review.probability <= 1
            and isinstance(review.log_odds, int | float)
            and not isinstance(review.log_odds, bool)
            and math.isfinite(review.log_odds)
            and math.isclose(
                float(review.probability),
                subversion_probability(0.0, float(review.log_odds)),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and review.failure_code is None
        )
    else:
        valid = (
            review.status == "failed"
            and common_valid
            and review.probability is None
            and review.log_odds is None
            and isinstance(review.failure_code, str)
            and bool(review.failure_code)
        )
    if valid:
        return review
    return WindowReview(
        status="failed",
        probability=None,
        log_odds=None,
        attempts=attempts if type(attempts) is int and attempts >= 1 else 1,
        latency_ms=(
            float(latency_ms)
            if isinstance(latency_ms, int | float)
            and not isinstance(latency_ms, bool)
            and math.isfinite(latency_ms)
            and latency_ms >= 0
            else 0.0
        ),
        failure_code="invalid_review",
    )


class _WindowScorer(Protocol):
    def prepare(self, text: str) -> PreparedText: ...

    def score(self, windows: tuple[Window, ...]) -> tuple[float, ...]: ...


class _WindowReviewer(Protocol):
    async def review(self, text: str) -> WindowReview: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReviewedWindow:
    index: int
    status: Literal["ok", "failed"]
    probability: float | None
    log_odds: float | None
    attempts: int
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class CascadeAssessment:
    decision: Literal["allow"]
    advisory_only: Literal[True]
    advisory_route: Literal["pass", "restrict"]
    complete: bool
    reason: str
    input_channel: Literal["direct_user", "untrusted_content"]
    artifact_sha256: str
    token_count: int
    window_count: int
    low_windows: int
    middle_windows: int
    high_windows: int
    max_mmbert_score: float
    reviewed_windows: tuple[ReviewedWindow, ...]
    deepseek_calls: int
    deepseek_failures: int
    max_deepseek_probability: float | None
    model_key: str
    model_id: str
    model_revision: str
    runtime: str
    onnx_sha256: str | None
    tokenizer_sha256: str | None
    prompt_sha256: str
    provider: str
    provider_request_sha256: str
    threshold_sha256: str
    local_latency_ms: float
    provider_latency_ms: float
    total_latency_ms: float


class CascadeScanner:
    """Assess complete artifacts behind one small advisory interface."""

    def __init__(
        self,
        *,
        scorer: _WindowScorer,
        reviewer: _WindowReviewer | None,
    ) -> None:
        self._scorer = scorer
        self._reviewer = reviewer
        self._local_semaphore = asyncio.Semaphore(1)
        self._remote_semaphore = asyncio.Semaphore(REMOTE_CONCURRENCY)

    @classmethod
    def from_artifacts(
        cls,
        *,
        manifest_path: Path,
        allow_remote: bool,
    ) -> CascadeScanner:
        if not isinstance(allow_remote, bool):
            raise ValueError("allow_remote must be a boolean")
        scorer = MmbertRuntime.from_artifacts(manifest_path)
        reviewer = DeepSeekReviewer.from_env() if allow_remote else None
        return cls(scorer=scorer, reviewer=reviewer)

    async def aclose(self) -> None:
        if self._reviewer is not None:
            await self._reviewer.aclose()

    async def assess_text(
        self,
        text: str,
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
    ) -> CascadeAssessment:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        self._validate_channel(input_channel)
        artifact_sha256 = hashlib.sha256(text.encode()).hexdigest()
        return await self._assess(
            text,
            input_channel=input_channel,
            artifact_sha256=artifact_sha256,
        )

    async def assess_chunks(
        self,
        chunks: AsyncIterable[str],
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
    ) -> CascadeAssessment:
        self._validate_channel(input_channel)
        digest = hashlib.sha256()
        with tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            newline="",
        ) as handle:
            async for chunk in chunks:
                if not isinstance(chunk, str):
                    raise ValueError("input chunks must be strings")
                handle.write(chunk)
                digest.update(chunk.encode())
            handle.seek(0)
            text = handle.read()
        if not text:
            raise ValueError("chunked input must contain text")
        return await self._assess(
            text,
            input_channel=input_channel,
            artifact_sha256=digest.hexdigest(),
        )

    @staticmethod
    def _validate_channel(input_channel: str) -> None:
        if input_channel not in ALLOWED_CHANNELS:
            raise ValueError("input_channel must come from trusted runtime metadata")

    async def _assess(
        self,
        text: str,
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
        artifact_sha256: str,
    ) -> CascadeAssessment:
        started = time.perf_counter()
        local_started = time.perf_counter()
        async with self._local_semaphore:
            local_task = asyncio.create_task(
                asyncio.to_thread(self._prepare_and_score, text)
            )
            try:
                prepared, scores = await asyncio.shield(local_task)
            except asyncio.CancelledError:
                # A cancelled to_thread call keeps running. Hold the semaphore until
                # local inference returns so a second call cannot overlap it.
                await asyncio.gather(local_task, return_exceptions=True)
                raise
        local_latency_ms = (time.perf_counter() - local_started) * 1000
        local_routes = tuple(route(score) for score in scores)
        low_windows = sum(result.route == "pass" for result in local_routes)
        high_windows = sum(result.route == "restrict" for result in local_routes)
        middle_windows = sum(result.route == "review" for result in local_routes)

        def finish(
            advisory_route: Literal["pass", "restrict"],
            complete: bool,
            reason: str,
            *,
            reviews: tuple[ReviewedWindow, ...] = (),
            provider_latency_ms: float = 0.0,
        ) -> CascadeAssessment:
            identity = getattr(self._scorer, "identity", None)
            probabilities = [
                review.probability
                for review in reviews
                if review.status == "ok" and review.probability is not None
            ]
            return CascadeAssessment(
                decision="allow",
                advisory_only=True,
                advisory_route=advisory_route,
                complete=complete,
                reason=reason,
                input_channel=input_channel,
                artifact_sha256=artifact_sha256,
                token_count=prepared.token_count,
                window_count=len(prepared.windows),
                low_windows=low_windows,
                middle_windows=middle_windows,
                high_windows=high_windows,
                max_mmbert_score=max(scores),
                reviewed_windows=reviews,
                deepseek_calls=sum(review.attempts for review in reviews),
                deepseek_failures=sum(review.status == "failed" for review in reviews),
                max_deepseek_probability=max(probabilities, default=None),
                model_key=DEFAULT_MODEL_KEY,
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                runtime=getattr(identity, "runtime", "injected"),
                onnx_sha256=getattr(identity, "onnx_sha256", None),
                tokenizer_sha256=getattr(identity, "tokenizer_sha256", None),
                prompt_sha256=PROMPT_SHA256,
                provider=PROVIDER,
                provider_request_sha256=REQUEST_SHA256,
                threshold_sha256=THRESHOLD_SHA256,
                local_latency_ms=local_latency_ms,
                provider_latency_ms=provider_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
            )

        if high_windows:
            return finish("restrict", True, "mmbert_high")
        if low_windows == len(scores):
            return finish("pass", True, "mmbert_low")
        if self._reviewer is None:
            return finish("restrict", False, "remote_review_disabled")

        pending_reviews = tuple(
            (window, score)
            for window, score, result in zip(
                prepared.windows,
                scores,
                local_routes,
                strict=True,
            )
            if result.route == "review"
        )
        if len(pending_reviews) > MAX_REMOTE_WINDOWS:
            return finish("restrict", False, "deepseek_window_limit")

        provider_started = time.perf_counter()
        raw_reviews = await asyncio.gather(
            *(
                self._review_window(
                    text
                    if len(prepared.windows) == 1
                    else prepared.normalized_text[window.char_start : window.char_end]
                )
                for window, _ in pending_reviews
            )
        )
        provider_latency_ms = (time.perf_counter() - provider_started) * 1000
        reviews = tuple(
            ReviewedWindow(
                index=window.index,
                status=review.status,
                probability=review.probability,
                log_odds=review.log_odds,
                attempts=review.attempts,
                latency_ms=review.latency_ms,
                input_tokens=review.input_tokens,
                output_tokens=review.output_tokens,
                failure_code=review.failure_code,
            )
            for (window, _), review in zip(
                pending_reviews,
                map(_validated_review, raw_reviews),
                strict=True,
            )
        )
        review_routes = tuple(
            route(
                score,
                llm_probability=(review.probability if review.status == "ok" else None),
                llm_failed=review.status == "failed",
            )
            for (_, score), review in zip(pending_reviews, reviews, strict=True)
        )
        outcome = next(
            (result for result in review_routes if result.reason == "deepseek_failed"),
            None,
        )
        if outcome is None:
            outcome = next(
                (result for result in review_routes if result.route == "restrict"),
                review_routes[0],
            )
        return finish(
            outcome.route,
            outcome.reason != "deepseek_failed",
            outcome.reason,
            reviews=reviews,
            provider_latency_ms=provider_latency_ms,
        )

    async def _review_window(self, text: str) -> WindowReview:
        if self._reviewer is None:
            raise AssertionError("remote reviewer is not configured")
        async with self._remote_semaphore:
            return await self._reviewer.review(text)

    def _prepare_and_score(
        self,
        text: str,
    ) -> tuple[PreparedText, tuple[float, ...]]:
        prepared = self._scorer.prepare(text)
        scores = tuple(self._scorer.score(prepared.windows))
        if len(scores) != len(prepared.windows) or not scores:
            raise ValueError("mmBERT returned the wrong number of scores")
        return prepared, scores
