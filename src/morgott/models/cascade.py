"""Maintained advisory cascade over the registered mmBERT and DeepSeek profile."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import tempfile
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from .deepseek_nooa import (
    MODEL,
    PACKET_PROMPT_SHA256,
    PROMPT_SHA256,
    PROVIDER,
    REMOTE_CONCURRENCY,
    REQUEST_SHA256,
    DeepSeekReviewer,
    WindowReview,
)
from .downstream import (
    LLM_FLAG_PROBABILITY,
    PIPELINE_PROFILE,
    THRESHOLD_CONTRACT,
    THRESHOLD_SHA256,
    route,
    subversion_probability,
)
from .mmbert.core import MODEL_ID, MODEL_REVISION
from .mmbert.inference import verified_artifact_path
from .mmbert.serving import (
    DEFAULT_MODEL_KEY,
    MODEL_MAX_TOKENS,
    WINDOW_OVERLAP,
    MmbertRuntime,
    PreparedText,
    Window,
)
from .retrieval import (
    DENSE_RRF_WEIGHT,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_REQUEST_SHA256,
    RRF_K,
    SPARSE_CANDIDATES,
    SPARSE_RRF_WEIGHT,
    OpenRouterEmbedder,
    RetrievalEngine,
    RetrievalResult,
    RetrievalTrace,
)

MAX_REMOTE_WINDOWS = 128
FULL_CONTEXT_REVIEW_INDEX = -1
ALLOWED_CHANNELS = frozenset({"direct_user", "untrusted_content"})
POLICY_FORMAT = "morgott-advisory-cascade-profile-v2"


def _verify_retrieval_parity(
    evidence: dict,
    retrieval_manifest: dict,
    retrieval_manifest_sha256: str,
) -> None:
    queries = evidence.get("queries")
    source = evidence.get("source")
    manifest_source = retrieval_manifest.get("source")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "passed"
        or evidence.get("provider_calls") is not False
        or evidence.get("contains_raw_text_or_vectors") is not False
        or type(queries) is not int
        or queries <= 0
        or evidence.get("different_packets") != 0
        or evidence.get("exact_packet_matches") != queries
        or evidence.get("sparse_branch_differences") != 0
        or evidence.get("variant") != "faiss_hnsw_ef1024_top160"
        or evidence.get("method")
        != "hybrid_pplx-4b_unicode_partitioned8_sparse50_dense20_rrf2_replay"
        or evidence.get("rrf")
        != {
            "dense_weight": DENSE_RRF_WEIGHT,
            "k": RRF_K,
            "sparse_weight": SPARSE_RRF_WEIGHT,
        }
        or not isinstance(source, dict)
        or not isinstance(manifest_source, dict)
        or source.get("serving_manifest_sha256") != retrieval_manifest_sha256
        or source.get("hnsw_extension_sha256")
        != manifest_source.get("extension_sha256")
        or source.get("sparse_identity_sha256")
        != manifest_source.get("sparse_identity_sha256")
    ):
        raise ValueError("registered retrieval parity evidence is invalid")


def _verify_registered_policy(manifest_path: Path) -> tuple[str, Path, str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("models", {}).get(DEFAULT_MODEL_KEY)
    serving = entry.get("serving") if isinstance(entry, dict) else None
    if not isinstance(serving, dict):
        raise ValueError("registered cascade policy is missing")
    retrieval_spec = serving.get("retrieval")
    retrieval_manifest_spec = (
        retrieval_spec.get("manifest") if isinstance(retrieval_spec, dict) else None
    )
    if (
        not isinstance(retrieval_spec, dict)
        or retrieval_spec.get("format") != "morgott-lineage-hybrid-v1"
        or not isinstance(retrieval_manifest_spec, dict)
    ):
        raise ValueError("registered retrieval bundle is missing")
    retrieval_manifest = verified_artifact_path(
        manifest_path.parent,
        retrieval_manifest_spec,
        name="retrieval manifest",
    )
    retrieval_manifest_data = json.loads(retrieval_manifest.read_text(encoding="utf-8"))
    spec = serving.get("cascade_policy")
    policy_path = verified_artifact_path(
        manifest_path.parent,
        spec,
        name="cascade policy",
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    evidence_path = verified_artifact_path(
        manifest_path.parent,
        policy.get("evidence"),
        name="retrieval parity evidence",
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _verify_retrieval_parity(
        evidence,
        retrieval_manifest_data,
        retrieval_manifest_spec["sha256"],
    )
    expected_contract = {
        "advisory_decision": "allow",
        "cascade": {
            "max_remote_windows": MAX_REMOTE_WINDOWS,
            "max_tokens": MODEL_MAX_TOKENS,
            "untrusted_multi_window": "full_context_first_then_middle_windows",
            "window_overlap": WINDOW_OVERLAP,
            "window_review_batch_size": REMOTE_CONCURRENCY,
        },
        "model_key": DEFAULT_MODEL_KEY,
        "profile": PIPELINE_PROFILE,
        "review_failure": "restrict_incomplete",
        "reviewer": {
            "fallbacks_allowed": False,
            "logprobs": True,
            "model": MODEL,
            "no_example_prompt_sha256": PROMPT_SHA256,
            "prompt_sha256": PACKET_PROMPT_SHA256,
            "reasoning_enabled": False,
            "remote_concurrency": REMOTE_CONCURRENCY,
            "request_sha256": REQUEST_SHA256,
            "requested_provider": PROVIDER,
            "strict_json_schema": True,
        },
        "retrieval": {
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_request_sha256": EMBEDDING_REQUEST_SHA256,
            "examples_per_request": 4,
            "failure_behavior": {
                "embedding_or_dense": "no_example_reviewer",
                "sparse": "dense_packet",
            },
            "labels_per_class": 2,
            "manifest_sha256": retrieval_manifest_spec["sha256"],
            "rrf": {
                "dense_weight": DENSE_RRF_WEIGHT,
                "k": RRF_K,
                "sparse_weight": SPARSE_RRF_WEIGHT,
            },
            "sparse_candidates_per_label": SPARSE_CANDIDATES,
            "variant": "lineage_hybrid_v1",
        },
        "threshold_sha256": THRESHOLD_SHA256,
        "thresholds": THRESHOLD_CONTRACT,
    }
    if (
        policy.get("schema_version") != 2
        or policy.get("format") != POLICY_FORMAT
        or policy.get("status") != "maintained_advisory"
        or policy.get("advisory_only") is not True
        or policy.get("runtime_contract") != expected_contract
    ):
        raise ValueError("registered cascade policy differs from maintained code")
    return spec["sha256"], retrieval_manifest, retrieval_manifest_spec["sha256"]


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
        and (review.retrieval is None or isinstance(review.retrieval, RetrievalTrace))
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
        retrieval=(
            review.retrieval if isinstance(review.retrieval, RetrievalTrace) else None
        ),
    )


class _WindowScorer(Protocol):
    def prepare(self, text: str) -> PreparedText: ...

    def score(self, windows: tuple[Window, ...]) -> tuple[float, ...]: ...


class _WindowReviewer(Protocol):
    async def review(
        self,
        text: str,
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
        query_text: str,
    ) -> WindowReview: ...

    async def aclose(self) -> None: ...


class RetrievalReviewer:
    """Add fail-soft retrieval to the maintained DeepSeek reviewer."""

    def __init__(
        self,
        reviewer: DeepSeekReviewer,
        engine: RetrievalEngine,
        embedder: OpenRouterEmbedder,
    ) -> None:
        self._reviewer = reviewer
        self._engine = engine
        self._embedder = embedder

    @property
    def available(self) -> bool:
        return self._engine.available

    @property
    def manifest_sha256(self) -> str:
        return self._engine.manifest_sha256

    async def review(
        self,
        text: str,
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
        query_text: str,
    ) -> WindowReview:
        if not isinstance(text, str) or not text:
            raise ValueError("review text must be a non-empty string")
        if not isinstance(query_text, str) or not query_text:
            raise ValueError("retrieval query text must be a non-empty string")
        if input_channel not in {"direct_user", "untrusted_content"}:
            raise ValueError("retrieval input channel is invalid")
        if not self._engine.available:
            review = await self._reviewer.review(
                text,
                input_channel=input_channel,
            )
            return replace(
                review, retrieval=_empty_retrieval_trace("bundle_unavailable")
            )

        started = time.perf_counter()
        sparse_task = asyncio.create_task(
            asyncio.to_thread(self._engine.sparse, query_text, input_channel)
        )
        try:
            embedding = await self._embedder.embed(query_text)
        except asyncio.CancelledError:
            sparse_task.cancel()
            raise
        except Exception:
            embedding_ms = (time.perf_counter() - started) * 1_000
            await asyncio.gather(sparse_task, return_exceptions=True)
            retrieval_ms = (time.perf_counter() - started) * 1_000
            review = await self._reviewer.review(
                text,
                input_channel=input_channel,
            )
            return replace(
                review,
                retrieval=RetrievalTrace(
                    status="embedding_failed",
                    total_ms=retrieval_ms,
                    embedding_ms=embedding_ms,
                    dense_ms=0.0,
                    sparse_ms=0.0,
                    fusion_ms=0.0,
                    embedding_input_tokens=None,
                    embedding_cost_usd=None,
                    selected_example_count=0,
                    selected_packet_sha256=None,
                    fallback_reason="embedding_failed",
                ),
            )
        try:
            sparse = await sparse_task
            result = await asyncio.to_thread(
                self._engine.retrieve,
                query_text,
                input_channel,
                embedding.vector,
                sparse,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            result = _failed_retrieval_result("retrieval_failed")
        trace = RetrievalTrace(
            status=(
                result.fallback_reason
                if result.status == "no_examples" and result.fallback_reason
                else result.status
            ),
            total_ms=(time.perf_counter() - started) * 1_000,
            embedding_ms=embedding.elapsed_ms,
            dense_ms=result.dense_ms,
            sparse_ms=result.sparse_ms,
            fusion_ms=result.fusion_ms,
            embedding_input_tokens=embedding.input_tokens,
            embedding_cost_usd=embedding.cost_usd,
            selected_example_count=len(result.examples),
            selected_packet_sha256=(
                hashlib.sha256(
                    json.dumps(
                        tuple(example.example_id for example in result.examples),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if len(result.examples) == 4
                else None
            ),
            fallback_reason=result.fallback_reason,
        )
        if len(result.examples) == 4:
            review = await self._reviewer.review_with_examples(
                text,
                input_channel=input_channel,
                examples=tuple(
                    (example.label, example.text) for example in result.examples
                ),
            )
        else:
            review = await self._reviewer.review(
                text,
                input_channel=input_channel,
            )
        return replace(review, retrieval=trace)

    async def aclose(self) -> None:
        await asyncio.gather(self._reviewer.aclose(), self._embedder.aclose())


def _empty_retrieval_trace(status: str) -> RetrievalTrace:
    return RetrievalTrace(
        status=status,
        total_ms=0.0,
        embedding_ms=0.0,
        dense_ms=0.0,
        sparse_ms=0.0,
        fusion_ms=0.0,
        embedding_input_tokens=None,
        embedding_cost_usd=None,
        selected_example_count=0,
        selected_packet_sha256=None,
        fallback_reason=status,
    )


def _failed_retrieval_result(reason: str) -> RetrievalResult:
    return RetrievalResult(
        status="no_examples",
        examples=(),
        fallback_reason=reason,
        dense_ms=0.0,
        sparse_ms=0.0,
        fusion_ms=0.0,
    )


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
    retrieval: RetrievalTrace | None


def _review_record(index: int, review: WindowReview) -> ReviewedWindow:
    review = _validated_review(review)
    return ReviewedWindow(
        index=index,
        status=review.status,
        probability=review.probability,
        log_odds=review.log_odds,
        attempts=review.attempts,
        latency_ms=review.latency_ms,
        input_tokens=review.input_tokens,
        output_tokens=review.output_tokens,
        failure_code=review.failure_code,
        retrieval=review.retrieval,
    )


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
    retrieval_status: str | None
    retrieval_fallback_reason: str | None
    retrieval_latency_ms: float
    embedding_latency_ms: float
    dense_retrieval_latency_ms: float
    sparse_retrieval_latency_ms: float
    fusion_latency_ms: float
    embedding_input_tokens: int | None
    embedding_cost_usd: float | None
    selected_example_count: int
    retrieval_packet_sha256: str | None
    model_key: str
    model_id: str
    model_revision: str
    runtime: str
    onnx_sha256: str | None
    tokenizer_sha256: str | None
    prompt_sha256: str | None
    no_example_prompt_sha256: str
    embedding_request_sha256: str
    provider: str
    provider_request_sha256: str
    pipeline_profile: str
    policy_sha256: str | None
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
        policy_sha256: str | None = None,
    ) -> None:
        self._scorer = scorer
        self._reviewer = reviewer
        self._policy_sha256 = policy_sha256
        self._local_semaphore = asyncio.Semaphore(1)
        self._remote_semaphore = asyncio.Semaphore(REMOTE_CONCURRENCY)

    @property
    def runtime_identity(self):
        return getattr(self._scorer, "identity", None)

    @property
    def policy_sha256(self) -> str | None:
        return self._policy_sha256

    @property
    def retrieval_enabled(self) -> bool:
        return bool(
            isinstance(self._reviewer, RetrievalReviewer) and self._reviewer.available
        )

    @property
    def retrieval_manifest_sha256(self) -> str | None:
        return (
            self._reviewer.manifest_sha256
            if isinstance(self._reviewer, RetrievalReviewer)
            else None
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        manifest_path: Path,
        inference_precision: Literal["auto", "bf16", "fp32"] = "bf16",
    ) -> CascadeScanner:
        policy_sha256, retrieval_manifest, retrieval_manifest_sha256 = (
            _verify_registered_policy(manifest_path)
        )
        scorer = MmbertRuntime.from_artifacts(
            manifest_path,
            inference_precision=inference_precision,
        )
        engine = RetrievalEngine(
            retrieval_manifest.parent,
            retrieval_manifest_sha256,
        )
        if not engine.available:
            raise ValueError("registered retrieval bundle failed verification")
        reviewer = RetrievalReviewer(
            DeepSeekReviewer.from_env(),
            engine,
            OpenRouterEmbedder.from_env(),
        )
        return cls(
            scorer=scorer,
            reviewer=reviewer,
            policy_sha256=policy_sha256,
        )

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
        local_routes = tuple(
            route(score, input_channel=input_channel) for score in scores
        )
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
            retrievals = [
                review.retrieval for review in reviews if review.retrieval is not None
            ]
            prompt_hashes = {
                (
                    PACKET_PROMPT_SHA256
                    if review.retrieval is not None
                    and review.retrieval.selected_example_count == 4
                    else PROMPT_SHA256
                )
                for review in reviews
            }
            non_ok_retrieval = next(
                (trace.status for trace in retrievals if trace.status != "ok"), None
            )
            fallback_reasons = {
                trace.fallback_reason
                for trace in retrievals
                if trace.fallback_reason is not None
            }
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
                retrieval_status=(non_ok_retrieval or ("ok" if retrievals else None)),
                retrieval_fallback_reason=(
                    next(iter(fallback_reasons))
                    if len(fallback_reasons) == 1
                    else "multiple"
                    if fallback_reasons
                    else None
                ),
                retrieval_latency_ms=sum(trace.total_ms for trace in retrievals),
                embedding_latency_ms=sum(trace.embedding_ms for trace in retrievals),
                dense_retrieval_latency_ms=sum(trace.dense_ms for trace in retrievals),
                sparse_retrieval_latency_ms=sum(
                    trace.sparse_ms for trace in retrievals
                ),
                fusion_latency_ms=sum(trace.fusion_ms for trace in retrievals),
                embedding_input_tokens=(
                    sum(
                        trace.embedding_input_tokens
                        for trace in retrievals
                        if trace.embedding_input_tokens is not None
                    )
                    if any(
                        trace.embedding_input_tokens is not None for trace in retrievals
                    )
                    else None
                ),
                embedding_cost_usd=(
                    sum(
                        trace.embedding_cost_usd
                        for trace in retrievals
                        if trace.embedding_cost_usd is not None
                    )
                    if any(trace.embedding_cost_usd is not None for trace in retrievals)
                    else None
                ),
                selected_example_count=sum(
                    trace.selected_example_count for trace in retrievals
                ),
                retrieval_packet_sha256=(
                    retrievals[0].selected_packet_sha256
                    if len(retrievals) == 1
                    else None
                ),
                model_key=getattr(identity, "model_key", DEFAULT_MODEL_KEY),
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                runtime=getattr(identity, "runtime", "injected"),
                onnx_sha256=getattr(identity, "onnx_sha256", None),
                tokenizer_sha256=getattr(identity, "tokenizer_sha256", None),
                prompt_sha256=(
                    next(iter(prompt_hashes)) if len(prompt_hashes) == 1 else None
                ),
                no_example_prompt_sha256=PROMPT_SHA256,
                embedding_request_sha256=EMBEDDING_REQUEST_SHA256,
                provider=PROVIDER,
                provider_request_sha256=REQUEST_SHA256,
                pipeline_profile=PIPELINE_PROFILE,
                policy_sha256=self._policy_sha256,
                threshold_sha256=THRESHOLD_SHA256,
                local_latency_ms=local_latency_ms,
                provider_latency_ms=provider_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
            )

        if high_windows:
            return finish("restrict", True, "mmbert_high")
        full_context_review = (
            self._reviewer is not None
            and input_channel == "untrusted_content"
            and len(prepared.windows) > 1
        )
        if not full_context_review and low_windows == len(scores):
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
        if (
            len(prepared.windows) if full_context_review else len(pending_reviews)
        ) > MAX_REMOTE_WINDOWS:
            return finish("restrict", False, "deepseek_window_limit")

        provider_started = time.perf_counter()
        reviews = []
        if full_context_review:
            query_index = max(range(len(scores)), key=scores.__getitem__)
            query_window = prepared.windows[query_index]
            full_review = _review_record(
                FULL_CONTEXT_REVIEW_INDEX,
                await self._review_window(
                    prepared.normalized_text,
                    input_channel=input_channel,
                    query_text=prepared.normalized_text[
                        query_window.char_start : query_window.char_end
                    ],
                ),
            )
            reviews.append(full_review)
            provider_latency_ms = (time.perf_counter() - provider_started) * 1000
            if full_review.status == "failed":
                return finish(
                    "restrict",
                    False,
                    "deepseek_failed",
                    reviews=tuple(reviews),
                    provider_latency_ms=provider_latency_ms,
                )
            if (
                full_review.probability is not None
                and full_review.probability >= LLM_FLAG_PROBABILITY
            ):
                return finish(
                    "restrict",
                    True,
                    "deepseek_full_context_flag",
                    reviews=tuple(reviews),
                    provider_latency_ms=provider_latency_ms,
                )
            if not pending_reviews:
                return finish(
                    "pass",
                    True,
                    "deepseek_full_context_clear",
                    reviews=tuple(reviews),
                    provider_latency_ms=provider_latency_ms,
                )

        outcome = None
        for offset in range(0, len(pending_reviews), REMOTE_CONCURRENCY):
            batch = pending_reviews[offset : offset + REMOTE_CONCURRENCY]
            raw_reviews = await asyncio.gather(
                *(
                    self._review_window(
                        (
                            text
                            if len(prepared.windows) == 1
                            else prepared.normalized_text[
                                window.char_start : window.char_end
                            ]
                        ),
                        input_channel=input_channel,
                        query_text=(
                            text
                            if len(prepared.windows) == 1
                            else prepared.normalized_text[
                                window.char_start : window.char_end
                            ]
                        ),
                    )
                    for window, _ in batch
                )
            )
            batch_reviews = tuple(
                _review_record(window.index, review)
                for (window, _), review in zip(
                    batch,
                    raw_reviews,
                    strict=True,
                )
            )
            batch_routes = tuple(
                route(
                    score,
                    input_channel=input_channel,
                    llm_probability=(
                        review.probability if review.status == "ok" else None
                    ),
                    llm_failed=review.status == "failed",
                )
                for (_, score), review in zip(batch, batch_reviews, strict=True)
            )
            reviews.extend(batch_reviews)
            outcome = next(
                (
                    result
                    for result in batch_routes
                    if result.reason == "deepseek_failed"
                ),
                None,
            )
            if outcome is None:
                outcome = next(
                    (result for result in batch_routes if result.route == "restrict"),
                    None,
                )
            if outcome is not None:
                break
            outcome = batch_routes[0]
        provider_latency_ms = (time.perf_counter() - provider_started) * 1000
        return finish(
            outcome.route,
            outcome.reason != "deepseek_failed",
            outcome.reason,
            reviews=tuple(reviews),
            provider_latency_ms=provider_latency_ms,
        )

    async def _review_window(
        self,
        text: str,
        *,
        input_channel: Literal["direct_user", "untrusted_content"],
        query_text: str,
    ) -> WindowReview:
        if self._reviewer is None:
            raise AssertionError("remote reviewer is not configured")
        async with self._remote_semaphore:
            return await self._reviewer.review(
                text,
                input_channel=input_channel,
                query_text=query_text,
            )

    def _prepare_and_score(
        self,
        text: str,
    ) -> tuple[PreparedText, tuple[float, ...]]:
        prepared = self._scorer.prepare(text)
        scores = tuple(self._scorer.score(prepared.windows))
        if len(scores) != len(prepared.windows) or not scores:
            raise ValueError("mmBERT returned the wrong number of scores")
        return prepared, scores
