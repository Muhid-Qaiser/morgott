import asyncio
import hashlib
import io
import json
import math
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from morgott.cli import main
from morgott.models.cascade import (
    FULL_CONTEXT_REVIEW_INDEX,
    CascadeScanner,
    RetrievalReviewer,
    _verify_retrieval_parity,
)
from morgott.models.deepseek_nooa import PROMPT_SHA256, REMOTE_CONCURRENCY, WindowReview
from morgott.models.mmbert.serving import MmbertRuntime, PreparedText, Window
from morgott.models.retrieval import (
    EmbeddingResult,
    RetrievalResult,
    RetrievedExample,
    SparseResult,
)


def _window(index=0, text="hello"):
    return Window(
        index=index,
        char_start=0,
        char_end=len(text),
        input_ids=(101, index + 1, 102),
        attention_mask=(1, 1, 1),
    )


def _retrieval_manifest_for(evidence):
    return {
        "source": {
            "extension_sha256": evidence["source"]["hnsw_extension_sha256"],
            "sparse_identity_sha256": evidence["source"]["sparse_identity_sha256"],
        }
    }


class _Scorer:
    def __init__(self, scores):
        self.scores = tuple(scores)

    def prepare(self, text):
        windows = tuple(_window(index, text) for index in range(len(self.scores)))
        return PreparedText(
            normalized_text=text,
            token_count=len(self.scores),
            windows=windows,
        )

    def score(self, windows):
        del windows
        return self.scores


class _CapturingScorer(_Scorer):
    def prepare(self, text):
        self.text = text
        return super().prepare(text)


class _BlockingScorer(_Scorer):
    def __init__(self):
        super().__init__([0.1])
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def score(self, windows):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            self.release.wait(timeout=2)
        return super().score(windows)


class _Reviewer:
    def __init__(self, probability=0.1, status="ok"):
        self.probability = probability
        self.status = status
        self.input_channels = []
        self.texts = []
        self.query_texts = []

    async def review(self, text, *, input_channel, query_text=None):
        self.texts.append(text)
        self.query_texts.append(query_text)
        self.input_channels.append(input_channel)
        log_odds = (
            math.log(self.probability / (1 - self.probability))
            if self.status == "ok" and self.probability is not None
            else None
        )
        return WindowReview(
            status=self.status,
            probability=self.probability if self.status == "ok" else None,
            log_odds=log_odds,
            attempts=1,
            latency_ms=2.0,
            failure_code=None if self.status == "ok" else "http_429",
        )


class _ParallelReviewer(_Reviewer):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def review(self, text, *, input_channel, query_text=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return await super().review(
            text,
            input_channel=input_channel,
            query_text=query_text,
        )


class _ClosableReviewer(_Reviewer):
    def __init__(self):
        super().__init__()
        self.closed = False

    async def aclose(self):
        self.closed = True


class _SequenceReviewer:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.texts = []

    async def review(self, text, *, input_channel, query_text=None):
        self.texts.append(text)
        del input_channel, query_text
        probability = next(self.probabilities)
        return WindowReview(
            status="ok",
            probability=probability,
            log_odds=math.log(probability / (1 - probability)),
            attempts=1,
            latency_ms=1.0,
        )


class _FirstDecisiveReviewer:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    async def review(self, text, *, input_channel, query_text=None):
        del text, input_channel, query_text
        self.calls += 1
        decisive = self.calls == 1
        failed = decisive and self.status == "failed"
        probability = 0.9 if decisive and not failed else 0.1
        return WindowReview(
            status="failed" if failed else "ok",
            probability=None if failed else probability,
            log_odds=(None if failed else math.log(probability / (1 - probability))),
            attempts=1,
            latency_ms=1.0,
            failure_code="http_429" if failed else None,
        )


class _ForbiddenReviewer:
    async def review(self, text, *, input_channel, query_text=None):
        del text, input_channel, query_text
        raise AssertionError("remote review must not be called")


class RetrievalReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_examples_and_trace_reach_the_reviewer(self):
        examples = tuple(
            RetrievedExample(
                example_id=f"example-{label}-{index}",
                text=f"example text {label} {index}",
                text_sha256="a" * 64,
                label=label,
                input_channel="direct_user",
                source=f"source-{label}-{index}",
                group_id=f"group-{label}-{index}",
            )
            for index in range(2)
            for label in (0, 1)
        )

        class Base(_Reviewer):
            async def review_with_examples(self, text, *, input_channel, examples):
                self.packet = (text, input_channel, examples)
                return await self.review(text, input_channel=input_channel)

            async def aclose(self):
                self.closed = True

        class Embedder:
            async def embed(self, text):
                self.text = text
                return EmbeddingResult(
                    vector=[1.0] + [0.0] * 255,
                    elapsed_ms=6.0,
                    input_tokens=3,
                    cost_usd=0.000001,
                )

            async def aclose(self):
                self.closed = True

        class Engine:
            available = True
            manifest_sha256 = "b" * 64

            def sparse(self, text, input_channel):
                return SparseResult(
                    status="ok",
                    candidate_ids=(("a",), ("b",)),
                    elapsed_ms=4.0,
                    bundle_sha256=self.manifest_sha256,
                    query_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    input_channel=input_channel,
                )

            def retrieve(self, text, input_channel, embedding, sparse_result):
                del text, input_channel, embedding, sparse_result
                return RetrievalResult(
                    status="ok",
                    examples=examples,
                    fallback_reason=None,
                    dense_ms=2.0,
                    sparse_ms=4.0,
                    fusion_ms=1.0,
                )

        base = Base()
        embedder = Embedder()
        reviewer = RetrievalReviewer(base, Engine(), embedder)

        review = await reviewer.review(
            "review text",
            input_channel="direct_user",
            query_text="query text",
        )

        self.assertEqual(embedder.text, "query text")
        self.assertEqual(
            base.packet,
            (
                "review text",
                "direct_user",
                tuple((example.label, example.text) for example in examples),
            ),
        )
        self.assertEqual(review.retrieval.status, "ok")
        self.assertEqual(review.retrieval.embedding_input_tokens, 3)
        self.assertEqual(review.retrieval.selected_example_count, 4)
        self.assertEqual(
            review.retrieval.selected_packet_sha256,
            hashlib.sha256(
                json.dumps(
                    tuple(example.example_id for example in examples),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        await reviewer.aclose()
        self.assertTrue(base.closed)
        self.assertTrue(embedder.closed)

    async def test_embedding_failure_keeps_embedding_and_total_latency_separate(self):
        review_started = False
        clock_calls = 0

        class Reviewer(_Reviewer):
            async def review(self, text, *, input_channel, query_text=None):
                nonlocal review_started
                review_started = True
                return await super().review(
                    text,
                    input_channel=input_channel,
                    query_text=query_text,
                )

        def clock():
            nonlocal clock_calls
            clock_calls += 1
            if clock_calls == 1:
                return 0.0
            return 0.1 if review_started else 0.01

        class Embedder:
            async def embed(self, text):
                del text
                raise RuntimeError("synthetic failure")

        class Engine:
            available = True
            manifest_sha256 = "b" * 64

            def sparse(self, text, input_channel):
                del text, input_channel
                return None

        base = Reviewer()
        reviewer = RetrievalReviewer(base, Engine(), Embedder())

        with mock.patch(
            "morgott.models.cascade.time.perf_counter",
            side_effect=clock,
        ):
            review = await reviewer.review(
                "review text",
                input_channel="direct_user",
                query_text="query text",
            )

        self.assertEqual(review.retrieval.status, "embedding_failed")
        self.assertEqual(review.retrieval.fallback_reason, "embedding_failed")
        self.assertEqual(review.retrieval.embedding_ms, 10.0)
        self.assertEqual(review.retrieval.total_ms, 10.0)

    async def test_invalid_channel_is_rejected_before_embedding(self):
        class Embedder:
            calls = 0

            async def embed(self, text):
                del text
                self.calls += 1

        class Engine:
            available = True
            manifest_sha256 = "b" * 64

        embedder = Embedder()
        reviewer = RetrievalReviewer(_Reviewer(), Engine(), embedder)

        with self.assertRaisesRegex(ValueError, "input channel"):
            await reviewer.review(
                "review text",
                input_channel="invalid",
                query_text="query text",
            )

        self.assertEqual(embedder.calls, 0)

    async def test_empty_review_text_is_rejected_before_embedding(self):
        class Embedder:
            calls = 0

            async def embed(self, text):
                del text
                self.calls += 1

        class Engine:
            available = True
            manifest_sha256 = "b" * 64

        embedder = Embedder()
        reviewer = RetrievalReviewer(_Reviewer(), Engine(), embedder)

        with self.assertRaisesRegex(ValueError, "review text"):
            await reviewer.review(
                "",
                input_channel="direct_user",
                query_text="query text",
            )

        self.assertEqual(embedder.calls, 0)

    async def test_embedding_failure_reports_the_prompt_actually_sent(self):
        class Embedder:
            async def embed(self, text):
                del text
                raise RuntimeError("synthetic failure")

        class Engine:
            available = True
            manifest_sha256 = "b" * 64

            def sparse(self, text, input_channel):
                del text, input_channel
                return None

        reviewer = RetrievalReviewer(_Reviewer(), Engine(), Embedder())
        scanner = CascadeScanner(scorer=_Scorer([0.5]), reviewer=reviewer)

        result = await scanner.assess_text(
            "review text",
            input_channel="direct_user",
        )

        self.assertEqual(result.prompt_sha256, PROMPT_SHA256)
        self.assertEqual(result.retrieval_fallback_reason, "embedding_failed")


class CascadeScannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_untrusted_content_reaches_the_lower_review_floor(self):
        reviewer = _Reviewer(probability=0.1)
        scanner = CascadeScanner(
            scorer=_Scorer([0.1]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "tool-return text",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.reason), ("pass", "deepseek_clear")
        )
        self.assertEqual(result.deepseek_calls, 1)

    async def test_all_low_windows_return_a_complete_shadow_pass(self):
        scanner = CascadeScanner(
            scorer=_Scorer([0.1, 0.199]),
            reviewer=None,
        )

        result = await scanner.assess_text(
            "ordinary text",
            input_channel="direct_user",
        )

        self.assertEqual((result.decision, result.advisory_only), ("allow", True))
        self.assertEqual((result.advisory_route, result.reason), ("pass", "mmbert_low"))
        self.assertTrue(result.complete)
        self.assertEqual((result.low_windows, result.middle_windows), (2, 0))
        self.assertEqual((result.high_windows, result.deepseek_calls), (0, 0))
        self.assertEqual(result.window_count, 2)

    async def test_untrusted_multi_window_full_context_flag_short_circuits(self):
        reviewer = _Reviewer(probability=0.9)
        scanner = CascadeScanner(
            scorer=_Scorer([0.01, 0.01]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "distributed instruction",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.reason, result.deepseek_calls),
            ("restrict", "deepseek_full_context_flag", 1),
        )
        self.assertEqual(result.reviewed_windows[0].index, FULL_CONTEXT_REVIEW_INDEX)
        self.assertEqual(reviewer.texts, ["distributed instruction"])

    async def test_full_context_retrieval_queries_the_highest_scoring_window(self):
        text = "first second"

        class SegmentScorer(_Scorer):
            def prepare(self, value):
                return PreparedText(
                    normalized_text=value,
                    token_count=2,
                    windows=(
                        Window(0, 0, 5, (101, 1, 102), (1, 1, 1)),
                        Window(1, 6, 12, (101, 2, 102), (1, 1, 1)),
                    ),
                )

        reviewer = _Reviewer(probability=0.9)
        scanner = CascadeScanner(
            scorer=SegmentScorer([0.01, 0.5]),
            reviewer=reviewer,
        )

        await scanner.assess_text(text, input_channel="untrusted_content")

        self.assertEqual(reviewer.texts, [text])
        self.assertEqual(reviewer.query_texts, ["second"])

    async def test_full_context_reviewer_uses_the_promoted_boundary(self):
        for probability, expected in (
            (math.nextafter(0.5, 0.0), ("pass", "deepseek_full_context_clear")),
            (0.5, ("restrict", "deepseek_full_context_flag")),
        ):
            with self.subTest(probability=probability):
                scanner = CascadeScanner(
                    scorer=_Scorer([0.01, 0.01]),
                    reviewer=_Reviewer(probability=probability),
                )

                result = await scanner.assess_text(
                    "long untrusted content",
                    input_channel="untrusted_content",
                )

                self.assertEqual((result.advisory_route, result.reason), expected)

    async def test_untrusted_multi_window_full_clear_passes_all_low_input(self):
        reviewer = _Reviewer(probability=0.1)
        scanner = CascadeScanner(
            scorer=_Scorer([0.01, 0.01]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "ordinary long content",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.reason, result.deepseek_calls),
            ("pass", "deepseek_full_context_clear", 1),
        )
        self.assertEqual(
            [review.index for review in result.reviewed_windows],
            [FULL_CONTEXT_REVIEW_INDEX],
        )

    async def test_untrusted_full_clear_falls_back_to_window_review(self):
        reviewer = _SequenceReviewer([0.1, 0.1, 0.9])
        scanner = CascadeScanner(
            scorer=_Scorer([0.5, 0.5]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "ambiguous long content",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.reason, result.deepseek_calls),
            ("restrict", "deepseek_flag", 3),
        )
        self.assertEqual(
            [review.index for review in result.reviewed_windows],
            [FULL_CONTEXT_REVIEW_INDEX, 0, 1],
        )

    async def test_untrusted_full_context_failure_restricts_incompletely(self):
        scanner = CascadeScanner(
            scorer=_Scorer([0.01, 0.01]),
            reviewer=_Reviewer(status="failed"),
        )

        result = await scanner.assess_text(
            "ordinary long content",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.complete, result.reason),
            ("restrict", False, "deepseek_failed"),
        )
        self.assertEqual(result.deepseek_failures, 1)
        self.assertEqual(result.reviewed_windows[0].index, FULL_CONTEXT_REVIEW_INDEX)

    async def test_any_high_window_returns_a_complete_shadow_restrict(self):
        for high_index in (0, 1, 2):
            with self.subTest(high_index=high_index):
                scores = [0.1, 0.5, 0.1]
                scores[high_index] = 0.9999
                scanner = CascadeScanner(
                    scorer=_Scorer(scores),
                    reviewer=None,
                )

                result = await scanner.assess_text(
                    "mixed text",
                    input_channel="untrusted_content",
                )

                self.assertEqual(
                    (result.decision, result.advisory_route, result.reason),
                    ("allow", "restrict", "mmbert_high"),
                )
                self.assertTrue(result.complete)
                self.assertEqual(result.high_windows, 1)
                self.assertEqual(result.deepseek_calls, 0)

    async def test_all_clear_middle_windows_return_a_complete_shadow_pass(self):
        reviewer = _Reviewer(probability=0.1)
        scanner = CascadeScanner(
            scorer=_Scorer([0.2, 0.5]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "ambiguous text",
            input_channel="direct_user",
        )

        self.assertEqual(
            (result.advisory_route, result.reason),
            ("pass", "deepseek_clear"),
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.deepseek_calls, 2)
        self.assertEqual(result.deepseek_failures, 0)
        self.assertEqual(reviewer.input_channels, ["direct_user", "direct_user"])
        self.assertEqual(
            [window.index for window in result.reviewed_windows],
            [0, 1],
        )

    async def test_chunked_input_is_assessed_as_one_complete_artifact(self):
        async def chunks():
            yield "first"
            yield " second"

        scorer = _CapturingScorer([0.1])
        scanner = CascadeScanner(
            scorer=scorer,
            reviewer=None,
        )

        result = await scanner.assess_chunks(
            chunks(),
            input_channel="untrusted_content",
        )

        self.assertEqual(scorer.text, "first second")
        self.assertEqual(
            result.artifact_sha256,
            "92088ec140fc553e4b1ede202edccb65a807bbf8a38d765a3ad38013c0f13688",
        )

    async def test_chunked_input_preserves_newlines_before_assessment(self):
        async def chunks():
            yield "first\r"
            yield "\nsecond"

        scorer = _CapturingScorer([0.1])
        scanner = CascadeScanner(scorer=scorer, reviewer=None)

        await scanner.assess_chunks(
            chunks(),
            input_channel="untrusted_content",
        )

        self.assertEqual(scorer.text, "first\r\nsecond")

    async def test_invalid_remote_result_fails_safe(self):
        scanner = CascadeScanner(
            scorer=_Scorer([0.5]),
            reviewer=_Reviewer(probability=None),
        )

        result = await scanner.assess_text(
            "ambiguous text",
            input_channel="direct_user",
        )

        self.assertEqual(
            (result.advisory_route, result.complete, result.reason),
            ("restrict", False, "deepseek_failed"),
        )
        self.assertEqual(result.deepseek_failures, 1)
        self.assertEqual(result.reviewed_windows[0].failure_code, "invalid_review")

    async def test_flag_threshold_and_remote_failure_both_restrict(self):
        cases = (
            (_Reviewer(probability=0.9), True, "deepseek_flag"),
            (_Reviewer(status="failed"), False, "deepseek_failed"),
        )
        for reviewer, complete, reason in cases:
            with self.subTest(reason=reason):
                scanner = CascadeScanner(
                    scorer=_Scorer([0.5]),
                    reviewer=reviewer,
                )

                result = await scanner.assess_text(
                    "ambiguous text",
                    input_channel="direct_user",
                )

                self.assertEqual(
                    (result.advisory_route, result.complete, result.reason),
                    ("restrict", complete, reason),
                )

    async def test_one_flagged_middle_window_restricts_mixed_reviews(self):
        scanner = CascadeScanner(
            scorer=_Scorer([0.2, 0.5, 0.8]),
            reviewer=_SequenceReviewer([0.1, 0.9, 0.2]),
        )

        result = await scanner.assess_text(
            "three ambiguous windows",
            input_channel="direct_user",
        )

        self.assertEqual(
            (result.advisory_route, result.reason, result.deepseek_calls),
            ("restrict", "deepseek_flag", 3),
        )

    async def test_transport_chunks_are_joined_before_unicode_normalization(self):
        async def chunks():
            yield "Ａ\t"
            yield "\nB"

        encoding = type(
            "Encoding",
            (),
            {
                "ids": [101, 1, 2, 102],
                "attention_mask": [1, 1, 1, 1],
                "offsets": [(0, 0), (0, 1), (2, 3), (0, 0)],
                "overflowing": [],
            },
        )()

        class Tokenizer:
            def enable_truncation(self, **kwargs):
                del kwargs

            def encode(self, text):
                self.text = text
                return encoding

        class Session:
            def __call__(self, inputs):
                del inputs
                return {"logit": [[-3.0]]}

        tokenizer = Tokenizer()
        scanner = CascadeScanner(
            scorer=MmbertRuntime(tokenizer=tokenizer, session=Session()),
            reviewer=None,
        )

        await scanner.assess_chunks(
            chunks(),
            input_channel="untrusted_content",
        )

        self.assertEqual(tokenizer.text, "a b")

    async def test_remote_window_cap_fails_safe_before_provider_calls(self):
        scanner = CascadeScanner(
            scorer=_Scorer([0.5] * 129),
            reviewer=_ForbiddenReviewer(),
        )

        result = await scanner.assess_text(
            "large ambiguous artifact",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.complete, result.reason),
            ("restrict", False, "deepseek_window_limit"),
        )
        self.assertEqual(result.deepseek_calls, 0)

    async def test_window_cap_restricts_all_low_untrusted_multiwindow_text(self):
        # With a reviewer configured, multi-window untrusted text goes to
        # full-context review even when every window scores low, so text over
        # the window cap fails safe instead of passing locally.
        scanner = CascadeScanner(
            scorer=_Scorer([0.01] * 129),
            reviewer=_ForbiddenReviewer(),
        )

        result = await scanner.assess_text(
            "large low-scoring artifact",
            input_channel="untrusted_content",
        )

        self.assertEqual(
            (result.advisory_route, result.complete, result.reason),
            ("restrict", False, "deepseek_window_limit"),
        )
        self.assertEqual(result.deepseek_calls, 0)

    async def test_middle_windows_are_reviewed_concurrently(self):
        reviewer = _ParallelReviewer()
        scanner = CascadeScanner(
            scorer=_Scorer([0.2, 0.5, 0.8]),
            reviewer=reviewer,
        )

        result = await scanner.assess_text(
            "ambiguous text",
            input_channel="direct_user",
        )

        self.assertEqual(result.advisory_route, "pass")
        self.assertGreater(reviewer.max_active, 1)

    async def test_decisive_remote_batch_skips_later_provider_calls(self):
        for status, complete, reason in (
            ("ok", True, "deepseek_flag"),
            ("failed", False, "deepseek_failed"),
        ):
            with self.subTest(reason=reason):
                reviewer = _FirstDecisiveReviewer(status)
                scanner = CascadeScanner(
                    scorer=_Scorer([0.5] * (REMOTE_CONCURRENCY + 1)),
                    reviewer=reviewer,
                )

                result = await scanner.assess_text(
                    "large ambiguous artifact",
                    input_channel="direct_user",
                )

                self.assertEqual(
                    (result.advisory_route, result.complete, result.reason),
                    ("restrict", complete, reason),
                )
                self.assertEqual(reviewer.calls, REMOTE_CONCURRENCY)
                self.assertEqual(len(result.reviewed_windows), REMOTE_CONCURRENCY)

    async def test_chunk_cancellation_closes_the_temporary_file(self):
        async def chunks():
            yield "partial"
            raise asyncio.CancelledError

        temporary = io.StringIO()
        scanner = CascadeScanner(
            scorer=_Scorer([0.1]),
            reviewer=None,
        )
        with (
            mock.patch(
                "morgott.models.cascade.tempfile.TemporaryFile",
                return_value=temporary,
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await scanner.assess_chunks(
                chunks(),
                input_channel="untrusted_content",
            )

        self.assertTrue(temporary.closed)

    async def test_cancelled_local_inference_remains_serialized_until_it_stops(self):
        scorer = _BlockingScorer()
        scanner = CascadeScanner(
            scorer=scorer,
            reviewer=None,
        )
        first = asyncio.create_task(
            scanner.assess_text("first", input_channel="direct_user")
        )
        self.assertTrue(await asyncio.to_thread(scorer.started.wait, 1))
        first.cancel()
        second = asyncio.create_task(
            scanner.assess_text("second", input_channel="direct_user")
        )
        await asyncio.sleep(0.01)
        self.assertEqual(scorer.calls, 1)

        scorer.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second

    async def test_assessment_never_contains_the_raw_input(self):
        text = "private marker that must not appear"
        scanner = CascadeScanner(
            scorer=_Scorer([0.1]),
            reviewer=None,
        )

        result = await scanner.assess_text(text, input_channel="direct_user")

        self.assertNotIn(text, json.dumps(asdict(result)))

    def test_cli_reads_a_file_and_emits_the_shadow_assessment(self):
        reviewer = _ClosableReviewer()
        scanner = CascadeScanner(
            scorer=_Scorer([0.1]),
            reviewer=reviewer,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("ordinary text", encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(
                    CascadeScanner,
                    "from_artifacts",
                    return_value=scanner,
                ),
                redirect_stdout(output),
            ):
                main(
                    [
                        "cascade",
                        str(path),
                        "--input-channel",
                        "direct_user",
                    ]
                )

        result = json.loads(output.getvalue())
        self.assertEqual(result["decision"], "allow")
        self.assertEqual(result["advisory_route"], "pass")
        self.assertTrue(reviewer.closed)

    def test_production_constructor_requires_registered_retrieval(self):
        scorer = _Scorer([0.1])
        reviewer = _ClosableReviewer()
        engine = mock.Mock(available=True, manifest_sha256="b" * 64)
        embedder = mock.AsyncMock()
        with (
            mock.patch(
                "morgott.models.cascade._verify_registered_policy",
                return_value=(
                    "a" * 64,
                    Path("bundle/manifest.json"),
                    "b" * 64,
                ),
            ),
            mock.patch(
                "morgott.models.cascade.MmbertRuntime.from_artifacts",
                return_value=scorer,
            ) as load_scorer,
            mock.patch(
                "morgott.models.cascade.DeepSeekReviewer.from_env",
                return_value=reviewer,
            ) as load_reviewer,
            mock.patch(
                "morgott.models.cascade.RetrievalEngine",
                return_value=engine,
            ) as load_engine,
            mock.patch(
                "morgott.models.cascade.OpenRouterEmbedder.from_env",
                return_value=embedder,
            ) as load_embedder,
        ):
            scanner = CascadeScanner.from_artifacts(
                manifest_path=Path("model-artifacts.json"),
                inference_precision="auto",
            )

        self.assertIsInstance(scanner._reviewer, RetrievalReviewer)
        self.assertEqual(scanner.policy_sha256, "a" * 64)
        self.assertEqual(scanner.retrieval_manifest_sha256, "b" * 64)
        self.assertTrue(scanner.retrieval_enabled)
        load_scorer.assert_called_once_with(
            Path("model-artifacts.json"),
            inference_precision="auto",
        )
        load_reviewer.assert_called_once()
        load_engine.assert_called_once_with(Path("bundle"), "b" * 64)
        load_embedder.assert_called_once()

    def test_retrieval_parity_must_match_the_registered_bundle(self):
        policy = json.loads(
            Path(
                "artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/promotion-retrieval.json"
            ).read_text(encoding="utf-8")
        )
        evidence = json.loads(
            Path(policy["evidence"]["path"]).read_text(encoding="utf-8")
        )
        retrieval_manifest = _retrieval_manifest_for(evidence)
        evidence["source"]["sparse_identity_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "parity evidence is invalid"):
            _verify_retrieval_parity(
                evidence,
                retrieval_manifest,
                evidence["source"]["serving_manifest_sha256"],
            )

    def test_production_constructor_rejects_registry_policy_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = json.loads(
                Path(
                    "artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/promotion-retrieval.json"
                ).read_text(encoding="utf-8")
            )
            evidence_source = Path(policy["evidence"]["path"])
            evidence = json.loads(evidence_source.read_text(encoding="utf-8"))
            evidence_path = root / "retrieval-parity.json"
            retrieval_path = root / "retrieval-manifest.json"
            retrieval_path.write_text(
                json.dumps(_retrieval_manifest_for(evidence)),
                encoding="utf-8",
            )
            evidence["source"]["serving_manifest_sha256"] = hashlib.sha256(
                retrieval_path.read_bytes()
            ).hexdigest()
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            policy["evidence"] = {
                "path": evidence_path.name,
                "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            }
            policy["runtime_contract"]["profile"] = "wrong-profile"
            policy_path = root / "promotion.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            manifest_path = root / "model-artifacts.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "advisory_only": True,
                        "models": {
                            "mmbert-lora-full-ctx1024-u17000-s42": {
                                "serving": {
                                    "cascade_policy": {
                                        "path": "promotion.json",
                                        "sha256": hashlib.sha256(
                                            policy_path.read_bytes()
                                        ).hexdigest(),
                                    },
                                    "retrieval": {
                                        "format": "morgott-lineage-hybrid-v1",
                                        "manifest": {
                                            "path": "retrieval-manifest.json",
                                            "sha256": hashlib.sha256(
                                                retrieval_path.read_bytes()
                                            ).hexdigest(),
                                        },
                                    },
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "morgott.models.cascade.MmbertRuntime.from_artifacts"
                ) as load_scorer,
                self.assertRaisesRegex(ValueError, "differs from maintained code"),
            ):
                CascadeScanner.from_artifacts(manifest_path=manifest_path)

            load_scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
