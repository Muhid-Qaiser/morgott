import asyncio
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
from morgott.models.cascade import FULL_CONTEXT_REVIEW_INDEX, CascadeScanner
from morgott.models.deepseek_nooa import REMOTE_CONCURRENCY, WindowReview
from morgott.models.mmbert.serving import MmbertRuntime, PreparedText, Window


def _window(index=0, text="hello"):
    return Window(
        index=index,
        char_start=0,
        char_end=len(text),
        input_ids=(101, index + 1, 102),
        attention_mask=(1, 1, 1),
    )


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

    async def review(self, text, *, input_channel):
        self.texts.append(text)
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

    async def review(self, text, *, input_channel):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return await super().review(text, input_channel=input_channel)


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

    async def review(self, text, *, input_channel):
        self.texts.append(text)
        del input_channel
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

    async def review(self, text, *, input_channel):
        del text, input_channel
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
    async def review(self, text, *, input_channel):
        del text, input_channel
        raise AssertionError("remote review must not be called")


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
                scores[high_index] = 0.99999
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


if __name__ == "__main__":
    unittest.main()
