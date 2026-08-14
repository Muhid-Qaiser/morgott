import copy
import importlib.util
import json
import math
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from morgott.models.deepseek_nooa import (
    LITELLM_MODEL,
    MODEL,
    PROMPT,
    PROMPT_SHA256,
    PROVIDER,
    REMOTE_CONCURRENCY,
    REQUEST_SHA256,
    DeepSeekReviewer,
    refuse_nooa_tracing,
)


class _Response:
    def __init__(self, raw_response):
        self.raw_response = raw_response
        self.usage = {"prompt_tokens": 20, "completion_tokens": 5}


class _Client:
    def __init__(self, response):
        self.response = response

    async def acall(self, messages, **kwargs):
        del messages, kwargs
        return self.response


class _ClosingClient(_Client):
    def __init__(self, response):
        super().__init__(response)
        self.closed = False

    async def aclose(self):
        self.closed = True


class _SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def acall(self, messages, **kwargs):
        del messages, kwargs
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class _HttpError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = type(
            "Response",
            (),
            {"status_code": status_code, "headers": headers or {}},
        )()


def _raw_response(verdict=1, logprob_0=math.log(0.1), logprob_1=math.log(0.9)):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": f'{{"subversion":{verdict}}}'},
                "logprobs": {
                    "content": [
                        {
                            "token": str(verdict),
                            "bytes": [48 + verdict],
                            "logprob": logprob_1 if verdict else logprob_0,
                            "top_logprobs": [
                                {
                                    "token": "0",
                                    "bytes": [48],
                                    "logprob": logprob_0,
                                },
                                {
                                    "token": "1",
                                    "bytes": [49],
                                    "logprob": logprob_1,
                                },
                            ],
                        }
                    ]
                },
            }
        ]
    }


def _wire_response():
    return {
        "id": "local-contract-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"subversion":1}',
                },
                "logprobs": _raw_response()["choices"][0]["logprobs"],
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
    }


class DeepSeekReviewerTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("nooa"),
        "NOOA cascade extra is not installed",
    )
    def test_programmatic_nooa_tracing_is_refused(self):
        with (
            mock.patch("nooa.tracing._enabled", True),
            self.assertRaisesRegex(RuntimeError, "tracing must be disabled"),
        ):
            refuse_nooa_tracing()

    @unittest.skipUnless(
        importlib.util.find_spec("nooa"),
        "NOOA cascade extra is not installed",
    )
    def test_production_reviewer_refuses_enabled_nooa_tracing(self):
        with (
            mock.patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "not-a-secret"},
                clear=True,
            ),
            mock.patch("nooa.tracing._enabled", True),
            mock.patch("nooa.unifiedllm.CompletionClient"),
            self.assertRaisesRegex(RuntimeError, "tracing must be disabled"),
        ):
            DeepSeekReviewer.from_env()

    @unittest.skipUnless(
        importlib.util.find_spec("nooa"),
        "NOOA cascade extra is not installed",
    )
    def test_production_reviewer_suppresses_litellm_error_banners(self):
        import litellm

        previous = litellm.suppress_debug_info
        litellm.suppress_debug_info = False
        try:
            with (
                mock.patch.dict(
                    "os.environ",
                    {"OPENROUTER_API_KEY": "not-a-secret"},
                    clear=True,
                ),
                mock.patch("morgott.models.deepseek_nooa.refuse_nooa_tracing"),
                mock.patch("nooa.unifiedllm.CompletionClient"),
            ):
                DeepSeekReviewer.from_env()

            self.assertTrue(litellm.suppress_debug_info)
        finally:
            litellm.suppress_debug_info = previous

    async def test_reviewer_closes_its_http_client(self):
        client = _ClosingClient(_Response(_raw_response()))
        reviewer = DeepSeekReviewer(client)

        await reviewer.aclose()

        self.assertTrue(client.closed)

    async def test_review_rejects_untyped_input_channel(self):
        reviewer = DeepSeekReviewer(_Client(_Response(_raw_response())))

        with self.assertRaisesRegex(ValueError, "trusted runtime metadata"):
            await reviewer.review("classify me", input_channel="model_output")

    async def test_valid_binary_logprobs_return_a_typed_review(self):
        reviewer = DeepSeekReviewer(_Client(_Response(_raw_response())))

        review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(review.status, "ok")
        self.assertEqual(review.attempts, 1)
        self.assertAlmostEqual(review.probability, 0.9)
        self.assertAlmostEqual(review.log_odds, math.log(9))
        self.assertEqual((review.input_tokens, review.output_tokens), (20, 5))
        self.assertIsNone(review.failure_code)

    async def test_invalid_response_is_retried_within_the_three_call_cap(self):
        invalid = _Response({"choices": []})
        valid = _Response(_raw_response(verdict=0))
        reviewer = DeepSeekReviewer(_SequenceClient([invalid, valid]))

        review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(review.status, "ok")
        self.assertEqual(review.attempts, 2)

    async def test_retryable_http_failure_honors_retry_after(self):
        reviewer = DeepSeekReviewer(
            _SequenceClient(
                [
                    _HttpError(429, {"Retry-After": "0"}),
                    _Response(_raw_response()),
                ]
            ),
        )

        with mock.patch(
            "morgott.models.deepseek_nooa.asyncio.sleep",
            new_callable=mock.AsyncMock,
        ) as sleep:
            review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(review.status, "ok")
        self.assertEqual(review.attempts, 2)
        sleep.assert_awaited_once_with(0.0)

    async def test_rate_limit_without_retry_after_uses_longer_backoff(self):
        reviewer = DeepSeekReviewer(
            _SequenceClient([_HttpError(429), _Response(_raw_response())])
        )

        with (
            mock.patch("morgott.models.deepseek_nooa.random.random", return_value=0.5),
            mock.patch(
                "morgott.models.deepseek_nooa.asyncio.sleep",
                new_callable=mock.AsyncMock,
            ) as sleep,
        ):
            review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual((review.status, review.attempts), ("ok", 2))
        sleep.assert_awaited_once_with(5.0)

    async def test_retry_after_delay_is_finite_and_bounded(self):
        for value in ("inf", "999999999999"):
            with self.subTest(value=value):
                reviewer = DeepSeekReviewer(
                    _SequenceClient(
                        [
                            _HttpError(429, {"Retry-After": value}),
                            _Response(_raw_response()),
                        ]
                    ),
                )
                with (
                    mock.patch(
                        "morgott.models.deepseek_nooa.random.random",
                        return_value=0.5,
                    ),
                    mock.patch(
                        "morgott.models.deepseek_nooa.asyncio.sleep",
                        new_callable=mock.AsyncMock,
                    ) as sleep,
                ):
                    review = await reviewer.review(
                        "classify me",
                        input_channel="direct_user",
                    )

                self.assertEqual(review.status, "ok")
                delay = sleep.await_args.args[0]
                self.assertTrue(math.isfinite(delay))
                self.assertLessEqual(delay, 15.0)

    async def test_client_side_invalid_response_error_is_retried(self):
        reviewer = DeepSeekReviewer(
            _SequenceClient(
                [
                    ValueError("malformed provider response"),
                    _Response(_raw_response()),
                ]
            ),
        )

        with mock.patch(
            "morgott.models.deepseek_nooa.asyncio.sleep",
            new_callable=mock.AsyncMock,
        ):
            review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual((review.status, review.attempts), ("ok", 2))

    async def test_non_retryable_http_failure_returns_without_retrying(self):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                reviewer = DeepSeekReviewer(_SequenceClient([_HttpError(status)]))

                review = await reviewer.review(
                    "classify me",
                    input_channel="direct_user",
                )

                self.assertEqual(
                    (review.status, review.attempts, review.failure_code),
                    ("failed", 1, f"http_{status}"),
                )

    async def test_provider_error_details_are_not_retained(self):
        secret = "sk-private-value"
        reviewer = DeepSeekReviewer(
            _SequenceClient([RuntimeError(f"provider exposed {secret}")])
        )

        review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(review.failure_code, "transport_error")
        self.assertNotIn(secret, repr(review))

    async def test_invalid_provider_output_exhausts_at_three_calls(self):
        invalid = _Response({"choices": []})
        reviewer = DeepSeekReviewer(
            _SequenceClient([invalid, invalid, invalid]),
        )

        review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(
            (review.status, review.attempts, review.failure_code),
            ("failed", 3, "invalid_response"),
        )

    async def test_missing_raw_response_exhausts_as_an_invalid_response(self):
        reviewer = DeepSeekReviewer(_Client(object()))

        review = await reviewer.review("classify me", input_channel="direct_user")

        self.assertEqual(
            (review.status, review.attempts, review.failure_code),
            ("failed", 3, "invalid_response"),
        )

    async def test_strict_parser_rejects_schema_and_logprob_ambiguity(self):
        extra_field = _raw_response()
        extra_field["choices"][0]["message"]["content"] = (
            '{"subversion":1,"explanation":"no"}'
        )
        disagreement = _raw_response()
        disagreement["choices"][0]["message"]["content"] = '{"subversion":0}'
        missing_class = _raw_response()
        missing_class["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [
            {
                "token": "1",
                "bytes": [49],
                "logprob": math.log(0.9),
            }
        ]
        duplicate_class = _raw_response()
        alternatives = duplicate_class["choices"][0]["logprobs"]["content"][0][
            "top_logprobs"
        ]
        alternatives.append(copy.deepcopy(alternatives[0]))
        nonfinite = _raw_response()
        nonfinite["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0][
            "logprob"
        ] = math.nan
        inconsistent_decision = _raw_response()
        alternatives = inconsistent_decision["choices"][0]["logprobs"]["content"][0][
            "top_logprobs"
        ]
        alternatives[0]["logprob"] = math.log(0.9)
        alternatives[1]["logprob"] = math.log(0.1)
        multiple_decisions = _raw_response()
        multiple_decisions["choices"][0]["logprobs"]["content"].append(
            copy.deepcopy(multiple_decisions["choices"][0]["logprobs"]["content"][0])
        )
        missing_bytes = _raw_response()
        decision = missing_bytes["choices"][0]["logprobs"]["content"][0]
        decision.pop("bytes")
        for alternative in decision["top_logprobs"]:
            alternative.pop("bytes")
        whitespace_bytes = _raw_response()
        decision = whitespace_bytes["choices"][0]["logprobs"]["content"][0]
        decision["bytes"] = [32, 49]
        decision["top_logprobs"][0]["bytes"] = [32, 48]
        decision["top_logprobs"][1]["bytes"] = [32, 49]
        for payload in (
            extra_field,
            disagreement,
            missing_class,
            duplicate_class,
            nonfinite,
            inconsistent_decision,
            multiple_decisions,
            missing_bytes,
            whitespace_bytes,
        ):
            with self.subTest(payload=payload):
                response = _Response(payload)
                reviewer = DeepSeekReviewer(
                    _SequenceClient([response, response, response])
                )

                review = await reviewer.review(
                    "classify me",
                    input_channel="direct_user",
                )

                self.assertEqual(
                    (review.status, review.failure_code),
                    ("failed", "invalid_response"),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("nooa"),
        "NOOA cascade extra is not installed",
    )
    async def test_real_nooa_client_preserves_the_complete_wire_contract(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["body"] = json.loads(self.rfile.read(length))
                body = json.dumps(_wire_response()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        from nooa.unifiedllm import CompletionClient, HttpConfig, RetryConfig

        client = CompletionClient(
            model=LITELLM_MODEL,
            api_key="not-a-secret",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            num_retries=0,
            retry_config=RetryConfig(
                max_retries=0,
                rate_limit_extra_retries=0,
            ),
            http_config=HttpConfig(
                max_connections=1,
                max_keepalive_connections=1,
            ),
        )
        try:
            review = await DeepSeekReviewer(client).review(
                "classify me",
                input_channel="untrusted_content",
            )
        finally:
            await client.aclose()
            server.shutdown()
            thread.join()
            server.server_close()

        self.assertEqual(review.status, "ok")
        self.assertEqual(MODEL, "deepseek/deepseek-v4-flash-0731")
        self.assertEqual(PROVIDER, "cloudflare")
        self.assertEqual(REMOTE_CONCURRENCY, 4)
        self.assertEqual(
            REQUEST_SHA256,
            "8138ecb7533351edfbd82194d591e2e491082443b6f13e61c7bf7f996568ce91",
        )
        self.assertEqual(
            PROMPT_SHA256,
            "6793cd3df00ea49c6da801692ef94b8200b212056fba27d298830186843b99a1",
        )
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(
            captured["body"],
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": PROMPT.format(input_channel="untrusted_content"),
                    },
                    {"role": "user", "content": "classify me"},
                ],
                "temperature": 0,
                "max_tokens": 16,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "instruction_subversion",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "subversion": {
                                    "type": "integer",
                                    "enum": [0, 1],
                                }
                            },
                            "required": ["subversion"],
                            "additionalProperties": False,
                        },
                    },
                },
                "logprobs": True,
                "top_logprobs": 20,
                "reasoning": {"enabled": False, "exclude": True},
                "provider": {
                    "order": [PROVIDER],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
