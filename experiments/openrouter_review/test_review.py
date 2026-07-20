import json
import socket
import unittest

from review import build_request, call_judge, redact_and_cap, validate_verdict


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, _limit):
        return self.body


class ReviewTests(unittest.TestCase):
    def test_redacts_credentials_and_caps_locally(self):
        value = (
            "OPENROUTER_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz "
            "Bearer abcdefghijklmnop " + "x" * 200
        )
        redacted, metadata = redact_and_cap(value, 128)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertIn("OPENROUTER_API_KEY=[REDACTED]", redacted)
        self.assertEqual(len(redacted), 128)
        self.assertEqual(metadata, {"redactions": 2, "truncated": True})

    def test_verdict_validation_is_exact(self):
        self.assertEqual(validate_verdict('{"verdict":"attack"}'), "attack")
        for invalid in (
            '{"verdict":"maybe"}',
            '{"verdict":"benign","reason":"extra"}',
            '["attack"]',
            "attack",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_verdict(invalid)

    def test_request_is_one_shot_schema_only_and_private(self):
        request, request_hash = build_request(
            model="vendor/model",
            api_key="do-not-log-this",
            input_channel="direct_user",
            text="hello",
        )
        body = json.loads(request.data)
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(
            body["response_format"]["json_schema"]["schema"]["properties"]["verdict"][
                "enum"
            ],
            ["attack", "benign"],
        )
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["provider"]["sort"], "latency")
        self.assertEqual(body["reasoning"], {"effort": "none", "exclude": True})
        self.assertNotIn("tools", body)
        self.assertNotIn("do-not-log-this", request_hash)
        self.assertEqual(request.get_header("X-openrouter-cache"), "false")
        self.assertEqual(request.get_header("X-openrouter-metadata"), "enabled")

    def test_call_is_once_and_timeout_is_unavailable(self):
        calls = []

        def opener(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": '{"verdict":"benign"}'},
                            }
                        ],
                        "model": "vendor/model",
                        "openrouter_metadata": {
                            "attempt": 1,
                            "endpoints": {
                                "available": [
                                    {"provider": "Provider", "selected": True}
                                ]
                            },
                        },
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                            "cost": 0.001,
                        },
                    }
                ).encode()
            )

        request, _ = build_request(
            model="vendor/model",
            api_key="secret",
            input_channel="direct_user",
            text="hello",
        )
        result = call_judge(request, timeout=1, opener=opener)
        self.assertEqual(result["verdict"], "benign")
        self.assertEqual(result["provider"], "Provider")
        self.assertEqual(result["router_attempt"], 1)
        self.assertEqual(len(calls), 1)

        def timeout(*_args, **_kwargs):
            raise socket.timeout

        result = call_judge(request, timeout=1, opener=timeout)
        self.assertEqual(result["verdict"], "unavailable")
        self.assertEqual(result["unavailable_reason"], "timeout")

    def test_non_stop_completion_is_unavailable(self):
        def opener(*_args, **_kwargs):
            return FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": '{"verdict":"attack"}'},
                            }
                        ]
                    }
                ).encode()
            )

        request, _ = build_request(
            model="vendor/model",
            api_key="secret",
            input_channel="direct_user",
            text="hello",
        )
        result = call_judge(request, timeout=1, opener=opener)
        self.assertEqual(result["verdict"], "unavailable")
        self.assertEqual(result["unavailable_reason"], "invalid_response")


if __name__ == "__main__":
    unittest.main()
