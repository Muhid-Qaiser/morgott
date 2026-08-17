import unittest

from experiments.pipeline_benchmark.gpt_oss_native import (
    MODEL,
    POLICY,
    PROVIDER,
    _parse,
)


class GptOssNativeTests(unittest.TestCase):
    def test_exact_policy_output(self):
        payload = {
            "model": MODEL,
            "provider": PROVIDER,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"violation":1,"policy_category":"INSTRUCTION_SUBVERSION","rationale":"The content overrides trusted instructions."}'
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.1},
        }
        self.assertEqual(_parse(payload)[0], 1)
        payload["choices"][0]["message"]["content"] = (
            '{"violation":1,"policy_category":null,"rationale":"bad"}'
        )
        with self.assertRaises(ValueError):
            _parse(payload)
        self.assertGreater(len(POLICY.split()), 300)


if __name__ == "__main__":
    unittest.main()
