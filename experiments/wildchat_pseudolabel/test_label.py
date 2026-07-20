import json
import tempfile
import unittest
from pathlib import Path

from label import (
    accepted_rows,
    audit_sample_ids,
    build_request,
    call_once,
    load_sample,
    load_journal,
    parse_completion,
    report_path,
    validate_label,
)


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit):
        return self.body


def _completion(label="benign", confidence="high", toxicity="not_toxic"):
    content = json.dumps(
        {"label": label, "confidence": confidence, "toxicity": toxicity}
    )
    return json.dumps(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "model": "vendor/model",
            "provider": "provider",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.1},
        }
    ).encode()


class RequestTests(unittest.TestCase):
    def test_report_paths_inside_repository_are_portable(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            report_path(root / "experiments/example/output.jsonl"),
            "experiments/example/output.jsonl",
        )

    def test_strict_label_schema(self):
        self.assertEqual(
            validate_label(
                '{"label":"benign","confidence":"high","toxicity":"not_toxic"}'
            )["label"],
            "benign",
        )
        for invalid in (
            '{"label":"benign","confidence":"high","toxicity":"not_toxic","why":"x"}',
            '{"label":"attack","confidence":"high","toxicity":"not_toxic"}',
            '{"label":[],"confidence":"high","toxicity":"not_toxic"}',
        ):
            with self.assertRaises(ValueError):
                validate_label(invalid)

    def test_request_has_privacy_and_no_agent_loop(self):
        request, request_hash = build_request("vendor/model", "secret-key", "hello")
        body = json.loads(request.data)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertTrue(body["reasoning"]["exclude"])
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertEqual(body["provider"]["data_collection"], "deny")
        self.assertTrue(body["provider"]["zdr"])
        self.assertNotIn("tools", body)
        self.assertNotIn("secret-key", request.data.decode())
        self.assertEqual(len(request_hash), 64)

    def test_one_call_and_no_raw_response_in_result(self):
        calls = 0

        def opener(_request, timeout):
            nonlocal calls
            calls += 1
            self.assertEqual(timeout, 5)
            return _Response(_completion())

        request, _ = build_request("vendor/model", "key", "hello")
        result = call_once(request, 5, opener=opener)
        self.assertEqual(calls, 1)
        self.assertEqual(result["label"], "benign")
        self.assertNotIn("raw_response", result)
        self.assertNotIn("content", result)

    def test_invalid_completion_is_unavailable(self):
        result = parse_completion(b'{"choices":[]}')
        self.assertEqual(result["label"], "unavailable")
        self.assertEqual(result["unavailable_reason"], "invalid_choices")

    def test_failure_categories_do_not_persist_provider_content(self):
        result = parse_completion(
            b'{"choices":[{"finish_reason":"length","message":{"content":"secret"}}]}'
        )
        self.assertEqual(result["unavailable_reason"], "finish_length")
        self.assertNotIn("content", result)

    def test_sample_offset_selects_a_fresh_slice(self):
        required = {
            "detector_elevated": False,
            "language": "en",
            "length_bucket": "short",
            "source_toxic": "unavailable",
            "topic": "general",
            "security_trigger": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            rows = []
            for index in range(3):
                text = f"row {index}"
                rows.append(
                    {
                        **required,
                        "sample_id": f"sample-{index}",
                        "text": text,
                        "text_sha256": __import__("hashlib")
                        .sha256(text.encode())
                        .hexdigest(),
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected, _ = load_sample(path, limit=1, offset=1)
        self.assertEqual(selected[0]["sample_id"], "sample-1")


class AgreementTests(unittest.TestCase):
    @staticmethod
    def row(sample_id="sample", elevated=False):
        return {
            "sample_id": sample_id,
            "text": "ordinary text",
            "text_sha256": "hash",
            "conversation_sha256": "conversation",
            "detector_elevated": elevated,
        }

    @staticmethod
    def judgment(stage, label="benign", confidence="high"):
        return {
            "stage": stage,
            "model_requested": f"vendor/{stage}",
            "model_returned": f"vendor/{stage}",
            "provider": "provider",
            "label": label,
            "confidence": confidence,
            "toxicity": "not_toxic",
            "request_sha256": "request",
            "response_sha256": "response",
        }

    def test_detector_hard_always_requires_third(self):
        row = self.row(elevated=True)
        self.assertIn(row["sample_id"], audit_sample_ids([row]))

    def test_audit_sample_is_exact_and_includes_detector_alerts(self):
        rows = [self.row(f"ordinary-{index}") for index in range(23)]
        rows += [self.row(f"hard-{index}", elevated=True) for index in range(2)]
        selected = audit_sample_ids(rows)
        self.assertEqual(len(selected), 5)
        self.assertTrue({"hard-0", "hard-1"} <= selected)

    def test_audited_row_requires_third_agreement(self):
        row = self.row(elevated=True)
        entries = {
            ("sample", "primary_a"): self.judgment("primary_a"),
            ("sample", "primary_b"): self.judgment("primary_b"),
            ("sample", "third"): self.judgment("third", label="uncertain"),
        }
        accepted, decisions = accepted_rows([row], entries)
        self.assertEqual(accepted, [])
        self.assertEqual(decisions["third_not_high_benign"], 1)
        entries[("sample", "third")] = self.judgment("third")
        accepted, _ = accepted_rows([row], entries)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["weak_label"])

    def test_journal_rejects_raw_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "run_fingerprint": "run",
                        "sample_id": "sample",
                        "stage": "primary_a",
                        "raw_response": "forbidden",
                    }
                )
                + "\n"
            )
            with self.assertRaises(ValueError):
                load_journal(path, "run")


if __name__ == "__main__":
    unittest.main()
