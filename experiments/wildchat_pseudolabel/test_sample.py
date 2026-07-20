import json
import tempfile
import unittest
from pathlib import Path

from sample import (
    NearIndex,
    SHARDS,
    extract_candidate,
    filter_candidates,
    load_reference_index,
    normalize,
    prepare_source_toxicity_stratum,
    redact,
    simhash,
    stratified_select,
)


class PrivacyTests(unittest.TestCase):
    def test_redacts_pii_and_secrets_before_capping(self):
        text = "mail me at person@example.com, bearer abcdefghijk, +1 (555) 123-4567"
        value, count, truncated = redact(text, max_chars=128)
        self.assertNotIn("person@example.com", value)
        self.assertNotIn("abcdefghijk", value)
        self.assertNotIn("555", value)
        self.assertGreaterEqual(count, 3)
        self.assertFalse(truncated)

    def test_redacts_url_credentials_queries_fragments_and_jwts(self):
        jwt = "eyJabcdefghijk.abcdefghijk.abcdefghijk"
        text = (
            "https://user:password@example.com/file?sign=0123456789abcdef#session "
            "and callback?sessionid=private-value and " + jwt
        )
        value, count, _ = redact(text)
        self.assertNotIn("password", value)
        self.assertNotIn("0123456789abcdef", value)
        self.assertNotIn("private-value", value)
        self.assertNotIn(jwt, value)
        self.assertIn("[REDACTED_QUERY]", value)
        self.assertGreaterEqual(count, 3)

    def test_extracts_only_one_user_turn_and_safe_metadata(self):
        row = {
            "conversation_hash": "source-hash",
            "country": "private-country",
            "hashed_ip": "private-ip-hash",
            "header": {"user-agent": "private-agent"},
            "conversation": [
                {
                    "role": "user",
                    "content": "first sufficiently long user turn",
                    "language": "English",
                },
                {
                    "role": "assistant",
                    "content": "never retain this assistant response",
                },
                {
                    "role": "user",
                    "content": "second sufficiently long user turn",
                    "toxic": True,
                },
            ],
            "openai_moderation": [
                {"flagged": False},
                {"flagged": False},
                {"flagged": True},
            ],
        }
        result = extract_candidate(row, 10, "seed")
        self.assertIsNotNone(result)
        self.assertNotIn("country", result)
        self.assertNotIn("hashed_ip", result)
        self.assertNotIn("header", result)
        self.assertNotIn("assistant", result["text"])
        self.assertEqual(result, extract_candidate(row, 10, "seed"))

    def test_missing_source_toxicity_is_not_invented_as_false(self):
        row = {
            "conversation_hash": "source-hash",
            "conversation": [
                {"role": "user", "content": "a sufficiently long ordinary user turn"}
            ],
        }
        self.assertEqual(
            extract_candidate(row, 10, "seed")["source_toxic"], "unavailable"
        )


class SamplingTests(unittest.TestCase):
    def test_unvarying_source_toxicity_stratum_is_marked_unavailable(self):
        rows = [{"source_toxic": False}, {"source_toxic": False}]
        self.assertEqual(
            prepare_source_toxicity_stratum(rows),
            "unavailable_no_source_variation",
        )
        self.assertEqual({row["source_toxic"] for row in rows}, {"unavailable"})

    def test_shards_are_scattered_and_content_pinned(self):
        self.assertEqual([shard["index"] for shard in SHARDS], [0, 6, 13])
        self.assertEqual(sum(shard["size"] for shard in SHARDS), 756_333_381)
        self.assertTrue(all(len(shard["sha256"]) == 64 for shard in SHARDS))

    def test_near_reference_overlap_branch(self):
        base = (
            "This is a sufficiently long prompt about writing ordinary Python code "
            "for sorting a list and explaining each step clearly."
        )
        changed = base.replace("a list", "a small list")
        distance = (simhash(base) ^ simhash(changed)).bit_count()
        self.assertGreaterEqual(len(normalize(base)), 80)
        self.assertGreater(distance, 0)
        self.assertLessEqual(distance, 6)
        index = NearIndex()
        index.add(simhash(base))
        rows = [
            {"sample_id": "b", "conversation_sha256": "2", "text": changed},
            {
                "sample_id": "c",
                "conversation_sha256": "3",
                "text": "A completely different long discussion about gardening and soil care.",
            },
        ]
        kept, dropped = filter_candidates(rows, set(), index)
        self.assertEqual([row["sample_id"] for row in kept], ["c"])
        self.assertEqual(dropped["near_reference_overlap"], 1)

    def test_pilot_near_overlap_branch(self):
        base = (
            "This is a sufficiently long prompt about writing ordinary Python code "
            "for sorting a list and explaining each step clearly."
        )
        changed = base.replace("a list", "a small list")
        rows = [
            {"sample_id": "a", "conversation_sha256": "1", "text": base},
            {"sample_id": "b", "conversation_sha256": "2", "text": changed},
        ]
        kept, dropped = filter_candidates(rows, set(), NearIndex())
        self.assertEqual([row["sample_id"] for row in kept], ["a"])
        self.assertEqual(dropped["pilot_near_duplicate"], 1)

    def test_reference_matching_uses_post_redaction_text(self):
        raw = "Please send this ordinary document to person@example.com after reviewing it."
        redacted, _, _ = redact(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(json.dumps({"text": raw}) + "\n", encoding="utf-8")
            exact, near, count = load_reference_index(Path(directory))
        rows = [{"sample_id": "a", "conversation_sha256": "1", "text": redacted}]
        kept, dropped = filter_candidates(rows, exact, near)
        self.assertEqual(count, 1)
        self.assertEqual(kept, [])
        self.assertEqual(dropped["exact_reference_overlap"], 1)

    def test_joint_strata_round_robin_is_deterministic(self):
        rows = []
        for index in range(20):
            rows.append(
                {
                    "sample_id": str(index),
                    "language": "English" if index % 2 else "Spanish",
                    "length_bucket": "short" if index % 3 else "long",
                    "source_toxic": bool(index % 5 == 0),
                    "topic": "general" if index % 2 else "code",
                    "security_trigger": bool(index % 7 == 0),
                }
            )
        first = stratified_select([dict(row) for row in rows], 10, "seed")
        second = stratified_select([dict(row) for row in rows], 10, "seed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertGreater(len({row["language"] for row in first}), 1)


if __name__ == "__main__":
    unittest.main()
