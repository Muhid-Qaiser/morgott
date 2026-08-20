"""Focused equivalence tests for the data-layer restructure.

These pin the parts of the refactor that must stay output-neutral: the shared
shard consumer returns exactly the summary that `morgott.corpus` and
`morgott.data.build_dataset` each produced before extraction, every loader
keeps the uniform four-element contract, and the private import surfaces that
tests and frozen experiments rely on keep resolving.
"""

import json
import tempfile
import unittest
from pathlib import Path

from morgott.data import (
    _consume_source_rows,
    _sample,
    _set_source_role,
    file_sha256,
)
from morgott.sources import LOADERS


def _synthetic_rows() -> list[dict]:
    rows = []
    for index, (label, role) in enumerate(
        ((1, "candidate"), (0, "dev_test"), (None, "uncertain"), (1, "auxiliary"))
    ):
        row = _sample(
            text=f"synthetic canonical row {index}",
            label=label,
            attack_type="direct_prompt_injection" if label else None,
            security_label="uncertain" if label is None else None,
            source="gandalf",
            source_split="train",
            source_id=f"row-{index}",
            group_id=f"gandalf:{index}",
        )
        rows.append(_set_source_role(row, role))
    return rows


class ConsumeSourceRowsTests(unittest.TestCase):
    def test_summary_matches_the_pre_extraction_golden(self):
        rows = _synthetic_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gandalf.jsonl"
            summary = _consume_source_rows(path, rows)
            written = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            observed_sha = file_sha256(path)
        # Golden values: the summary `corpus._consume_source` produced and the
        # per-source fields `build_dataset` derived in place before the shared
        # helper existed. The manifest content must not change.
        self.assertEqual(
            summary,
            {
                "rows": 4,
                "roles": {
                    "auxiliary": 1,
                    "candidate": 1,
                    "dev_test": 1,
                    "uncertain": 1,
                },
                "security_labels": {
                    "benign": 1,
                    "direct_prompt_injection": 2,
                    "uncertain": 1,
                },
                "routing_benign": 1,
                "routing_non_benign": 3,
                "sha256": observed_sha,
            },
        )
        self.assertEqual(written, rows)

    def test_duplicate_ids_keep_both_historical_messages(self):
        rows = _synthetic_rows()
        rows.append(dict(rows[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gandalf.jsonl"
            with self.assertRaisesRegex(ValueError, "^duplicate canonical row id: "):
                _consume_source_rows(path, rows)
            with self.assertRaisesRegex(
                ValueError, "^duplicate canonical row id in gandalf: "
            ):
                _consume_source_rows(path, rows, source="gandalf")

    def test_inconsistent_eligibility_fails_closed(self):
        rows = _synthetic_rows()
        rows[0]["routing_training_eligible"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gandalf.jsonl"
            with self.assertRaisesRegex(ValueError, "source role/eligibility"):
                _consume_source_rows(path, rows)


class LoaderContractTests(unittest.TestCase):
    def test_loader_names_match_their_source_keys(self):
        for source, loader in LOADERS.items():
            self.assertEqual(loader.__name__, f"_{source}_rows")

    def test_loader_annotations_declare_the_four_element_contract(self):
        for source, loader in LOADERS.items():
            annotation = loader.__annotations__.get("return")
            self.assertIsNotNone(annotation, source)
            self.assertIn("Iterator[dict] | None", str(annotation), source)


class ImportSurfaceTests(unittest.TestCase):
    def test_private_contract_names_stay_importable_from_data(self):
        from morgott.data import (
            _atomic_text_writer,
            _fetch,
            _github_raw,
            _sample,
            _set_core_routing_role,
            _write_json,
            _write_jsonl,
        )

        for helper in (
            _atomic_text_writer,
            _fetch,
            _github_raw,
            _sample,
            _set_core_routing_role,
            _write_json,
            _write_jsonl,
        ):
            self.assertTrue(callable(helper))

    def test_frozen_experiment_names_stay_importable_from_tasks(self):
        from morgott.sources.tasks import (
            _SENSITIVE_TEXT_PATTERNS,
            _public_declared_license,
            _sensitive_text_reasons,
        )

        self.assertTrue(_SENSITIVE_TEXT_PATTERNS)
        self.assertTrue(callable(_public_declared_license))
        self.assertEqual(_sensitive_text_reasons("no sensitive content here"), [])

    def test_core_loaders_import_without_circularity(self):
        from morgott.sources import core

        for name in (
            "_load_toxic_chat",
            "_load_prompt_injections",
            "_load_xstest",
            "_load_multi_turn",
            "_load_oasst1",
            "_load_harmbench",
            "_load_do_not_answer",
            "_load_bipia",
            "_load_notinject",
            "_load_jailbreaks_over_time",
            "_load_tensor_trust",
            "_load_nemotron_agentic_ipi",
            "_oasst_position_stress",
            "_parse_nemotron_agentic_ipi",
        ):
            self.assertTrue(callable(getattr(core, name)))


if __name__ == "__main__":
    unittest.main()
