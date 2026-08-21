"""Focused equivalence tests for the data-layer restructure.

These pin the parts of the refactor that must stay output-neutral: the shared
shard consumer returns exactly the summary that `morgott.corpus` and
`morgott.data.build_dataset` each produced before extraction, every loader
keeps the uniform four-element contract, and the private import surfaces that
tests and frozen experiments rely on keep resolving. They also pin the pooled
routing build (parallel ingest and pipelined overlap batches) to the serial
build byte for byte.
"""

import json
import os
import signal
import tempfile
import threading
import unittest
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

from corpus_test_support import _row, _source_output

from morgott.data import (
    _consume_source_rows,
    _sample,
    _set_source_role,
    file_sha256,
)
from morgott.routing import _pipelined_map, materialize_routing_views
from morgott.sources import LOADERS

_KILL_MARKER = "kill this worker"


def _echo_or_kill(value: str) -> str:
    if value == _KILL_MARKER:
        os.kill(os.getpid(), signal.SIGKILL)
    return value


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


def _routing_source_outputs(root: Path) -> dict[str, dict]:
    """A mini corpus covering every routing path the pools touch.

    Official dev-test rows, both routing labels, an exact duplicate across
    lineages, an exact label conflict, a strict near duplicate of a dev-test
    row, an auxiliary row, and uncertain rows that both do and do not overlap
    the supervised views.
    """
    gandalf = [
        _row(
            f"held-{index}",
            f"official held marker {index} maple{index} elm{index}",
            f"official:{index}",
            "dev_test",
        )
        for index in range(4)
    ]
    gandalf += [
        _row(
            f"cand-{index}",
            f"gandalf candidate {index} oak{index} birch{index}",
            f"gandalf:{index}",
            label=index % 2,
        )
        for index in range(16)
    ]
    gandalf.append(_row("exact-dup", "gandalf candidate 3 oak3 birch3", "dup:lineage"))
    gandalf.append(
        _row("near-dup", "official held marker 1\u200b maple1 elm1", "near:1")
    )
    gandalf.append(_row("conflict-a", "conflicting marker text alpha", "conflict:a"))
    gandalf.append(
        _row("conflict-b", "conflicting marker text alpha", "conflict:b", label=0)
    )
    gandalf.append(_row("aux", "auxiliary only text", "aux", "auxiliary"))
    for source_id, text in (
        ("uncertain-overlap", "gandalf candidate 5 oak5 birch5"),
        ("uncertain-free", "entirely unresolved marker text"),
    ):
        uncertain = _sample(
            text=text,
            label=None,
            attack_type=None,
            security_label="uncertain",
            source="gandalf",
            source_split="train",
            source_id=source_id,
            group_id=source_id,
        )
        gandalf.append(_set_source_role(uncertain, "uncertain"))
    prompt_injections = [
        _row(
            f"pi-{index}",
            f"prompt injection {index} cedar{index} pine{index}",
            f"pi:{index}",
            label=index % 2,
            source="prompt_injections",
        )
        for index in range(16)
    ]
    return {
        "gandalf": _source_output(root, gandalf),
        "prompt_injections": _source_output(root, prompt_injections),
    }


class RoutingParallelismEquivalenceTests(unittest.TestCase):
    """The pooled routing build must be byte-identical to the serial build."""

    def _materialize(self, *, workers: int, batch: int):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = _routing_source_outputs(root)
            build = root / "build"
            build.mkdir()
            with (
                patch("morgott.routing.INGEST_WORKERS", workers),
                patch("morgott.routing.INGEST_BATCH_LINES", batch),
                patch("morgott.routing.OVERLAP_WORKERS", workers),
                patch("morgott.routing.OVERLAP_BATCH_SIZE", batch),
                patch("morgott.routing.get_context", wraps=get_context) as context,
            ):
                views, quarantine, stats = materialize_routing_views(
                    root, outputs, build
                )
            contents = {
                name: (root / summary["path"]).read_bytes()
                for name, summary in {**views, "quarantine": quarantine}.items()
            }
        return views, quarantine, stats, contents, context.call_count

    def test_pooled_build_matches_the_serial_build_byte_for_byte(self):
        serial = self._materialize(workers=1, batch=2)
        pooled = self._materialize(workers=2, batch=2)
        self.assertEqual(serial[4], 0)
        self.assertEqual(pooled[4], 2)
        self.assertEqual(serial[:4], pooled[:4])

    def test_pooled_ingest_keeps_the_serial_duplicate_id_error(self):
        # A row that is both a duplicate and invalid must fail on the
        # duplicate id, exactly as the serial check order decides.
        rows = [
            _row(f"fill-{index}", f"filler text {index} oak{index}", f"fill:{index}")
            for index in range(6)
        ]
        rows.append(_row("dup-id", "duplicated row text", "dup:a"))
        broken = _row("dup-id", "second body with a bad role", "dup:b")
        broken["source_role"] = "bogus"
        rows.append(broken)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            with (
                patch("morgott.routing.INGEST_WORKERS", 2),
                patch("morgott.routing.INGEST_BATCH_LINES", 1),
                self.assertRaisesRegex(ValueError, "gandalf:8 has a duplicate id"),
            ):
                materialize_routing_views(root, {"gandalf": output}, build)

    def test_pooled_ingest_raises_a_worker_validation_error(self):
        # A worker-detected invalid row returns fields=None, the same shape as
        # an auxiliary row, so if the parent stopped raising worker errors the
        # pooled build would silently drop the row instead of failing closed.
        rows = [
            _row(f"fill-{index}", f"filler text {index} oak{index}", f"fill:{index}")
            for index in range(6)
        ]
        corrupted = _row("bad-hash", "row with a corrupted hash", "bad:hash")
        corrupted["normalized_text_sha256"] = "0" * 64
        rows.append(corrupted)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            with (
                patch("morgott.routing.INGEST_WORKERS", 2),
                patch("morgott.routing.INGEST_BATCH_LINES", 1),
                self.assertRaisesRegex(ValueError, "gandalf:7 has invalid text hash"),
            ):
                materialize_routing_views(root, {"gandalf": output}, build)

    def test_a_killed_worker_aborts_the_pipelined_map_instead_of_hanging(self):
        # An OOM-killed worker must fail the build closed with an exception,
        # not leave the parent blocked forever on a result that never arrives
        # (multiprocessing.Pool silently respawns dead workers and hangs).
        outcome = {}

        def drain() -> None:
            batches = [["a", "b"], ["c", _KILL_MARKER], ["d", "e"]]
            try:
                with ProcessPoolExecutor(
                    2, mp_context=get_context("spawn")
                ) as executor:
                    for _ in _pipelined_map(
                        executor, _echo_or_kill, batches, chunksize=1
                    ):
                        pass
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive(), "a killed worker hung the pipelined map")
        self.assertIsInstance(outcome.get("error"), BrokenProcessPool)


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
