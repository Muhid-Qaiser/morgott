import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from corpus_test_support import _read_rows, _row, _source_output

from morgott.corpus import (
    _consume_source,
    rebuild_routing,
)
from morgott.data import SOURCES, _sample, _set_source_role
from morgott.routing import materialize_routing_views


class CorpusTests(unittest.TestCase):
    def test_source_writer_rejects_duplicate_ids_and_inconsistent_role(self):
        first = _row("duplicate", "first attack", "first")
        second = {**first, "text": "second attack"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            with self.assertRaisesRegex(ValueError, "duplicate canonical row id"):
                _consume_source(path, (first, second))
            broken = {**first, "routing_training_eligible": False}
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                _consume_source(path, (broken,))

    def test_auxiliary_rows_are_hash_and_label_validated(self):
        for field, value, message in (
            ("normalized_text_sha256", "invalid", "invalid text hash"),
            ("routing_label", 2, "invalid routing label"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = _row("aux", "auxiliary text", "aux", "auxiliary")
                row[field] = value
                output = _source_output(root, [row])
                build = root / "build"
                build.mkdir()
                with self.assertRaisesRegex(ValueError, message):
                    materialize_routing_views(root, {"gandalf": output}, build)

    def test_singleton_exact_group_preserves_the_canonical_row(self):
        row = _row("one", "single canonical attack", "group-one")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, [row])
            build = root / "build"
            build.mkdir()
            with patch(
                "morgott.routing.zlib.compress", wraps=zlib.compress
            ) as compress:
                views, _, _ = materialize_routing_views(
                    root, {"gandalf": output}, build
                )
            routed = [
                candidate
                for name in ("train", "validation", "dev_test")
                for candidate in _read_rows(root / views[name]["path"])
            ]

        expected = {**row, "data_role": routed[0]["data_role"]}
        self.assertEqual(routed, [expected])
        self.assertEqual(compress.call_count, 1)

    def test_routing_quarantines_strict_train_dev_overlap(self):
        rows = [
            _row("held", "alpha beta gamma delta", "held", "dev_test"),
            _row("candidate", "alpha\u200b beta gamma delta", "candidate"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            with (
                patch("morgott.routing.OVERLAP_BATCH_SIZE", 1),
                patch("morgott.routing.OVERLAP_WORKERS", 2),
            ):
                views, quarantine, _ = materialize_routing_views(
                    root, {"gandalf": output}, build
                )
            supervised = [
                row
                for name in ("train", "validation", "dev_test")
                for row in _read_rows(root / views[name]["path"])
            ]
            quarantine_rows = _read_rows(root / quarantine["path"])

        self.assertEqual([row["source_id"] for row in supervised], ["held"])
        self.assertEqual(
            [row["quarantine_reason"] for row in quarantine_rows],
            ["strict_dev_test_overlap"],
        )

    def test_routing_quarantines_strict_validation_train_overlap(self):
        rows = [
            _row("held-a", "official held alpha beta", "held-a", "dev_test"),
            _row("held-b", "official held gamma delta", "held-b", "dev_test"),
            *[
                _row(
                    f"candidate-{index}",
                    f"alpha{'\u200b' * index} beta gamma delta",
                    f"candidate-{index}",
                )
                for index in range(1, 9)
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            views, quarantine, _ = materialize_routing_views(
                root, {"gandalf": output}, build
            )
            validation = _read_rows(root / views["validation"]["path"])
            quarantine_rows = _read_rows(root / quarantine["path"])

        self.assertEqual(validation, [])
        # The hash-based partitioner decides which split each colliding
        # candidate lands in first, so assert the strict quarantine family
        # rather than one exact reason ordering.
        self.assertTrue(quarantine_rows)
        self.assertTrue(
            all(
                row["quarantine_reason"].startswith("strict_")
                for row in quarantine_rows
            )
        )

    def test_exact_duplicates_do_not_connect_unrelated_lineage_networks(self):
        rows = [
            _row("one", "same attack", "group-one"),
            _row("two", "SAME ATTACK", "group-two"),
            _row("three", "unique candidate marker", "group-two"),
            _row("held-duplicate", "same attack", "official-group", "dev_test"),
            _row("held", "held out marker", "official-group", "dev_test"),
            _row("independent", "completely separate candidate", "independent"),
            _row("aux", "auxiliary only text", "auxiliary", "auxiliary"),
        ]
        uncertain = _sample(
            text="completely separate candidate",
            label=None,
            attack_type=None,
            security_label="uncertain",
            source="gandalf",
            source_split="train",
            source_id="uncertain",
            group_id="uncertain",
        )
        rows.append(_set_source_role(uncertain, "uncertain"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            views, quarantine, stats = materialize_routing_views(
                root, {"gandalf": output}, build
            )
            dev_rows = _read_rows(root / views["dev_test"]["path"])
            supervised = [
                row
                for name in ("train", "validation", "dev_test")
                for row in _read_rows(root / views[name]["path"])
            ]
            quarantine_rows = _read_rows(root / quarantine["path"])

        dev_ids = {origin["source_id"] for row in dev_rows for origin in row["origins"]}
        self.assertTrue({"one", "two", "held-duplicate", "held"} <= dev_ids)
        self.assertNotIn("three", dev_ids)
        self.assertEqual(sum(row["text"] == "same attack" for row in supervised), 1)
        self.assertTrue(
            any(row["text"] == "completely separate candidate" for row in supervised)
        )
        self.assertFalse(
            any(row["text"] == "auxiliary only text" for row in supervised)
        )
        self.assertTrue(
            any(
                row["quarantine_reason"] == "exact_supervised_overlap"
                for row in quarantine_rows
            )
        )
        self.assertEqual(stats["cross_lineage_exact_duplicates"], 1)

    def test_grouped_split_targets_each_source_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {}
            for source in ("gandalf", "prompt_injections"):
                rows = [
                    _row(
                        f"{label}-{index}",
                        f"{source} token{label}{index} oak{index} birch{index}",
                        f"{source}:{label}:{index}",
                        (
                            "dev_test"
                            if source == "prompt_injections"
                            and label == 1
                            and index < 20
                            else "candidate"
                        ),
                        label=label,
                        source=source,
                    )
                    for label in (0, 1)
                    for index in range(100)
                ]
                outputs[source] = _source_output(root, rows)
            build = root / "build"
            build.mkdir()
            views, _, stats = materialize_routing_views(root, outputs, build)
            counts = {}
            for name in ("train", "validation", "dev_test"):
                for row in _read_rows(root / views[name]["path"]):
                    counts[name, row["source"], row["routing_label"]] = (
                        counts.get((name, row["source"], row["routing_label"]), 0) + 1
                    )

        for source in outputs:
            for label in (0, 1):
                self.assertEqual(counts["train", source, label], 70)
                self.assertEqual(counts["validation", source, label], 10)
                self.assertEqual(counts["dev_test", source, label], 20)
        self.assertEqual(
            stats["target_ratios"], {"train": 0.7, "validation": 0.1, "dev_test": 0.2}
        )

    def test_routing_only_rebuild_requires_manifest_verified_sources(self):
        rows = [_row("one", "one two three four", "group-one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            manifest_path = root / "manifest.json"
            manifest = {
                "schema_version": 4,
                "canonical_row_schema_version": 5,
                "sources": {"gandalf": SOURCES["gandalf"]},
                "source_outputs": {"gandalf": output},
                "quarantines": {},
            }
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "canonical schema 5"):
                rebuild_routing(root)
            manifest["schema_version"] = 5
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source set mismatch"):
                rebuild_routing(root)
            with (
                patch("morgott.corpus.SOURCES", {"gandalf": SOURCES["gandalf"]}),
                patch("morgott.routing.SOURCES", {"gandalf": SOURCES["gandalf"]}),
            ):
                rebuilt = rebuild_routing(root)
            self.assertEqual(rebuilt["routing_views"]["train"]["rows"], 1)
            source_path = root / output["path"]
            source_path.write_bytes(source_path.read_bytes().replace(b'": ', b'":', 1))
            with (
                patch("morgott.corpus.SOURCES", {"gandalf": SOURCES["gandalf"]}),
                patch("morgott.routing.SOURCES", {"gandalf": SOURCES["gandalf"]}),
                self.assertRaisesRegex(RuntimeError, "source shard changed"),
            ):
                rebuild_routing(root)
            self.assertTrue(manifest_path.is_file())

    def test_routing_reads_each_verified_source_once(self):
        rows = [_row("one", "one two three four", "group-one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _source_output(root, rows)
            source_path = root / output["path"]
            build = root / "build"
            build.mkdir()
            source_opens = 0
            original_open = Path.open

            def counting_open(path, *args, **kwargs):
                nonlocal source_opens
                if path == source_path:
                    source_opens += 1
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", counting_open):
                materialize_routing_views(root, {"gandalf": output}, build)

        self.assertEqual(source_opens, 1)

    def test_official_holdout_is_counted_before_flexible_split(self):
        candidates = [
            _row(
                f"candidate-{label}-{index}",
                f"candidate token{label}{index} oak{index} birch{index}",
                f"candidate:{label}:{index}",
                label=label,
            )
            for label in (0, 1)
            for index in range(80)
        ]
        official = [
            _row(
                f"official-{index}",
                f"official token{index} maple{index} elm{index}",
                f"official:{index}",
                "dev_test",
                source="bipia",
            )
            for index in range(40)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                "gandalf": _source_output(root, candidates),
                "bipia": _source_output(root, official),
            }
            build = root / "build"
            build.mkdir()
            views, _, stats = materialize_routing_views(root, outputs, build)

        self.assertEqual(
            {name: views[name]["rows"] for name in ("train", "validation", "dev_test")},
            {"train": 140, "validation": 20, "dev_test": 40},
        )
        self.assertEqual(stats["official_dev_test_rows"], 40)
        self.assertEqual(views["dev_test"]["largest_split_group"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
