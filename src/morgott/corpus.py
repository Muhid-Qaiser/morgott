from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from datasets import disable_progress_bars

from .data import SOURCES, atomic_write_text, build_dataset, file_sha256
from .routing import materialize_routing_views
from .sources import LOADERS


def _consume_source(path: Path, rows: Iterable[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    counts = Counter()
    row_ids = set()
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                if row["id"] in row_ids:
                    raise ValueError(f"duplicate canonical row id: {row['id']}")
                row_ids.add(row["id"])
                role = row.get("source_role")
                eligible = row.get("routing_training_eligible")
                expected_eligible = role in {"candidate", "dev_test"}
                if type(eligible) is not bool or eligible != expected_eligible:
                    raise ValueError(
                        f"{row['id']} has inconsistent routing source role/eligibility"
                    )
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts["rows"] += 1
                counts[f"role:{role}"] += 1
                counts[f"security:{row['security_label']}"] += 1
                counts[f"routing:{row['routing_label']}"] += 1
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary = {
        "rows": counts["rows"],
        "roles": {
            key.removeprefix("role:"): value
            for key, value in sorted(counts.items())
            if key.startswith("role:")
        },
        "security_labels": {
            key.removeprefix("security:"): value
            for key, value in sorted(counts.items())
            if key.startswith("security:")
        },
        "routing_benign": counts["routing:0"],
        "routing_non_benign": counts["routing:1"],
        "sha256": file_sha256(path),
    }
    return summary


def _consume_source_quarantine(path: Path, rows: Iterable[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    counts = Counter()
    row_ids = set()
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                row_id = row.get("id")
                if (
                    not isinstance(row_id, str)
                    or not row_id
                    or row_id in row_ids
                    or not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                    or row.get("source_role") != "uncertain"
                    or row.get("routing_training_eligible") is not False
                    or row.get("data_role") != "quarantine"
                    or not isinstance(row.get("quarantine_reason"), str)
                ):
                    raise ValueError("invalid source-level quarantine row")
                row_ids.add(row_id)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts["rows"] += 1
                counts[f"reason:{row['quarantine_reason']}"] += 1
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": counts["rows"],
        "reasons": {
            key.removeprefix("reason:"): value
            for key, value in sorted(counts.items())
            if key.startswith("reason:")
        },
        "sha256": file_sha256(path),
    }


def _extend_corpus(data_dir: Path, *, core_manifest_path: Path) -> dict:
    disable_progress_bars()
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    source_dir = data_dir / "sources"
    source_outputs = {}
    source_profiles = {}
    source_quarantines = {}
    downloads = {}
    for source, loader in LOADERS.items():
        loaded = loader()
        if len(loaded) == 3:
            rows, source_downloads, source_profile = loaded
            quarantine_rows = None
        elif len(loaded) == 4:
            rows, source_downloads, source_profile, quarantine_rows = loaded
        else:
            raise ValueError(f"{source} loader returned an invalid result")
        summary = _consume_source(source_dir / f"{source}.jsonl", rows)
        summary["path"] = str((source_dir / f"{source}.jsonl").relative_to(data_dir))
        source_outputs[source] = summary
        source_profiles[source] = source_profile
        downloads[source] = source_downloads
        if quarantine_rows is not None:
            quarantine_path = data_dir / "quarantine" / f"{source}_sensitive.jsonl"
            quarantine_summary = _consume_source_quarantine(
                quarantine_path, quarantine_rows
            )
            quarantine_summary["path"] = str(quarantine_path.relative_to(data_dir))
            source_quarantines[f"{source}_sensitive"] = quarantine_summary

    combined_outputs = {**core_manifest["source_outputs"], **source_outputs}
    with tempfile.TemporaryDirectory(
        dir=data_dir, prefix=".routing-build-"
    ) as directory:
        routing_views, routing_quarantine, routing_stats = materialize_routing_views(
            data_dir, combined_outputs, Path(directory)
        )
    manifest = {
        **core_manifest,
        "schema_version": 5,
        "canonical_row_schema_version": 5,
        "sources": {
            **core_manifest["sources"],
            **{source: SOURCES[source] for source in source_outputs},
        },
        "source_outputs": combined_outputs,
        "source_profiles": {
            **core_manifest["source_profiles"],
            **source_profiles,
        },
        "download_sha256": {
            **core_manifest["download_sha256"],
            **{
                f"{source}/{filename}": digest
                for source, source_downloads in downloads.items()
                for filename, digest in source_downloads.items()
            },
        },
        "routing_views": routing_views,
        "quarantines": {
            **core_manifest["quarantines"],
            **source_quarantines,
            "routing": routing_quarantine,
        },
        "routing_deduplication": routing_stats,
    }
    atomic_write_text(
        data_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_corpus(data_dir: Path = Path("data")) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir, prefix=".core-build-") as directory:
        core_manifest_path = Path(directory) / "manifest.json"
        build_dataset(data_dir, manifest_path=core_manifest_path)
        return _extend_corpus(data_dir, core_manifest_path=core_manifest_path)


def rebuild_routing(data_dir: Path = Path("data")) -> dict:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 5
        or manifest.get("canonical_row_schema_version") != 5
    ):
        raise ValueError("data manifest is not canonical schema 5")
    source_outputs = manifest.get("source_outputs")
    if not isinstance(source_outputs, dict) or not source_outputs:
        raise ValueError("data manifest has no canonical source outputs")
    expected_sources = set(SOURCES)
    if set(source_outputs) != expected_sources:
        missing = sorted(expected_sources - set(source_outputs))
        unexpected = sorted(set(source_outputs) - expected_sources)
        raise ValueError(
            f"data manifest source set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    expected_metadata = {source: SOURCES[source] for source in SOURCES}
    if manifest.get("sources") != expected_metadata:
        raise ValueError("data manifest source metadata does not match current sources")
    with tempfile.TemporaryDirectory(
        dir=data_dir, prefix=".routing-build-"
    ) as directory:
        routing_views, routing_quarantine, routing_stats = materialize_routing_views(
            data_dir,
            source_outputs,
            Path(directory),
            invalidate_manifest=manifest_path,
        )
    updated = {
        **manifest,
        "schema_version": 5,
        "routing_views": routing_views,
        "quarantines": {
            **manifest.get("quarantines", {}),
            "routing": routing_quarantine,
        },
        "routing_deduplication": routing_stats,
    }
    atomic_write_text(
        manifest_path, json.dumps(updated, indent=2, sort_keys=True) + "\n"
    )
    return updated
