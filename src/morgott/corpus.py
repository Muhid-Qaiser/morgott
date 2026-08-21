from __future__ import annotations

import json
import os
import tempfile
import traceback
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

from datasets import disable_progress_bars

from .data import (
    SOURCES,
    _atomic_text_writer,
    atomic_write_text,
    build_dataset,
    file_sha256,
)
from .data import (
    _consume_source_rows as _consume_source,
)
from .routing import materialize_routing_views
from .sources import LOADERS


def _consume_source_quarantine(path: Path, rows: Iterable[dict]) -> dict:
    counts = Counter()
    row_ids = set()
    with _atomic_text_writer(path) as handle:
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


def _load_source(
    data_dir: Path, item: tuple[str, Callable]
) -> tuple[str, dict, dict, dict, dict | None]:
    """Run one loader and consume its rows inside a worker process.

    The loader call and consumption stay in the same process because some
    loaders return generators whose consumption mutates the profile dict;
    the profile returned here is the post-consumption one.
    """
    disable_progress_bars()
    source, loader = item
    try:
        rows, source_downloads, source_profile, quarantine_rows = loader()
        source_path = data_dir / "sources" / f"{source}.jsonl"
        summary = _consume_source(source_path, rows)
        summary["path"] = str(source_path.relative_to(data_dir))
        quarantine_summary = None
        if quarantine_rows is not None:
            quarantine_path = data_dir / "quarantine" / f"{source}_sensitive.jsonl"
            quarantine_summary = _consume_source_quarantine(
                quarantine_path, quarantine_rows
            )
            quarantine_summary["path"] = str(quarantine_path.relative_to(data_dir))
    except Exception as exc:
        # Some loader exceptions pickle but cannot be rebuilt in the parent
        # (urllib's HTTPError, hf_hub gated-repo errors), which would break
        # result handling; re-raise as a plain transportable error instead.
        raise RuntimeError(
            f"{source} loader failed: {exc!r}\n{traceback.format_exc()}"
        ) from None
    return source, summary, source_profile, source_downloads, quarantine_summary


def _extend_corpus(
    data_dir: Path, *, core_manifest_path: Path, workers: int | None = None
) -> dict:
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    if workers is None:
        # ponytail: capped at 4 workers for the shared 30GB box, raise the
        # cap when builds move to a machine with more memory.
        workers = min(4, os.cpu_count() or 1)
    # Each worker writes its own shard file. executor.map yields results in
    # LOADERS iteration order regardless of completion order, so manifest
    # assembly stays deterministic. A loader failure raises here once the
    # results ordered before it have arrived (not-yet-started loaders are
    # cancelled, in-flight ones finish first), and a worker that dies
    # abruptly (for example the OOM killer) raises BrokenProcessPool, so the
    # build always aborts instead of hanging (fail closed).
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        results = list(executor.map(partial(_load_source, data_dir), LOADERS.items()))
    source_outputs = {}
    source_profiles = {}
    source_quarantines = {}
    downloads = {}
    for (
        source,
        summary,
        source_profile,
        source_downloads,
        quarantine_summary,
    ) in results:
        source_outputs[source] = summary
        source_profiles[source] = source_profile
        downloads[source] = source_downloads
        if quarantine_summary is not None:
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
