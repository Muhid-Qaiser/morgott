"""Text-free, resumable numeric score shards for expensive evaluations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COLUMN = re.compile(r"[a-z][a-z0-9_]{0,63}")


def require_disjoint_paths(output: Path, score_journal: Path) -> None:
    """Reject equal or nested final-output and scratch-journal trees."""

    resolved_output = output.resolve()
    resolved_journal = score_journal.resolve()
    if (
        resolved_output == resolved_journal
        or resolved_output in resolved_journal.parents
        or resolved_journal in resolved_output.parents
    ):
        raise ValueError("score journal and final output paths must be disjoint")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical_json(value) + b"\n")


@dataclass(frozen=True)
class ScoreJournalSpec:
    """Immutable scoring identity; all free-form or corpus text is excluded."""

    model_sha256: str
    panel_sha256: str
    scoring_sha256: str
    rows: int
    batch_size: int
    columns: tuple[str, ...] = ("score",)

    def __post_init__(self) -> None:
        for value in (
            self.model_sha256,
            self.panel_sha256,
            self.scoring_sha256,
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError("score-journal identities must be SHA-256 digests")
        if type(self.rows) is not int or self.rows < 1:
            raise ValueError("score-journal row count must be positive")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("score-journal batch size must be positive")
        if (
            not isinstance(self.columns, tuple)
            or not self.columns
            or len(set(self.columns)) != len(self.columns)
            or any(
                not isinstance(column, str) or _COLUMN.fullmatch(column) is None
                for column in self.columns
            )
        ):
            raise ValueError("score-journal columns must be unique safe identifiers")

    def as_dict(self) -> dict:
        return {
            "model_sha256": self.model_sha256,
            "panel_sha256": self.panel_sha256,
            "scoring_sha256": self.scoring_sha256,
            "rows": self.rows,
            "batch_size": self.batch_size,
            "columns": list(self.columns),
            # Schema 1 recorded this fixed storage detail in identity. Keep the
            # literal so existing journal manifests reopen byte-for-byte.
            "dtype": "float64",
        }


class ScoreJournal:
    """Append-only, contiguous numeric score journal with atomic publication."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, spec: ScoreJournalSpec):
        self.root = root.resolve()
        self.spec = spec
        self.identity = spec.as_dict()
        self.identity_sha256 = _sha256_bytes(_canonical_json(self.identity))
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".writer.lock"
        self.shard_directory = self.root / "shards"
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_directory.mkdir(exist_ok=True)
        with self._exclusive_lock():
            self._refresh_locked(create=True)

    @contextmanager
    def _exclusive_lock(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _refresh_locked(self, *, create: bool = False) -> None:
        if self.manifest_path.exists():
            self._manifest = self._read_manifest()
        else:
            if not create:
                raise ValueError("score-journal manifest disappeared")
            unexpected = [
                path
                for path in self.root.iterdir()
                if path not in {self.lock_path, self.shard_directory}
            ]
            if unexpected:
                raise ValueError("score-journal directory has no valid manifest")
            self._manifest = self._empty_manifest()
            _atomic_json(self.manifest_path, self._manifest)
        self._validate_manifest_shards()
        self._recover_orphans()

    def _empty_manifest(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "shards": [],
        }

    def _read_manifest(self) -> dict:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid score-journal manifest") from error
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "identity",
                "identity_sha256",
                "shards",
            }
            or value.get("schema_version") != self.SCHEMA_VERSION
            or value.get("identity") != self.identity
            or value.get("identity_sha256") != self.identity_sha256
            or not isinstance(value.get("shards"), list)
        ):
            raise ValueError("score-journal identity or schema mismatch")
        return value

    @staticmethod
    def _shard_name(start: int, stop: int) -> str:
        return f"scores-{start:012d}-{stop:012d}.npz"

    def _read_shard(self, path: Path) -> tuple[int, int, np.ndarray]:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {
                    "identity_sha256",
                    "scores",
                    "start",
                    "stop",
                }:
                    raise ValueError("invalid score-journal shard arrays")
                identity_sha256 = str(payload["identity_sha256"].item())
                start = int(payload["start"].item())
                stop = int(payload["stop"].item())
                scores = np.asarray(payload["scores"])
        except (OSError, ValueError, EOFError) as error:
            raise ValueError(f"invalid score-journal shard: {path.name}") from error
        if (
            identity_sha256 != self.identity_sha256
            or path.name != self._shard_name(start, stop)
            or start < 0
            or stop <= start
            or stop > self.spec.rows
            or scores.dtype != np.dtype(np.float64)
            or scores.shape != (stop - start, len(self.spec.columns))
            or not np.isfinite(scores).all()
        ):
            raise ValueError(f"score-journal shard contract failed: {path.name}")
        return start, stop, scores

    def _validate_manifest_shards(self) -> None:
        expected_start = 0
        for entry in self._manifest["shards"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256", "start", "stop"}
                or type(entry["start"]) is not int
                or type(entry["stop"]) is not int
                or not isinstance(entry["path"], str)
                or not isinstance(entry["sha256"], str)
                or _SHA256.fullmatch(entry["sha256"]) is None
                or entry["start"] != expected_start
                or entry["path"] != self._shard_name(entry["start"], entry["stop"])
            ):
                raise ValueError("invalid score-journal shard manifest")
            path = self.shard_directory / entry["path"]
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise ValueError("score-journal shard hash mismatch")
            start, stop, _ = self._read_shard(path)
            if start != entry["start"] or stop != entry["stop"]:
                raise ValueError("score-journal shard range mismatch")
            expected_start = stop

    def _recover_orphans(self) -> None:
        known = {entry["path"] for entry in self._manifest["shards"]}
        orphans = sorted(
            path
            for path in self.shard_directory.glob("*.npz")
            if path.name not in known
        )
        changed = False
        for path in orphans:
            start, stop, _ = self._read_shard(path)
            if start != self.completed_rows:
                raise ValueError("non-contiguous orphan score-journal shard")
            self._manifest["shards"].append(
                {
                    "path": path.name,
                    "sha256": _file_sha256(path),
                    "start": start,
                    "stop": stop,
                }
            )
            changed = True
        if changed:
            _atomic_json(self.manifest_path, self._manifest)

    @property
    def completed_rows(self) -> int:
        shards = self._manifest["shards"]
        return int(shards[-1]["stop"]) if shards else 0

    @property
    def complete(self) -> bool:
        return self.completed_rows == self.spec.rows

    def append(
        self, scores: np.ndarray, *, start: int | None = None
    ) -> tuple[int, int]:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim == 1 and len(self.spec.columns) == 1:
            values = values[:, np.newaxis]
        if (
            values.ndim != 2
            or values.shape[1] != len(self.spec.columns)
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid score-journal score array")
        observed_start = self.completed_rows
        requested_start = start
        with self._exclusive_lock():
            self._refresh_locked()
            expected_start = self.completed_rows
            start = observed_start if requested_start is None else requested_start
            stop = start + len(values)
            if (
                type(start) is not int
                or start != expected_start
                or (requested_start is None and expected_start != observed_start)
                or stop > self.spec.rows
            ):
                raise ValueError(
                    "score-journal append is not the next contiguous range"
                )

            path = self.shard_directory / self._shard_name(start, stop)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.shard_directory,
                prefix=f".{path.name}.",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez(
                        handle,
                        identity_sha256=np.asarray(self.identity_sha256),
                        scores=values,
                        start=np.asarray(start, dtype=np.int64),
                        stop=np.asarray(stop, dtype=np.int64),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(self.shard_directory)
            finally:
                temporary.unlink(missing_ok=True)

            entry = {
                "path": path.name,
                "sha256": _file_sha256(path),
                "start": start,
                "stop": stop,
            }
            updated = {
                **self._manifest,
                "shards": [*self._manifest["shards"], entry],
            }
            _atomic_json(self.manifest_path, updated)
            self._manifest = updated
            return start, stop

    def missing_ranges(self, shard_rows: int) -> list[tuple[int, int]]:
        if type(shard_rows) is not int or shard_rows < 1:
            raise ValueError("shard row count must be positive")
        return [
            (start, min(start + shard_rows, self.spec.rows))
            for start in range(self.completed_rows, self.spec.rows, shard_rows)
        ]

    def scores(self) -> np.ndarray:
        if not self.complete:
            raise ValueError("score journal is incomplete")
        values = [
            self._read_shard(self.shard_directory / entry["path"])[2]
            for entry in self._manifest["shards"]
        ]
        return np.concatenate(values, axis=0)
