"""Text-free, resumable numeric scores for expensive evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
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


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
            "dtype": "float64",
        }


class ScoreJournal:
    """Append-only, contiguous numeric score journal backed by SQLite."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, spec: ScoreJournalSpec):
        self.root = root.resolve()
        self.spec = spec
        self.identity = spec.as_dict()
        self._identity_json = _canonical_json(self.identity)
        self.identity_sha256 = hashlib.sha256(
            self._identity_json.encode("utf-8")
        ).hexdigest()
        self.database_path = self.root / "scores.sqlite3"

        self.root.mkdir(parents=True, exist_ok=True)
        database_files = {
            self.database_path.name,
            f"{self.database_path.name}-journal",
            f"{self.database_path.name}-shm",
            f"{self.database_path.name}-wal",
        }
        unexpected = [
            path for path in self.root.iterdir() if path.name not in database_files
        ]
        if unexpected:
            raise ValueError("score-journal directory has no valid database")
        try:
            with closing(self._connect()) as database:
                self._initialize(database)
                self._completed_rows = self._read_completed(database)
                self._read_scores(database, self._completed_rows)
        except sqlite3.Error as error:
            raise ValueError("invalid score-journal database") from error

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.database_path, timeout=30)
        database.execute("PRAGMA busy_timeout = 30000")
        database.execute("PRAGMA synchronous = FULL")
        return database

    def _initialize(self, database: sqlite3.Connection) -> None:
        with database:
            database.execute("BEGIN IMMEDIATE")
            version = database.execute("PRAGMA user_version").fetchone()[0]
            tables = database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            if version != 0 or tables:
                return
            database.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            database.execute(
                """
                CREATE TABLE journal (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    identity_json TEXT NOT NULL,
                    completed_rows INTEGER NOT NULL CHECK (completed_rows >= 0)
                )
                """
            )
            database.execute(
                """
                CREATE TABLE scores (
                    row_index INTEGER NOT NULL CHECK (row_index >= 0),
                    column_index INTEGER NOT NULL CHECK (column_index >= 0),
                    value REAL NOT NULL CHECK (typeof(value) = 'real'),
                    PRIMARY KEY (row_index, column_index)
                ) WITHOUT ROWID
                """
            )
            database.execute(
                "INSERT INTO journal VALUES (1, ?, 0)",
                (self._identity_json,),
            )

    def _read_completed(self, database: sqlite3.Connection) -> int:
        try:
            version = database.execute("PRAGMA user_version").fetchone()
            rows = database.execute(
                "SELECT identity_json, completed_rows FROM journal"
            ).fetchall()
        except sqlite3.Error as error:
            raise ValueError("score-journal identity or schema mismatch") from error
        if (
            version != (self.SCHEMA_VERSION,)
            or len(rows) != 1
            or rows[0][0] != self._identity_json
            or type(rows[0][1]) is not int
            or not 0 <= rows[0][1] <= self.spec.rows
        ):
            raise ValueError("score-journal identity or schema mismatch")
        return rows[0][1]

    def _read_scores(
        self,
        database: sqlite3.Connection,
        completed_rows: int,
    ) -> np.ndarray:
        rows = database.execute(
            "SELECT row_index, column_index, value "
            "FROM scores ORDER BY row_index, column_index"
        ).fetchall()
        width = len(self.spec.columns)
        if len(rows) != completed_rows * width:
            raise ValueError("score-journal score payload is not contiguous")
        for offset, (row_index, column_index, value) in enumerate(rows):
            expected_row, expected_column = divmod(offset, width)
            if (
                row_index != expected_row
                or column_index != expected_column
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("invalid score-journal score payload")
        return np.asarray([row[2] for row in rows], dtype=np.float64).reshape(
            completed_rows,
            width,
        )

    @property
    def completed_rows(self) -> int:
        return self._completed_rows

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
        requested_start = observed_start if start is None else start
        with closing(self._connect()) as database, database:
            database.execute("BEGIN IMMEDIATE")
            expected_start = self._read_completed(database)
            self._completed_rows = expected_start
            if type(requested_start) is not int:
                raise ValueError(
                    "score-journal append is not the next contiguous range"
                )
            stop = requested_start + len(values)
            if requested_start != expected_start or stop > self.spec.rows:
                raise ValueError(
                    "score-journal append is not the next contiguous range"
                )
            database.executemany(
                "INSERT INTO scores VALUES (?, ?, ?)",
                (
                    (requested_start + row, column, float(values[row, column]))
                    for row in range(len(values))
                    for column in range(len(self.spec.columns))
                ),
            )
            database.execute(
                "UPDATE journal SET completed_rows = ? WHERE singleton = 1",
                (stop,),
            )

        self._completed_rows = stop
        return requested_start, stop

    def missing_ranges(self, shard_rows: int) -> list[tuple[int, int]]:
        if type(shard_rows) is not int or shard_rows < 1:
            raise ValueError("shard row count must be positive")
        return [
            (start, min(start + shard_rows, self.spec.rows))
            for start in range(self.completed_rows, self.spec.rows, shard_rows)
        ]

    def scores(self) -> np.ndarray:
        with closing(self._connect()) as database:
            completed_rows = self._read_completed(database)
            self._completed_rows = completed_rows
            if completed_rows != self.spec.rows:
                raise ValueError("score journal is incomplete")
            return self._read_scores(database, completed_rows)
