"""Hash-bound retrieval for the maintained advisory reviewer."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from morgott.normalization import strict_normalize
from morgott.sources.tasks import _public_declared_license, _sensitive_text_reasons

_MANIFEST_NAME = "manifest.json"
_VARIANT = "lineage_hybrid_v1"
EMBEDDING_DIMENSION = 256
EMBEDDING_MODEL = "perplexity/pplx-embed-v1-4b"
EMBEDDING_PROVIDER = "perplexity/int8"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_TIMEOUT_SECONDS = 1.0
_EMBEDDING_RESPONSE_MAX_BYTES = 1024 * 1024
_EMBEDDING_BODY = {
    "model": EMBEDDING_MODEL,
    "dimensions": EMBEDDING_DIMENSION,
    "encoding_format": "float",
    "provider": {
        "order": [EMBEDDING_PROVIDER],
        "allow_fallbacks": False,
    },
}
EMBEDDING_REQUEST_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "url": EMBEDDING_URL,
            "timeout_seconds": EMBEDDING_TIMEOUT_SECONDS,
            "body": _EMBEDDING_BODY,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
_CHANNELS = ("direct_user", "untrusted_content")
_LABELS = (0, 1)
_HNSW_PARAMETERS = {
    "m": 32,
    "ef_construction": 200,
    "ef_search": 1_024,
    "overretrieve": 160,
    "exact_rescore": 20,
    "exact_rescore_dtype": "float32",
}
_SPARSE_TOKENIZER = "unicode61 remove_diacritics 2"
_SPARSE_MAX_TERMS = 8
SPARSE_CANDIDATES = 50
_SPARSE_TIMEOUT_MS = 250.0
_MAX_EXAMPLE_BYTES = 1_024
RRF_K = 60
DENSE_RRF_WEIGHT = 2.0
SPARSE_RRF_WEIGHT = 1.0
_EMPTY_RANKINGS: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
_BANK_COLUMNS = (
    "rowid",
    "example_id",
    "text",
    "text_sha256",
    "label",
    "input_channel",
    "source",
    "group_id",
    "subtype",
    "license",
)
_SPARSE_TABLES = {
    (channel, label): f"fts_{channel}_{label}"
    for channel in _CHANNELS
    for label in _LABELS
}


def provider_egress_contract() -> dict[str, Any]:
    """Return the exact eligibility gate applied to remotely sent examples."""
    return {
        "provider_safe": True,
        "license_policy": "public_allowlist_v2",
        "sensitive_text_screen": "morgott.sources.tasks._sensitive_text_reasons",
        "max_example_bytes": _MAX_EXAMPLE_BYTES,
    }


@dataclass(frozen=True)
class RetrievedExample:
    example_id: str
    text: str
    text_sha256: str
    label: int
    input_channel: str
    source: str
    group_id: str

    @property
    def lineage(self) -> tuple[str, str]:
        return self.source, self.group_id


@dataclass(frozen=True)
class SparseResult:
    status: str
    candidate_ids: tuple[tuple[str, ...], tuple[str, ...]]
    elapsed_ms: float
    bundle_sha256: str
    query_sha256: str
    input_channel: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieval decision with in-module stage timings."""

    status: str
    examples: tuple[RetrievedExample, ...]
    fallback_reason: str | None
    dense_ms: float
    sparse_ms: float
    fusion_ms: float


@dataclass(frozen=True)
class RetrievalTrace:
    status: str
    total_ms: float
    embedding_ms: float
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    embedding_input_tokens: int | None
    embedding_cost_usd: float | None
    selected_example_count: int
    selected_packet_sha256: str | None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("retrieval trace status is required")
        timings = (
            self.total_ms,
            self.embedding_ms,
            self.dense_ms,
            self.sparse_ms,
            self.fusion_ms,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in timings
        ):
            raise ValueError("retrieval trace timings must be finite and nonnegative")
        if self.embedding_input_tokens is not None and (
            isinstance(self.embedding_input_tokens, bool)
            or not isinstance(self.embedding_input_tokens, int)
            or self.embedding_input_tokens < 0
        ):
            raise ValueError("embedding token count must be nonnegative")
        if self.embedding_cost_usd is not None and (
            isinstance(self.embedding_cost_usd, bool)
            or not isinstance(self.embedding_cost_usd, (int, float))
            or not math.isfinite(self.embedding_cost_usd)
            or self.embedding_cost_usd < 0
        ):
            raise ValueError("embedding cost must be finite and nonnegative")
        if (
            isinstance(self.selected_example_count, bool)
            or not isinstance(self.selected_example_count, int)
            or self.selected_example_count not in {0, 4}
        ):
            raise ValueError("retrieval selects either zero or four examples")
        if self.selected_example_count == 4:
            if not _valid_sha256(self.selected_packet_sha256):
                raise ValueError("selected retrieval packet hash is invalid")
        elif self.selected_packet_sha256 is not None:
            raise ValueError("empty retrieval cannot have a packet hash")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason
        ):
            raise ValueError("retrieval fallback reason must be a non-empty string")
        if (self.status == "ok") != (self.fallback_reason is None):
            raise ValueError("retrieval fallback reason must match its status")


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    elapsed_ms: float
    input_tokens: int | None
    cost_usd: float | None


class OpenRouterEmbedder:
    """Embed one routed review query through the frozen OpenRouter contract."""

    def __init__(self, api_key: str, *, session: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("OpenRouter API key is required")
        self._api_key = api_key
        self._session = session
        self._owns_session = False

    @classmethod
    def from_env(cls) -> OpenRouterEmbedder:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for query embedding")
        return cls(api_key)

    async def embed(self, text: str) -> EmbeddingResult:
        if not isinstance(text, str) or not text:
            raise ValueError("embedding text must be a non-empty string")
        if self._session is None:
            try:
                import aiohttp
            except ImportError as error:
                raise RuntimeError("install the retrieval extra") from error
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=EMBEDDING_TIMEOUT_SECONDS)
            )
            self._owns_session = True
        started = time.perf_counter()
        body = {**_EMBEDDING_BODY, "input": [text]}
        try:
            async with self._session.post(
                EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Morgott advisory retrieval",
                },
                json=body,
            ) as response:
                raw = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    raw.extend(chunk)
                    if len(raw) > _EMBEDDING_RESPONSE_MAX_BYTES:
                        raise ValueError("query embedding response is too large")
                if response.status != 200:
                    raise RuntimeError("query embedding failed")
                payload = json.loads(raw)
        except Exception as error:
            raise RuntimeError("query embedding failed") from error
        if not isinstance(payload, dict):
            raise RuntimeError("query embedding returned invalid data")
        returned_model = payload.get("model")
        if not isinstance(returned_model, str) or returned_model.casefold() not in {
            EMBEDDING_MODEL.casefold(),
            EMBEDDING_MODEL.rsplit("/", 1)[-1].casefold(),
        }:
            raise RuntimeError("query embedding model identity changed")
        data = payload.get("data")
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
            or data[0].get("index") != 0
        ):
            raise RuntimeError("query embedding returned invalid rows")
        try:
            vector = _normalized_embedding(data[0].get("embedding"))
        except (TypeError, ValueError) as error:
            raise RuntimeError("query embedding returned an invalid vector") from error
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tokens = usage.get("prompt_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise RuntimeError("query embedding returned invalid token usage")
        raw_cost = usage.get("cost")
        cost = None
        if raw_cost is not None:
            try:
                parsed_cost = Decimal(str(raw_cost))
            except (InvalidOperation, ValueError) as error:
                raise RuntimeError("query embedding returned invalid cost") from error
            if not parsed_cost.is_finite() or parsed_cost < 0:
                raise RuntimeError("query embedding returned invalid cost")
            cost = float(parsed_cost)
            if not math.isfinite(cost):
                raise RuntimeError("query embedding returned invalid cost")
        return EmbeddingResult(
            vector=vector,
            elapsed_ms=_elapsed_ms(started),
            input_tokens=tokens,
            cost_usd=cost,
        )

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()


class RetrievalEngine:
    """Load one immutable lineage bundle and return balanced example packets."""

    def __init__(
        self,
        bundle: Path,
        manifest_sha256: str,
        *,
        faiss_module: Any | None = None,
    ) -> None:
        self._bundle = Path(bundle).resolve()
        self._manifest_sha256 = manifest_sha256
        self._available = False
        self._faiss_module: Any | None = None
        self._indexes: dict[tuple[str, int], Any] = {}
        self._rowids: dict[tuple[str, int], np.ndarray] = {}
        self._bank_path = self._bundle / "bank.sqlite3"
        self._sparse_path = self._bundle / "sparse.sqlite3"
        try:
            indexes, rowids, bank_path, sparse_path = self._load(faiss_module)
        except Exception:
            return
        self._indexes = indexes
        self._rowids = rowids
        self._bank_path = bank_path
        self._sparse_path = sparse_path
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    def sparse(self, text: str, input_channel: str) -> SparseResult:
        started = time.perf_counter()
        query_sha256 = _text_sha256(text) if isinstance(text, str) else ""
        if not self._available:
            return self._sparse_failure(
                "invalid", "bundle_unavailable", query_sha256, input_channel, started
            )
        if not isinstance(text, str) or not text or input_channel not in _CHANNELS:
            return self._sparse_failure(
                "invalid", "invalid_query", query_sha256, input_channel, started
            )
        query = _fts_query(text)
        if query is None:
            return self._sparse_failure(
                "fallback", "no_sparse_terms", query_sha256, input_channel, started
            )

        deadline = started + _SPARSE_TIMEOUT_MS / 1_000
        timed_out = False

        def interrupt_after_deadline() -> int:
            nonlocal timed_out
            timed_out = time.perf_counter() >= deadline
            return int(timed_out)

        sparse = bank = None
        try:
            sparse = _read_only_sqlite(self._sparse_path)
            bank = _read_only_sqlite(self._bank_path)
            sparse.set_progress_handler(interrupt_after_deadline, 1_000)
            rankings: list[tuple[str, ...]] = []
            for label in _LABELS:
                table = _SPARSE_TABLES[(input_channel, label)]
                positions = tuple(
                    int(rowid)
                    for (rowid,) in sparse.execute(
                        f"""
                        SELECT rowid FROM {table}
                        WHERE {table} MATCH ?
                        ORDER BY rank, rowid
                        LIMIT ?
                        """,
                        (query, SPARSE_CANDIDATES),
                    )
                )
                if time.perf_counter() >= deadline:
                    raise TimeoutError("sparse deadline exceeded")
                rankings.append(
                    _sparse_example_ids(bank, positions, input_channel, label)
                )
            candidate_ids = (rankings[0], rankings[1])
            if not any(candidate_ids):
                return self._sparse_failure(
                    "fallback",
                    "insufficient_sparse_candidates",
                    query_sha256,
                    input_channel,
                    started,
                )
            return SparseResult(
                status="ok",
                candidate_ids=candidate_ids,
                elapsed_ms=_elapsed_ms(started),
                bundle_sha256=self._manifest_sha256,
                query_sha256=query_sha256,
                input_channel=input_channel,
            )
        except TimeoutError:
            return self._sparse_failure(
                "fallback", "sparse_timeout", query_sha256, input_channel, started
            )
        except sqlite3.OperationalError:
            reason = "sparse_timeout" if timed_out else "sparse_unavailable"
            return self._sparse_failure(
                "fallback", reason, query_sha256, input_channel, started
            )
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return self._sparse_failure(
                "invalid", "sparse_invalid", query_sha256, input_channel, started
            )
        finally:
            if sparse is not None:
                sparse.set_progress_handler(None, 0)
                sparse.close()
            if bank is not None:
                bank.close()

    def retrieve(
        self,
        text: str,
        input_channel: str,
        embedding: Any,
        sparse_result: SparseResult | None,
    ) -> RetrievalResult:
        if not self._available:
            return _empty_result("bundle_unavailable")
        if not isinstance(text, str) or not text or input_channel not in _CHANNELS:
            return _empty_result("invalid_query")
        sparse_binding_failure = (
            sparse_result is not None
            and not self._valid_sparse_result(sparse_result, text, input_channel)
        )
        if sparse_binding_failure:
            sparse_result = None
        try:
            query = _normalized_embedding(embedding)
        except (TypeError, ValueError):
            return _empty_result("invalid_embedding")

        dense_started = time.perf_counter()
        try:
            dense_rankings = self._dense_rank(query, input_channel)
            dense_examples = self._select(dense_rankings, input_channel)
        except Exception:
            return _empty_result("dense_failure")
        dense_ms = _elapsed_ms(dense_started)
        sparse_ms = 0.0 if sparse_result is None else sparse_result.elapsed_ms

        if sparse_result is None or sparse_result.status in {"fallback", "invalid"}:
            return RetrievalResult(
                status="dense_fallback",
                examples=dense_examples,
                fallback_reason=(
                    "invalid_sparse_result"
                    if sparse_binding_failure
                    else "sparse_unavailable"
                    if sparse_result is None
                    else sparse_result.fallback_reason
                ),
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                fusion_ms=0.0,
            )

        fusion_started = time.perf_counter()
        try:
            hybrid_rankings = _rrf(dense_rankings, sparse_result.candidate_ids)
            hybrid_examples = self._select(hybrid_rankings, input_channel)
        except Exception:
            fusion_ms = _elapsed_ms(fusion_started)
            return RetrievalResult(
                status="dense_fallback",
                examples=dense_examples,
                fallback_reason="invalid_balanced_packet",
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                fusion_ms=fusion_ms,
            )
        return RetrievalResult(
            status="ok",
            examples=hybrid_examples,
            fallback_reason=None,
            dense_ms=dense_ms,
            sparse_ms=sparse_ms,
            fusion_ms=_elapsed_ms(fusion_started),
        )

    def _load(
        self, faiss_module: Any | None
    ) -> tuple[
        dict[tuple[str, int], Any],
        dict[tuple[str, int], np.ndarray],
        Path,
        Path,
    ]:
        if not _valid_sha256(self._manifest_sha256):
            raise ValueError("invalid retrieval manifest hash")
        manifest_path = _safe_payload(self._bundle, _MANIFEST_NAME)
        if _file_sha256(manifest_path) != self._manifest_sha256:
            raise ValueError("retrieval manifest hash mismatch")
        if manifest_path.stat().st_size > 1_000_000:
            raise ValueError("retrieval manifest is too large")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files, partitions, bank_spec, sparse_spec = _validate_manifest(manifest)
        paths = {
            name: _verify_payload(self._bundle, record)
            for name, record in files.items()
        }
        bank_path = paths[str(bank_spec["path"])]
        sparse_path = paths[str(sparse_spec["path"])]
        bank = _validate_bank(bank_path, partitions, int(bank_spec["rows"]))
        try:
            _validate_sparse(sparse_path, partitions)
            if faiss_module is None:
                faiss_module = importlib.import_module("faiss")
            _pin_faiss_threads(faiss_module)
            self._faiss_module = faiss_module
            indexes: dict[tuple[str, int], Any] = {}
            rowids: dict[tuple[str, int], np.ndarray] = {}
            for name, spec in partitions.items():
                key = (str(spec["input_channel"]), int(spec["label"]))
                index = faiss_module.read_index(str(paths[str(spec["index_path"])]))
                if (
                    not isinstance(index, faiss_module.IndexHNSWFlat)
                    or int(index.d) != EMBEDDING_DIMENSION
                    or int(index.ntotal) != int(spec["rows"])
                    or int(index.metric_type) != int(faiss_module.METRIC_INNER_PRODUCT)
                    or int(index.hnsw.efConstruction)
                    != _HNSW_PARAMETERS["ef_construction"]
                    or int(index.hnsw.nb_neighbors(1)) != _HNSW_PARAMETERS["m"]
                ):
                    raise ValueError(f"invalid HNSW partition: {name}")
                index.hnsw.efSearch = _HNSW_PARAMETERS["ef_search"]
                mapped = np.load(
                    paths[str(spec["rowids_path"])],
                    allow_pickle=False,
                    mmap_mode="r",
                )
                expected = {
                    int(rowid)
                    for (rowid,) in bank.execute(
                        """
                        SELECT rowid FROM examples
                        WHERE input_channel = ? AND label = ?
                        """,
                        key,
                    )
                }
                if (
                    mapped.dtype != np.dtype("uint32")
                    or mapped.shape != (int(spec["rows"]),)
                    or len(expected) != len(mapped)
                    or expected != {int(value) for value in mapped}
                ):
                    raise ValueError(f"invalid HNSW row map: {name}")
                indexes[key] = index
                rowids[key] = mapped
        finally:
            bank.close()
        return indexes, rowids, bank_path, sparse_path

    def _dense_rank(
        self, query: np.ndarray, input_channel: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self._faiss_module is None:
            raise ValueError("Faiss is unavailable")
        _pin_faiss_threads(self._faiss_module)
        bank = _read_only_sqlite(self._bank_path)
        try:
            rankings = []
            for label in _LABELS:
                index = self._indexes[(input_channel, label)]
                count = min(_HNSW_PARAMETERS["overretrieve"], int(index.ntotal))
                _, raw_positions = index.search(query.reshape(1, -1), count)
                positions = np.asarray(
                    list(
                        dict.fromkeys(
                            int(value)
                            for value in np.asarray(raw_positions).reshape(-1)
                            if int(value) >= 0
                        )
                    ),
                    dtype=np.int64,
                )
                if (
                    len(positions) < 2
                    or np.any(positions >= int(index.ntotal))
                    or np.any(positions < 0)
                ):
                    raise ValueError("HNSW returned invalid positions")
                vectors = np.asarray(
                    index.reconstruct_batch(positions), dtype=np.float32
                )
                if vectors.shape != (len(positions), EMBEDDING_DIMENSION) or not np.all(
                    np.isfinite(vectors)
                ):
                    raise ValueError("HNSW returned invalid vectors")
                candidate_rowids = self._rowids[(input_channel, label)][positions]
                metadata = _metadata_by_rowid(bank, candidate_rowids)
                ranked = []
                for rowid, score in zip(candidate_rowids, vectors @ query, strict=True):
                    row = metadata[int(rowid)]
                    if (row.input_channel, row.label) != (input_channel, label):
                        raise ValueError("HNSW partition identity changed")
                    ranked.append((row.example_id, float(score)))
                ranked.sort(key=lambda value: (-value[1], value[0]))
                rankings.append(
                    tuple(
                        example_id
                        for example_id, _ in ranked[: _HNSW_PARAMETERS["exact_rescore"]]
                    )
                )
            return rankings[0], rankings[1]
        finally:
            bank.close()

    def _select(
        self,
        rankings: tuple[tuple[str, ...], tuple[str, ...]],
        input_channel: str,
    ) -> tuple[RetrievedExample, ...]:
        bank = _read_only_sqlite(self._bank_path)
        try:
            ids = tuple(dict.fromkeys(rankings[0] + rankings[1]))
            metadata = _metadata_by_example_id(bank, ids)
        finally:
            bank.close()
        choices = {
            label: tuple(metadata[example_id] for example_id in rankings[label])
            for label in _LABELS
        }
        selected: list[RetrievedExample] = []
        sources: set[str] = set()
        lineages: set[tuple[str, str]] = set()
        for _ in range(2):
            for label in _LABELS:
                eligible = tuple(
                    row
                    for row in choices[label]
                    if row.input_channel == input_channel
                    and row.label == label
                    and row.lineage not in lineages
                    and row not in selected
                )
                choice = next(
                    (row for row in eligible if row.source not in sources),
                    eligible[0] if eligible else None,
                )
                if choice is None:
                    raise ValueError("retrieval did not produce a balanced packet")
                if _text_sha256(choice.text) != choice.text_sha256:
                    raise ValueError("retrieved example text hash changed")
                selected.append(choice)
                sources.add(choice.source)
                lineages.add(choice.lineage)
        return tuple(selected)

    def _valid_sparse_result(
        self, sparse_result: SparseResult, text: str, input_channel: str
    ) -> bool:
        valid_rankings = (
            isinstance(sparse_result, SparseResult)
            and len(sparse_result.candidate_ids) == 2
            and all(
                isinstance(ranking, tuple)
                and len(ranking) <= SPARSE_CANDIDATES
                and len(set(ranking)) == len(ranking)
                and all(
                    isinstance(example_id, str) and example_id for example_id in ranking
                )
                for ranking in sparse_result.candidate_ids
            )
        )
        return (
            isinstance(sparse_result, SparseResult)
            and sparse_result.status in {"ok", "fallback", "invalid"}
            and sparse_result.bundle_sha256 == self._manifest_sha256
            and sparse_result.query_sha256 == _text_sha256(text)
            and sparse_result.input_channel == input_channel
            and sparse_result.elapsed_ms >= 0
            and math.isfinite(sparse_result.elapsed_ms)
            and valid_rankings
            and (
                (
                    sparse_result.status == "ok"
                    and any(sparse_result.candidate_ids)
                    and sparse_result.fallback_reason is None
                )
                or (
                    sparse_result.status != "ok"
                    and sparse_result.candidate_ids == _EMPTY_RANKINGS
                    and bool(sparse_result.fallback_reason)
                )
            )
        )

    def _sparse_failure(
        self,
        status: str,
        reason: str,
        query_sha256: str,
        input_channel: str,
        started: float,
    ) -> SparseResult:
        return SparseResult(
            status=status,
            candidate_ids=_EMPTY_RANKINGS,
            elapsed_ms=_elapsed_ms(started),
            bundle_sha256=self._manifest_sha256,
            query_sha256=query_sha256,
            input_channel=input_channel,
            fallback_reason=reason,
        )


def _validate_manifest(
    manifest: Any,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    expected_partitions = {
        f"{channel}-{label}" for channel in _CHANNELS for label in _LABELS
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("variant") != _VARIANT
        or manifest.get("parameters") != _HNSW_PARAMETERS
        or set(manifest.get("partitions", {})) != expected_partitions
    ):
        raise ValueError("retrieval manifest contract changed")
    dense = manifest.get("dense", {})
    if any(
        dense.get(key) != value
        for key, value in {
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "input_type": None,
            "metric": "inner_product",
        }.items()
    ):
        raise ValueError("dense embedding contract changed")
    source = manifest.get("source", {})
    if (
        not _valid_sha256(source.get("data_manifest_sha256"))
        or not _valid_sha256(source.get("routing_view_sha256"))
        or manifest.get("provider_egress") != provider_egress_contract()
    ):
        raise ValueError("provider egress contract changed")
    bank = manifest.get("bank", {})
    sparse = manifest.get("sparse", {})
    bank_rows = bank.get("rows")
    if (
        not isinstance(bank_rows, int)
        or isinstance(bank_rows, bool)
        or bank_rows < 1
        or bank.get("mode") != "full_lineage"
        or sparse.get("bank_sha256") != bank.get("sha256")
        or sparse.get("tokenizer") != _SPARSE_TOKENIZER
        or sparse.get("contentless") is not True
        or sparse.get("maximum_terms") != _SPARSE_MAX_TERMS
        or sparse.get("candidates_per_label") != SPARSE_CANDIDATES
        or sparse.get("timeout_ms") != _SPARSE_TIMEOUT_MS
    ):
        raise ValueError("retrieval bank contract changed")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 10:
        raise ValueError("retrieval payload list changed")
    files: dict[str, dict[str, Any]] = {}
    roles: Counter[str] = Counter()
    for record in raw_files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "bytes",
            "role",
        }:
            raise ValueError("invalid retrieval payload record")
        path = record["path"]
        if not isinstance(path, str) or path in files:
            raise ValueError("duplicate retrieval payload")
        files[path] = record
        roles[str(record["role"])] += 1
    if roles != Counter({"index": 4, "row_map": 4, "bank": 1, "sparse": 1}):
        raise ValueError("retrieval payload roles changed")

    expected_files = {str(bank.get("path")), str(sparse.get("path"))}
    partitions = manifest["partitions"]
    total_rows = 0
    for name, spec in partitions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"invalid retrieval partition: {name}")
        channel = spec.get("input_channel")
        label = spec.get("label")
        if name != f"{channel}-{label}" or (channel, label) not in _SPARSE_TABLES:
            raise ValueError(f"invalid retrieval partition identity: {name}")
        if (
            not isinstance(spec.get("rows"), int)
            or isinstance(spec["rows"], bool)
            or spec["rows"] < 2
            or spec.get("rowids_dtype") != "uint32"
        ):
            raise ValueError(f"invalid retrieval partition rows: {name}")
        total_rows += spec["rows"]
        expected_files.update(
            (str(spec.get("index_path")), str(spec.get("rowids_path")))
        )
        _match_spec(files, spec, "index", "index")
        _match_spec(files, spec, "rowids", "row_map")
    _match_spec(files, bank, "", "bank")
    _match_spec(files, sparse, "", "sparse")
    if set(files) != expected_files or total_rows != bank_rows:
        raise ValueError("retrieval payload membership changed")
    return files, partitions, bank, sparse


def _match_spec(
    files: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    prefix: str,
    role: str,
) -> None:
    key = f"{prefix}_" if prefix else ""
    path = spec.get(f"{key}path")
    record = files.get(path)
    if (
        record is None
        or record["role"] != role
        or record["sha256"] != spec.get(f"{key}sha256")
        or record["bytes"] != spec.get(f"{key}bytes")
    ):
        raise ValueError(f"retrieval {role} payload changed")


def _validate_bank(
    path: Path, partitions: dict[str, dict[str, Any]], expected_rows: int
) -> sqlite3.Connection:
    connection = _read_only_sqlite(path)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("retrieval bank integrity check failed")
        columns = tuple(
            str(row[1]) for row in connection.execute("PRAGMA table_info(examples)")
        )
        if columns != _BANK_COLUMNS:
            raise ValueError("retrieval bank schema changed")
        if connection.execute("SELECT COUNT(*) FROM examples").fetchone() != (
            expected_rows,
        ):
            raise ValueError("retrieval bank row count changed")
        for text, license_name in connection.execute(
            "SELECT text, license FROM examples"
        ):
            if (
                not isinstance(text, str)
                or len(text.encode()) > _MAX_EXAMPLE_BYTES
                or not _public_declared_license(license_name)
                or _sensitive_text_reasons(text)
            ):
                raise ValueError("retrieval bank provider egress gate changed")
        for spec in partitions.values():
            count = connection.execute(
                """
                SELECT COUNT(*) FROM examples
                WHERE input_channel = ? AND label = ?
                """,
                (spec["input_channel"], spec["label"]),
            ).fetchone()
            if count != (spec["rows"],):
                raise ValueError("retrieval bank partition count changed")
        return connection
    except BaseException:
        connection.close()
        raise


def _pin_faiss_threads(faiss_module: Any) -> None:
    faiss_module.omp_set_num_threads(1)
    get_max_threads = getattr(faiss_module, "omp_get_max_threads", None)
    if callable(get_max_threads) and int(get_max_threads()) != 1:
        raise ValueError("Faiss native thread limit changed")


def _validate_sparse(path: Path, partitions: dict[str, dict[str, Any]]) -> None:
    connection = _read_only_sqlite(path)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("sparse index integrity check failed")
        definitions = {
            str(name): str(sql)
            for name, sql in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
            )
        }
        for spec in partitions.values():
            table = _SPARSE_TABLES[(spec["input_channel"], spec["label"])]
            sql = "".join(definitions.get(table, "").lower().split())
            if (
                "usingfts5(" not in sql
                or "content=''" not in sql
                or "tokenize='unicode61remove_diacritics2'" not in sql
                or connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                != (spec["rows"],)
            ):
                raise ValueError(f"sparse partition changed: {table}")
    finally:
        connection.close()


def _safe_payload(bundle: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or relative.parts != (name,)
        or relative.name != name
    ):
        raise ValueError("retrieval payload path must be a safe basename")
    path = bundle / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("retrieval payload is missing or not a regular file")
    return path


def _verify_payload(bundle: Path, record: dict[str, Any]) -> Path:
    path = _safe_payload(bundle, str(record["path"]))
    expected_bytes = record["bytes"]
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 1
        or path.stat().st_size != expected_bytes
        or not _valid_sha256(record["sha256"])
        or _file_sha256(path) != record["sha256"]
    ):
        raise ValueError("retrieval payload identity changed")
    return path


def _read_only_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _fts_query(text: str) -> str | None:
    terms = []
    for raw in strict_normalize(text).split():
        term = raw.strip('"').replace('"', '""')[:64]
        if len(term) >= 3 and term not in terms:
            terms.append(term)
        if len(terms) == _SPARSE_MAX_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in terms) if terms else None


def _sparse_example_ids(
    bank: sqlite3.Connection,
    rowids: tuple[int, ...],
    input_channel: str,
    label: int,
) -> tuple[str, ...]:
    if not rowids:
        return ()
    metadata = _metadata_by_rowid(bank, rowids)
    seen: set[tuple[str, str]] = set()
    result = []
    for rowid in rowids:
        row = metadata[rowid]
        if (row.input_channel, row.label) != (input_channel, label):
            raise ValueError("sparse partition identity changed")
        if row.lineage in seen:
            continue
        seen.add(row.lineage)
        result.append(row.example_id)
    return tuple(result)


def _metadata_by_rowid(
    bank: sqlite3.Connection, rowids: Any
) -> dict[int, RetrievedExample]:
    values = tuple(int(value) for value in rowids)
    if not values:
        return {}
    placeholders = ",".join("?" for _ in values)
    rows = {
        int(row[0]): _example(row[1:])
        for row in bank.execute(
            f"""
            SELECT rowid, example_id, text, text_sha256, label, input_channel,
                   source, group_id
            FROM examples WHERE rowid IN ({placeholders})
            """,
            values,
        )
    }
    if set(rows) != set(values):
        raise ValueError("retrieval row map returned an unknown bank row")
    return rows


def _metadata_by_example_id(
    bank: sqlite3.Connection, example_ids: tuple[str, ...]
) -> dict[str, RetrievedExample]:
    if not example_ids:
        return {}
    placeholders = ",".join("?" for _ in example_ids)
    rows = {
        str(row[0]): _example(row)
        for row in bank.execute(
            f"""
            SELECT example_id, text, text_sha256, label, input_channel,
                   source, group_id
            FROM examples WHERE example_id IN ({placeholders})
            """,
            example_ids,
        )
    }
    if set(rows) != set(example_ids):
        raise ValueError("retrieval ranking returned an unknown example")
    return rows


def _example(row: Any) -> RetrievedExample:
    return RetrievedExample(
        example_id=str(row[0]),
        text=str(row[1]),
        text_sha256=str(row[2]),
        label=int(row[3]),
        input_channel=str(row[4]),
        source=str(row[5]),
        group_id=str(row[6]),
    )


def _rrf(
    dense: tuple[tuple[str, ...], tuple[str, ...]],
    sparse: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    result = []
    for label in _LABELS:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranking, weight in (
            (dense[label], DENSE_RRF_WEIGHT),
            (sparse[label], SPARSE_RRF_WEIGHT),
        ):
            for rank, example_id in enumerate(ranking, 1):
                scores[example_id] += weight / (RRF_K + rank)
        result.append(
            tuple(
                example_id
                for example_id, _ in sorted(
                    scores.items(), key=lambda value: (-value[1], value[0])
                )[:SPARSE_CANDIDATES]
            )
        )
    return result[0], result[1]


def _normalized_embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ValueError("invalid query embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid query embedding norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _empty_result(reason: str) -> RetrievalResult:
    return RetrievalResult(
        status="no_examples",
        examples=(),
        fallback_reason=reason,
        dense_ms=0.0,
        sparse_ms=0.0,
        fusion_ms=0.0,
    )


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1_000)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
