#!/usr/bin/env python3
"""Build a bounded, deterministic WildChat weak-label candidate sample."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import http.client
import json
import re
import socket
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


DATASET = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
TOTAL_ROWS = 837_989
TOTAL_BYTES = 3_360_836_020
REPO_ENDPOINT = "https://huggingface.co/api/datasets/allenai/WildChat-1M"
DEFAULT_SEED = "vulsight-wildchat-pilot-v1"
CACHE_SCHEMA_VERSION = 2
SHARDS = (
    {
        "index": 0,
        "path": "data/train-00000-of-00014.parquet",
        "size": 230_786_095,
        "sha256": "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c",
        "row_offset": 0,
        "rows": 59_857,
    },
    {
        "index": 6,
        "path": "data/train-00006-of-00014.parquet",
        "size": 189_507_683,
        "sha256": "0bd2fff31c74feb6b44b0397b4e7ab6285dd69ccbe4cc8b5f19c42cdbf691303",
        "row_offset": 359_141,
        "rows": 59_856,
    },
    {
        "index": 13,
        "path": "data/train-00013-of-00014.parquet",
        "size": 336_039_603,
        "sha256": "440be579bf012a6e98d179b8dd0bedd73634e5527f7000c067d19c88b5cefa6c",
        "row_offset": 778_133,
        "rows": 59_856,
    },
)
_CANDIDATE_FIELDS = {
    "sample_id",
    "text",
    "text_sha256",
    "source_row_index",
    "conversation_sha256",
    "user_turn_index",
    "language",
    "length_bucket",
    "source_toxic",
    "topic",
    "security_trigger",
    "local_redactions",
    "truncated",
}

_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b((?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|access[_-]?token|"
    r"auth(?:orization)?|password|passwd|secret(?:[_-]?access)?[_-]?key|secret|token))"
    r"(\s*[:=]\s*)([\"']?)[^\s,;\"']{8,}([\"']?)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}=*")
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-(?:proj-)?|hf_|gh[pousr]_|xox[baprs]-|AKIA)[A-Za-z0-9_-]{8,}\b"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_URL = re.compile(r"(?i)https?://[^\s<>\[\]\"']+")
_SENSITIVE_QUERY_VALUE = re.compile(
    r"(?i)([?&](?:access[_-]?token|api[_-]?key|auth|authorization|code|"
    r"credential|jwt|key|password|passwd|secret|session(?:id)?|sid|sig|sign|"
    r"signature|token|x-amz-(?:credential|security-token|signature))=)"
    r"[^&#\s<>\"']+"
)
_SECURITY = re.compile(
    r"(?i)\b(?:ignore (?:all |any )?(?:previous|prior|above)|system prompt|"
    r"developer message|jailbreak|prompt injection|DAN\b|bypass (?:the )?(?:rules|"
    r"filter|safety)|reveal (?:your |the )?(?:instructions|prompt)|do not obey|"
    r"disregard (?:the )?(?:instructions|rules))"
)

_TOPICS = {
    "security": re.compile(
        r"(?i)\b(?:security|cyber|malware|phishing|exploit|vulnerab|prompt injection|jailbreak)"
    ),
    "code": re.compile(
        r"(?i)\b(?:python|javascript|typescript|java\b|c\+\+|sql\b|regex|function|code|debug|api\b)"
    ),
    "health": re.compile(
        r"(?i)\b(?:health|medical|doctor|symptom|diagnos|medicine|therapy|mental health)"
    ),
    "finance": re.compile(
        r"(?i)\b(?:finance|money|stock|crypto|invest|tax\b|bank|loan|budget)"
    ),
    "education": re.compile(
        r"(?i)\b(?:homework|school|student|teacher|study|learn|exam|essay|research)"
    ),
    "creative": re.compile(
        r"(?i)\b(?:story|poem|song|character|roleplay|role-play|novel|creative|fiction)"
    ),
    "relationships": re.compile(
        r"(?i)\b(?:relationship|girlfriend|boyfriend|husband|wife|dating|friend|family)"
    ),
}

_LANGUAGES = {
    "english": "English",
    "chinese": "Chinese",
    "spanish": "Spanish",
    "russian": "Russian",
    "french": "French",
    "german": "German",
    "portuguese": "Portuguese",
    "japanese": "Japanese",
    "korean": "Korean",
    "arabic": "Arabic",
}


def sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def redact(text: str, max_chars: int = 12_000) -> tuple[str, int, bool]:
    """Remove common PII/secret shapes locally; return the exact judged text."""
    replacements = 0

    def replace(pattern: re.Pattern[str], value: str, replacement: str) -> str:
        nonlocal replacements
        value, count = pattern.subn(replacement, value)
        replacements += count
        return value

    value = replace(_PRIVATE_KEY, text, "[REDACTED_PRIVATE_KEY]")
    value = replace(_ASSIGNED_SECRET, value, r"\1\2[REDACTED]")
    value = replace(_BEARER, value, "Bearer [REDACTED]")
    value = replace(_KNOWN_TOKEN, value, "[REDACTED_TOKEN]")
    value = replace(_JWT, value, "[REDACTED_TOKEN]")

    def redact_url(match: re.Match[str]) -> str:
        nonlocal replacements
        raw = match.group(0)
        parsed = urllib.parse.urlsplit(raw)
        netloc = parsed.netloc
        changed = False
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[1]
            changed = True
        query = parsed.query
        fragment = parsed.fragment
        if query:
            query = "[REDACTED_QUERY]"
            changed = True
        if fragment:
            fragment = "[REDACTED_FRAGMENT]"
            changed = True
        if not changed:
            return raw
        replacements += 1
        return urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, query, fragment)
        )

    value = _URL.sub(redact_url, value)
    value = replace(_SENSITIVE_QUERY_VALUE, value, r"\1[REDACTED]")
    value = replace(_EMAIL, value, "[REDACTED_EMAIL]")
    value = replace(_PHONE, value, "[REDACTED_PHONE]")
    value = replace(_IPV4, value, "[REDACTED_IP]")
    truncated = len(value) > max_chars
    if truncated:
        suffix = "\n[TRUNCATED]"
        value = value[: max_chars - len(suffix)] + suffix
    return value.strip(), replacements, truncated


def simhash(text: str) -> int:
    normalized = normalize(text)
    grams = {normalized[i : i + 4] for i in range(max(1, len(normalized) - 3))}
    totals = [0] * 64
    for gram in grams:
        value = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest())
        for bit in range(64):
            totals[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, total in enumerate(totals) if total >= 0)


class NearIndex:
    """Eight 8-bit bands make every <=7-bit SimHash neighbor a candidate."""

    def __init__(self) -> None:
        self._bands: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add(self, value: int) -> None:
        for band in range(8):
            self._bands[(band, (value >> (8 * band)) & 0xFF)].append(value)

    def contains(self, value: int, distance: int = 6) -> bool:
        candidates: set[int] = set()
        for band in range(8):
            candidates.update(self._bands.get((band, (value >> (8 * band)) & 0xFF), ()))
        return any(
            (value ^ candidate).bit_count() <= distance for candidate in candidates
        )


def language_group(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    lowered = value.strip().casefold()
    return _LANGUAGES.get(lowered, "Other")


def length_bucket(text: str) -> str:
    length = len(text)
    if length < 160:
        return "short"
    if length < 800:
        return "medium"
    if length < 3_000:
        return "long"
    return "very_long"


def topic(text: str) -> str:
    for name, pattern in _TOPICS.items():
        if pattern.search(text):
            return name
    return "general"


def source_toxicity(raw: dict, turn: dict, conversation_index: int) -> bool | str:
    """Return a source-derived flag, or unavailable instead of invented false."""
    available = False
    toxic = False
    for value in (turn.get("toxic"), raw.get("toxic")):
        if isinstance(value, bool):
            available = True
            toxic = toxic or value
    openai_moderation = raw.get("openai_moderation")
    if isinstance(openai_moderation, list) and conversation_index < len(
        openai_moderation
    ):
        moderation = openai_moderation[conversation_index]
        if isinstance(moderation, dict) and isinstance(moderation.get("flagged"), bool):
            available = True
            toxic = toxic or moderation["flagged"]
    detoxify = raw.get("detoxify_moderation")
    if isinstance(detoxify, list) and conversation_index < len(detoxify):
        moderation = detoxify[conversation_index]
        if isinstance(moderation, dict):
            score = moderation.get("toxicity")
            if isinstance(score, (int, float)):
                available = True
                toxic = toxic or score >= 0.5
    return toxic if available else "unavailable"


def extract_candidate(raw: dict, row_index: int, seed: str) -> dict | None:
    conversation = raw.get("conversation")
    if not isinstance(conversation, list):
        return None
    user_turns = [
        (index, turn)
        for index, turn in enumerate(conversation)
        if isinstance(turn, dict)
        and turn.get("role") == "user"
        and isinstance(turn.get("content"), str)
    ]
    if not user_turns:
        return None
    conversation_hash = str(raw.get("conversation_hash", row_index))
    selected = int(sha256(f"{seed}\0{conversation_hash}"), 16) % len(user_turns)
    conversation_index, turn = user_turns[selected]
    text, redactions, truncated = redact(turn["content"])
    if len(normalize(text)) < 16:
        return None
    lang = language_group(turn.get("language") or raw.get("language"))
    source_toxic = source_toxicity(raw, turn, conversation_index)
    sample_id = sha256(f"{REVISION}\0{row_index}\0{conversation_hash}\0{selected}")
    return {
        "sample_id": sample_id,
        "text": text,
        "text_sha256": sha256(text),
        "source_row_index": row_index,
        "conversation_sha256": sha256(conversation_hash),
        "user_turn_index": conversation_index,
        "language": lang,
        "length_bucket": length_bucket(text),
        "source_toxic": source_toxic,
        "topic": topic(text),
        "security_trigger": bool(_SECURITY.search(text)),
        "local_redactions": redactions,
        "truncated": truncated,
    }


def stratified_select(candidates: list[dict], count: int, seed: str) -> list[dict]:
    """Diversity-weighted round robin over fixed joint strata."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        key = (
            row["language"],
            row["length_bucket"],
            row["source_toxic"],
            row["topic"],
            row["security_trigger"],
        )
        strata[key].append(row)
    for key, rows in strata.items():
        rows.sort(key=lambda row: sha256(f"{seed}\0{key}\0{row['sample_id']}"))
    keys = sorted(strata, key=lambda key: sha256(f"{seed}\0{key}"))
    selected: list[dict] = []
    index = 0
    while len(selected) < count and keys:
        key = keys[index % len(keys)]
        rows = strata[key]
        if rows:
            selected.append(rows.pop())
            index += 1
        else:
            keys.remove(key)
            if keys:
                index %= len(keys)
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible rows; need {count}")
    return sorted(selected, key=lambda row: row["sample_id"])


def prepare_source_toxicity_stratum(rows: list[dict]) -> str:
    """Disable a nominal source stratum when the sampled source has no variation."""
    values = {
        row["source_toxic"] for row in rows if isinstance(row["source_toxic"], bool)
    }
    if values == {False, True}:
        return "available"
    for row in rows:
        row["source_toxic"] = "unavailable"
    return "unavailable_no_source_variation" if values else "unavailable_missing"


def _get_json(url: str, timeout: float = 30) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "morgott-wildchat/1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            if attempt == 4:
                raise
            delay = (
                float(retry_after) if retry_after and retry_after.isdigit() else 10.0
            )
            time.sleep(min(30.0, max(1.0, delay)))
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(min(4, 2**attempt))
    if not isinstance(value, dict):
        raise ValueError("unexpected JSON response")
    return value


def current_revision() -> str:
    value = _get_json(REPO_ENDPOINT)
    revision = value.get("sha")
    if not isinstance(revision, str):
        raise ValueError("dataset repository did not return a revision")
    return revision


def shard_url(shard: dict) -> str:
    path = urllib.parse.quote(shard["path"], safe="/")
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"


class DownloadIntegrityError(Exception):
    pass


def download_shard(shard: dict, target: Path) -> int:
    """Download one exact-revision shard and verify its LFS size and SHA-256."""
    retries = 0
    for attempt in range(3):
        digest = hashlib.sha256()
        downloaded = 0
        request = urllib.request.Request(
            shard_url(shard),
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "morgott-wildchat/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with target.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > shard["size"]:
                            raise DownloadIntegrityError("shard exceeded pinned size")
                        digest.update(chunk)
                        handle.write(chunk)
            if downloaded != shard["size"] or digest.hexdigest() != shard["sha256"]:
                raise DownloadIntegrityError("shard size or SHA-256 mismatch")
            return retries
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            exc.close()
            if not retryable or attempt == 2:
                raise
        except (
            DownloadIntegrityError,
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
            http.client.IncompleteRead,
        ):
            if attempt == 2:
                raise
        retries += 1
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def extract_shard_candidates(
    path: Path, shard: dict, seed: str, candidate_limit: int
) -> tuple[list[dict], dict]:
    """Read one ephemeral shard and retain its deterministic lowest hash ranks."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != shard["rows"]:
        raise ValueError(f"unexpected row count in {shard['path']}")
    columns = [
        "conversation_hash",
        "conversation",
        "language",
        "openai_moderation",
        "detoxify_moderation",
        "toxic",
    ]
    heap: list[tuple[int, str, dict]] = []
    rows_without_user_text = 0
    user_turn_candidates = 0
    local_index = 0
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        for raw in batch.to_pylist():
            row_index = shard["row_offset"] + local_index
            local_index += 1
            candidate = extract_candidate(raw, row_index, seed)
            if candidate is None:
                rows_without_user_text += 1
                continue
            user_turn_candidates += 1
            rank = int(sha256(f"{seed}\0pool\0{candidate['sample_id']}"), 16)
            item = (-rank, candidate["sample_id"], candidate)
            if len(heap) < candidate_limit:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    if local_index != shard["rows"]:
        raise ValueError(f"did not scan every row in {shard['path']}")
    rows = sorted((item[2] for item in heap), key=lambda row: row["sample_id"])
    return rows, {
        "source_rows_scanned": local_index,
        "rows_without_user_text": rows_without_user_text,
        "user_turn_candidates": user_turn_candidates,
        "retained_candidates": len(rows),
    }


def validate_cache(value: dict, shard: dict, seed: str, candidate_limit: int) -> None:
    if (
        value.get("schema_version") != CACHE_SCHEMA_VERSION
        or value.get("dataset_revision") != REVISION
        or value.get("seed") != seed
        or value.get("shard") != shard
        or value.get("candidate_limit") != candidate_limit
        or not isinstance(value.get("stats"), dict)
        or value["stats"].get("source_rows_scanned") != shard["rows"]
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) > candidate_limit
    ):
        raise ValueError("invalid sanitized shard cache metadata")
    sample_ids = set()
    for row in value["rows"]:
        if (
            not isinstance(row, dict)
            or set(row) != _CANDIDATE_FIELDS
            or not isinstance(row.get("text"), str)
            or sha256(row["text"]) != row.get("text_sha256")
            or row.get("sample_id") in sample_ids
            or not (
                shard["row_offset"]
                <= row.get("source_row_index", -1)
                < shard["row_offset"] + shard["rows"]
            )
            or not (
                isinstance(row.get("source_toxic"), bool)
                or row.get("source_toxic") == "unavailable"
            )
        ):
            raise ValueError("invalid sanitized shard cache row")
        sample_ids.add(row["sample_id"])


def cached_candidates(
    shard: dict, seed: str, cache_dir: Path, candidate_limit: int
) -> tuple[list[dict], dict, bool, int]:
    """Persist only a bounded, locally redacted candidate pool, never raw source."""
    path = cache_dir / f"shard-{shard['index']:05d}.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate_cache(value, shard, seed, candidate_limit)
        except ValueError as exc:
            raise ValueError(f"{exc}: {path}") from exc
        return value["rows"], value["stats"], True, 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"wildchat-{shard['index']:05d}-", suffix=".parquet"
    ) as temporary:
        retries = download_shard(shard, Path(temporary.name))
        rows, stats = extract_shard_candidates(
            Path(temporary.name), shard, seed, candidate_limit
        )
    value = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_revision": REVISION,
        "seed": seed,
        "shard": shard,
        "candidate_limit": candidate_limit,
        "stats": stats,
        "rows": rows,
    }
    validate_cache(value, shard, seed, candidate_limit)
    temporary_cache = path.with_suffix(".tmp")
    temporary_cache.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_cache.replace(path)
    return rows, stats, False, retries


def load_reference_index(processed: Path) -> tuple[set[str], NearIndex, int]:
    exact: set[str] = set()
    near = NearIndex()
    rows = 0
    for path in sorted(processed.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("text")
                if not isinstance(text, str):
                    continue
                redacted, _, _ = redact(text)
                normalized = normalize(redacted)
                exact.add(normalized)
                if len(normalized) >= 80:
                    near.add(simhash(normalized))
                rows += 1
    if not rows:
        raise ValueError(f"no processed reference rows found under {processed}")
    return exact, near, rows


def filter_candidates(
    raw_candidates: list[dict], reference_exact: set[str], reference_near: NearIndex
) -> tuple[list[dict], Counter]:
    kept = []
    dropped: Counter = Counter()
    pilot_exact: set[str] = set()
    pilot_conversations: set[str] = set()
    pilot_near = NearIndex()
    for row in raw_candidates:
        if row["conversation_sha256"] in pilot_conversations:
            dropped["pilot_conversation_duplicate"] += 1
            continue
        normalized = normalize(row["text"])
        if normalized in reference_exact:
            dropped["exact_reference_overlap"] += 1
            continue
        value = simhash(normalized)
        if len(normalized) >= 80 and reference_near.contains(value):
            dropped["near_reference_overlap"] += 1
            continue
        if normalized in pilot_exact:
            dropped["pilot_exact_duplicate"] += 1
            continue
        if len(normalized) >= 80 and pilot_near.contains(value):
            dropped["pilot_near_duplicate"] += 1
            continue
        pilot_exact.add(normalized)
        pilot_conversations.add(row["conversation_sha256"])
        if len(normalized) >= 80:
            pilot_near.add(value)
        kept.append(row)
    return kept, dropped


def attach_detector_scores(rows: list[dict], artifact_path: Path) -> float:
    import joblib

    artifact = joblib.load(artifact_path)
    sensor = artifact["channels"]["direct_user"]
    texts = [normalize(row["text"]) for row in rows]
    scores = sensor["model"].predict_proba(texts)[:, 1]
    threshold = float(sensor["threshold"])
    for row, score in zip(rows, scores, strict=True):
        row["detector_score"] = round(float(score), 8)
        row["detector_elevated"] = bool(score >= threshold)
    return threshold


def distribution(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def execute(args: argparse.Namespace) -> dict:
    cache_only = all(
        (args.candidate_cache_dir / f"shard-{shard['index']:05d}.json").exists()
        for shard in SHARDS
    )
    before = None if cache_only else current_revision()
    reference_exact, reference_near, reference_rows = load_reference_index(
        args.processed_dir
    )
    extracted = []
    shard_stats = []
    reused_shards = 0
    download_retries = 0
    verified_download_bytes = 0
    for shard in SHARDS:
        candidates, stats, reused, retries = cached_candidates(
            shard, args.seed, args.candidate_cache_dir, args.candidates_per_shard
        )
        extracted.extend(candidates)
        reused_shards += int(reused)
        download_retries += retries
        if not reused:
            verified_download_bytes += shard["size"]
        shard_stats.append({**shard, **stats, "cache_reused": reused})
    source_toxicity_status = prepare_source_toxicity_stratum(extracted)
    if reused_shards != len(SHARDS):
        after = current_revision()
        if after != before:
            raise ValueError("WildChat repository head changed during sampling")
    eligible, dropped = filter_candidates(extracted, reference_exact, reference_near)
    selected = stratified_select(eligible, args.sample_size, args.seed)
    detector_threshold = attach_detector_scores(selected, args.artifact)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_sha = sha256(args.output.read_bytes())
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "repository_head_observed": before,
        "revision_verification": (
            "validated sanitized cache revision and pinned source LFS hashes"
            if cache_only
            else "repository head stable; exact-revision source LFS size and SHA-256 verified"
        ),
        "dataset_rows": TOTAL_ROWS,
        "dataset_bytes": TOTAL_BYTES,
        "sampling": {
            "seed": args.seed,
            "source_mode": "exact-revision Parquet shards",
            "selected_shards": shard_stats,
            "selected_source_bytes": sum(shard["size"] for shard in SHARDS),
            "source_bytes_not_selected": TOTAL_BYTES
            - sum(shard["size"] for shard in SHARDS),
            "verified_parquet_bytes_downloaded_this_run": verified_download_bytes,
            "maximum_network_bytes_with_three_attempts": sum(
                shard["size"] for shard in SHARDS
            )
            * 3,
            "download_retries": download_retries,
            "raw_parquet_lifetime": "one shard at a time in an ephemeral temporary file",
            "raw_source_persisted": False,
            "candidates_per_shard": args.candidates_per_shard,
            "sanitized_candidate_cache": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "directory": portable_path(args.candidate_cache_dir),
                "maximum_rows": len(SHARDS) * args.candidates_per_shard,
                "reused_shards": reused_shards,
                "written_shards": len(SHARDS) - reused_shards,
                "raw_source_metadata_cached": False,
            },
            "source_rows_scanned": sum(
                stats["source_rows_scanned"] for stats in shard_stats
            ),
            "user_turn_candidates_before_pool_cap": sum(
                stats["user_turn_candidates"] for stats in shard_stats
            ),
            "bounded_candidates": len(extracted),
            "rows_without_user_text": sum(
                stats["rows_without_user_text"] for stats in shard_stats
            ),
            "reference_rows_checked": reference_rows,
            "filtered_by_reason": dict(sorted(dropped.items())),
            "eligible_before_stratification": len(eligible),
            "sampled_rows": len(selected),
            "strategy": "diversity-weighted round robin over joint safe strata",
        },
        "privacy": {
            "metadata_retained": [
                "source_row_index",
                "conversation_sha256",
                "user_turn_index",
            ],
            "metadata_dropped": [
                "country",
                "hashed_ip",
                "header",
                "timestamp",
                "model",
                "state",
                "moderation_scores",
            ],
            "rows_with_local_redactions": sum(
                row["local_redactions"] > 0 for row in selected
            ),
            "rows_truncated": sum(row["truncated"] for row in selected),
            "raw_text_in_report": False,
            "source_toxicity_stratum": source_toxicity_status,
        },
        "detector": {
            "artifact": portable_path(args.artifact),
            "artifact_sha256": sha256(args.artifact.read_bytes()),
            "threshold": detector_threshold,
            "elevated_rows": sum(row["detector_elevated"] for row in selected),
        },
        "strata": {
            field: distribution(selected, field)
            for field in (
                "language",
                "length_bucket",
                "source_toxic",
                "topic",
                "security_trigger",
            )
        },
        "output": {
            "path": portable_path(args.output),
            "sha256": output_sha,
            "contains_local_redacted_text": True,
            "git_ignored": True,
        },
        "label_status": "not_run",
        "metric_status": "weak-training-sample; never production FPR",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--candidates-per-shard", type=int, default=10_000)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed")
    parser.add_argument(
        "--artifact", type=Path, default=root / "artifacts/guard_bundle.joblib"
    )
    parser.add_argument(
        "--candidate-cache-dir",
        "--page-cache-dir",
        dest="candidate_cache_dir",
        type=Path,
        default=Path(__file__).with_name("outputs") / "sanitized_shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "pilot_5k.jsonl",
    )
    parser.add_argument(
        "--report", type=Path, default=root / "reports/wildchat-sample.json"
    )
    parser.add_argument(
        "--execute-fetch",
        action="store_true",
        help="download bounded pinned shards and write the ignored local sample",
    )
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 50_000:
        parser.error("--sample-size must be between 1 and 50000")
    if not 1 <= args.candidates_per_shard <= 60_000:
        parser.error("--candidates-per-shard must be between 1 and 60000")
    if args.candidates_per_shard * len(SHARDS) < args.sample_size:
        parser.error("candidate row budget cannot be smaller than sample size")
    plan = {
        "mode": "execute-fetch" if args.execute_fetch else "offline-plan",
        "dataset": DATASET,
        "revision": REVISION,
        "sampled_rows_target": args.sample_size,
        "accepted_negative_rows": "unknown until model judgments",
        "shards": [shard["index"] for shard in SHARDS],
        "exact_revision_parquet_bytes": sum(shard["size"] for shard in SHARDS),
        "candidates_per_shard": args.candidates_per_shard,
        "full_dataset_download": False,
        "output": str(args.output),
        "report": str(args.report),
    }
    if not args.execute_fetch:
        print(json.dumps(plan, sort_keys=True))
        return
    report = execute(args)
    print(
        json.dumps(
            {
                **plan,
                "output_sha256": report["output"]["sha256"],
                "sampled_rows": report["sampling"]["sampled_rows"],
                "eligible_rows": report["sampling"]["eligible_before_stratification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
