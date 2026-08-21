from __future__ import annotations

import bz2
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import ijson
import pyarrow.parquet as pq

from ..data import _sample, _set_source_role, text_hash
from ._shared import FILES, _download, _download_files, _parquet_dataset


def _gandalf_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    dataset, downloads = _parquet_dataset("gandalf")

    def rows() -> Iterator[dict]:
        for split in ("train", "validation", "test"):
            role = "candidate" if split == "train" else "dev_test"
            for index, source_row in enumerate(dataset[split]):
                text = source_row["text"]
                row = _sample(
                    text=text,
                    label=1,
                    attack_type="direct_prompt_injection",
                    source="gandalf",
                    source_split=split,
                    source_id=f"{split}:{index}:{text_hash(text)}",
                    group_id=f"gandalf:{text_hash(text)}",
                    label_basis="filtered_human_attack_attempt",
                )
                row["source_similarity"] = float(source_row["similarity"])
                yield _set_source_role(row, role)

    return rows(), downloads, {}, None


def _llmail_attack_attempt(value: object) -> str:
    if value is True or value == Decimal(1):
        return "True"
    if value is False or value == Decimal(0):
        return "False"
    if isinstance(value, str) and value in {"True", "False", "Unclear"}:
        return value
    if isinstance(value, list) and all(
        item in {"True", "False", "Unclear"} for item in value
    ):
        values = set(value)
        return values.pop() if len(values) == 1 else "Unclear"
    raise ValueError("llmail has an unexpected attack-attempt label")


def _llmail_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    paths, downloads = _download_files("llmail")
    profile = {
        "normalized_source_labels": {},
        "raw_rows": {},
        "raw_rows_without_unique_annotation": {},
        "incomplete_raw_rows": {},
        "labelled_unique_rows_without_raw_submission": {},
        "false_positive_controls": 0,
        "raw_projection": {
            "detector_text": "subject + body",
            "retained_metadata": [
                "RowKey",
                "job_id",
                "team_id",
                "scenario",
                "objectives",
                "scheduled_time",
                "started_time",
                "completed_time",
                "sha256(output)",
            ],
            "excluded_from_canonical_shard": ["output", "Timestamp"],
        },
    }

    def rows() -> Iterator[dict]:
        for phase in ("phase1", "phase2"):
            annotations = {}
            with paths[f"{phase}_labels"].open("rb") as handle:
                for text, annotation in ijson.kvitems(handle, ""):
                    if not isinstance(annotation, dict) or not {
                        "attack_attempt",
                        "reason",
                    } <= set(annotation):
                        raise ValueError(f"llmail:{phase} has an unexpected annotation")
                    attack_attempt = _llmail_attack_attempt(
                        annotation["attack_attempt"]
                    )
                    label_key = f"{phase}:{attack_attempt}"
                    profile["normalized_source_labels"][label_key] = (
                        profile["normalized_source_labels"].get(label_key, 0) + 1
                    )
                    annotations[hashlib.sha256(text.encode()).digest()] = (
                        attack_attempt,
                        str(annotation["reason"]),
                    )

            matched = set()
            raw_rows = 0
            raw_without_annotation = 0
            incomplete_raw_rows = 0
            with paths[f"{phase}_raw"].open(encoding="utf-8") as handle:
                for line in handle:
                    source_row = json.loads(line)
                    required = {
                        "RowKey",
                        "Timestamp",
                        "body",
                        "completed_time",
                        "job_id",
                        "objectives",
                        "output",
                        "scenario",
                        "scheduled_time",
                        "started_time",
                        "subject",
                        "team_id",
                    }
                    if set(source_row) != required:
                        raise ValueError(f"llmail:{phase} has an unexpected raw schema")
                    string_fields = required - {
                        "Timestamp",
                        "completed_time",
                        "output",
                        "started_time",
                    }
                    if not all(
                        isinstance(source_row[field], str) for field in string_fields
                    ) or not all(
                        source_row[field] is None or isinstance(source_row[field], str)
                        for field in ("completed_time", "output", "started_time")
                    ):
                        raise ValueError(f"llmail:{phase} has invalid raw field types")
                    text = (
                        f"Subject of the email: {source_row['subject']}.   "
                        f"Body: {source_row['body']}"
                    )
                    raw_digest = hashlib.sha256(text.encode()).digest()
                    annotation = annotations.get(raw_digest)
                    attack_attempt = annotation[0] if annotation else "Unlabelled"
                    reason = annotation[1] if annotation else "no_unique_annotation"
                    positive = attack_attempt == "True"
                    role = (
                        "candidate"
                        if phase == "phase1" and positive
                        else "dev_test"
                        if phase == "phase2" and positive
                        else "uncertain"
                    )
                    objectives = json.loads(source_row["objectives"])
                    if not isinstance(objectives, dict) or not all(
                        isinstance(key, str) and type(value) is bool
                        for key, value in objectives.items()
                    ):
                        raise ValueError(f"llmail:{phase} has invalid objectives")
                    team_id = str(source_row["team_id"])
                    level = str(source_row["scenario"])
                    row = _sample(
                        text=text,
                        label=1 if positive else None,
                        attack_type=("indirect_prompt_injection" if positive else None),
                        security_label=(
                            "indirect_prompt_injection" if positive else "uncertain"
                        ),
                        source="llmail",
                        source_split=phase,
                        source_id=f"{phase}:{source_row['job_id']}",
                        group_id=f"llmail:{phase}:{text_hash(text)}",
                        split_group_id=(f"llmail:{phase}:team:{team_id}:level:{level}"),
                        category=reason,
                        input_channel="untrusted_content",
                        label_basis=(
                            "challenge_attack_attempt_annotation"
                            if annotation
                            else "unlabelled_raw_challenge_submission"
                        ),
                    )
                    row.update(
                        {
                            "source_attack_attempt": attack_attempt,
                            "source_completed_time": source_row["completed_time"],
                            "source_row_key": source_row["RowKey"],
                            "source_objectives": objectives,
                            "source_raw_submission": True,
                            "source_scenario": level,
                            "source_scheduled_time": source_row["scheduled_time"],
                            "source_started_time": source_row["started_time"],
                            "source_target_output_sha256": (
                                hashlib.sha256(
                                    source_row["output"].encode()
                                ).hexdigest()
                                if source_row["output"] is not None
                                else None
                            ),
                            "source_team_id": team_id,
                        }
                    )
                    raw_rows += 1
                    if annotation:
                        matched.add(raw_digest)
                    else:
                        raw_without_annotation += 1
                    if source_row["output"] is None:
                        incomplete_raw_rows += 1
                    yield _set_source_role(row, role)

            profile["raw_rows"][phase] = raw_rows
            profile["raw_rows_without_unique_annotation"][phase] = (
                raw_without_annotation
            )
            profile["incomplete_raw_rows"][phase] = incomplete_raw_rows

            unmatched = 0
            with paths[f"{phase}_labels"].open("rb") as handle:
                for text, annotation in ijson.kvitems(handle, ""):
                    raw_digest = hashlib.sha256(text.encode()).digest()
                    if raw_digest in matched:
                        continue
                    attack_attempt = _llmail_attack_attempt(
                        annotation["attack_attempt"]
                    )
                    positive = attack_attempt == "True"
                    role = (
                        "candidate"
                        if phase == "phase1" and positive
                        else "dev_test"
                        if phase == "phase2" and positive
                        else "uncertain"
                    )
                    row = _sample(
                        text=text,
                        label=1 if positive else None,
                        attack_type=("indirect_prompt_injection" if positive else None),
                        security_label=(
                            "indirect_prompt_injection" if positive else "uncertain"
                        ),
                        source="llmail",
                        source_split=phase,
                        source_id=f"{phase}:label:{raw_digest.hex()}",
                        group_id=f"llmail:{phase}:{text_hash(text)}",
                        category=str(annotation["reason"]),
                        input_channel="untrusted_content",
                        label_basis="challenge_attack_attempt_annotation_without_raw_submission",
                    )
                    row.update(
                        {
                            "source_attack_attempt": attack_attempt,
                            "source_raw_submission": False,
                        }
                    )
                    unmatched += 1
                    yield _set_source_role(row, role)
            profile["labelled_unique_rows_without_raw_submission"][phase] = unmatched

        controls = json.loads(paths["false_positive_controls"].read_text())
        if not isinstance(controls, list) or not all(
            isinstance(text, str) and text.strip() for text in controls
        ):
            raise ValueError("llmail false-positive controls have an unexpected schema")
        profile["false_positive_controls"] = len(controls)
        for index, text in enumerate(controls):
            row = _sample(
                text=text,
                label=0,
                attack_type=None,
                source="llmail",
                source_split="false_positive_controls",
                source_id=f"false_positive:{index}:{text_hash(text)}",
                group_id=f"llmail:false_positive:{text_hash(text)}",
                input_channel="untrusted_content",
                label_basis="challenge_false_positive_control",
            )
            row["source_attack_attempt"] = "False"
            yield _set_source_role(row, "dev_test")

    return rows(), downloads, profile, None


@contextmanager
def _bz2_text(path: Path) -> Iterator[io.TextIOWrapper]:
    """Open a bz2 file for text reading, preferring the C bzip2 binary.

    Callers must drain the stream to EOF; that is the contract. A nonzero
    bzip2 exit code raises after the read so a killed or failing
    decompressor cannot silently truncate the stream. An early exit that
    closes the pipe usually kills bzip2 with SIGPIPE and trips the same
    check, but only when the unread output exceeds the OS pipe buffer, so
    it is best-effort detection of a non-draining caller, not a guarantee.
    """
    bzip2 = shutil.which("bzip2")
    if bzip2 is None:
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            yield handle
        return
    process = subprocess.Popen([bzip2, "-dc", str(path)], stdout=subprocess.PIPE)
    try:
        with io.TextIOWrapper(process.stdout, encoding="utf-8") as handle:
            yield handle
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.wait() != 0:
        raise RuntimeError(
            f"bzip2 -dc exited with code {process.returncode} while reading {path}"
        )


def _tensor_trust_raw_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    paths, downloads = _download_files("tensor_trust_raw")
    profile = {
        "raw_attack_rows": 0,
        "raw_defense_rows": 0,
        "extraction_detection_rows": 0,
        "empty_attacker_input_rows_omitted": 0,
        "empty_defense_rows_omitted": 0,
        "incomplete_defense_rows": 0,
        "empty_extraction_output_rows_omitted": 0,
        "excluded_files": {
            "raw-data/v1": "superseded by the pinned v2 raw dumps",
        },
    }

    def rows() -> Iterator[dict]:
        with _bz2_text(paths["attacks_v2"]) as handle:
            for line in handle:
                source_row = json.loads(line)
                profile["raw_attack_rows"] += 1
                required = {
                    "attack_id",
                    "attacker_id_anonymized",
                    "defender_id_anonymized",
                    "attacker_input",
                    "llm_choice",
                    "output_is_access_granted",
                    "is_self_attack",
                    "timestamp",
                }
                if not required <= set(source_row):
                    raise ValueError("tensor_trust_raw:v2 has an unexpected schema")
                if (
                    type(source_row["attack_id"]) is not int
                    or type(source_row["attacker_id_anonymized"]) is not int
                    or type(source_row["defender_id_anonymized"]) is not int
                    or type(source_row["output_is_access_granted"]) is not bool
                    or type(source_row["is_self_attack"]) is not bool
                    or not isinstance(source_row["llm_choice"], str)
                    or not isinstance(source_row["timestamp"], str)
                ):
                    raise ValueError("tensor_trust_raw:v2 has invalid field types")
                text = source_row["attacker_input"]
                if not isinstance(text, str) or not text.strip():
                    profile["empty_attacker_input_rows_omitted"] += 1
                    continue
                attacker = str(source_row["attacker_id_anonymized"])
                defender = str(source_row["defender_id_anonymized"])
                row = _sample(
                    text=text,
                    label=1,
                    attack_type="direct_prompt_injection",
                    source="tensor_trust_raw",
                    source_split="raw_v2",
                    source_id=str(source_row["attack_id"]),
                    group_id=f"tensor_trust_raw:attack:{source_row['attack_id']}",
                    split_group_id=f"tensor_trust_raw:attacker:{attacker}",
                    category="game_attack_attempt",
                    label_basis="human_game_attack_attempt",
                )
                row.update(
                    {
                        "source_attacker_id_anonymized": attacker,
                        "source_defender_id_anonymized": defender,
                        "source_timestamp": str(source_row["timestamp"]),
                        "source_model": str(source_row["llm_choice"]),
                        "source_attack_success": source_row["output_is_access_granted"],
                        "source_self_attack": source_row["is_self_attack"],
                    }
                )
                yield _set_source_role(row, "candidate")

        with _bz2_text(paths["defenses_v2"]) as handle:
            for line in handle:
                source_row = json.loads(line)
                profile["raw_defense_rows"] += 1
                required = {
                    "defense_id",
                    "defender_id_anonymized",
                    "opening_defense",
                    "closing_defense",
                    "access_code",
                    "llm_choice",
                    "llm_output",
                    "output_is_access_granted",
                    "timestamp",
                }
                if set(source_row) != required:
                    raise ValueError(
                        "tensor_trust_raw:defenses_v2 has an unexpected schema"
                    )
                opening = source_row["opening_defense"] or ""
                closing = source_row["closing_defense"] or ""
                if (
                    not isinstance(source_row["defense_id"], int)
                    or not isinstance(source_row["defender_id_anonymized"], int)
                    or type(source_row["output_is_access_granted"]) is not bool
                    or not all(
                        isinstance(source_row[field], str)
                        for field in (
                            "llm_choice",
                            "timestamp",
                        )
                    )
                    or not all(
                        source_row[field] is None or isinstance(source_row[field], str)
                        for field in (
                            "opening_defense",
                            "closing_defense",
                            "access_code",
                            "llm_output",
                        )
                    )
                ):
                    raise ValueError(
                        "tensor_trust_raw:defenses_v2 has invalid field types"
                    )
                if not opening.strip() and not closing.strip():
                    profile["empty_defense_rows_omitted"] += 1
                    continue
                if any(
                    source_row[field] is None
                    for field in (
                        "opening_defense",
                        "closing_defense",
                        "access_code",
                        "llm_output",
                    )
                ):
                    profile["incomplete_defense_rows"] += 1
                defender = str(source_row["defender_id_anonymized"])
                text = f"Opening defense:\n{opening}\n\nClosing defense:\n{closing}"
                row = _sample(
                    text=text,
                    label=None,
                    attack_type=None,
                    security_label="uncertain",
                    source="tensor_trust_raw",
                    source_split="raw_defenses_v2",
                    source_id=f"defense:{source_row['defense_id']}",
                    group_id=f"tensor_trust_raw:defense:{source_row['defense_id']}",
                    split_group_id=f"tensor_trust_raw:defender:{defender}",
                    category="game_defense",
                    input_channel="trusted_instruction",
                    label_basis="human_game_defense_not_injection_supervision",
                )
                row.update(
                    {
                        "source_access_code_sha256": hashlib.sha256(
                            source_row["access_code"].encode()
                        ).hexdigest()
                        if source_row["access_code"] is not None
                        else None,
                        "source_access_granted": source_row["output_is_access_granted"],
                        "source_defender_id_anonymized": defender,
                        "source_model": str(source_row["llm_choice"]),
                        "source_target_output_sha256": hashlib.sha256(
                            source_row["llm_output"].encode()
                        ).hexdigest()
                        if source_row["llm_output"] is not None
                        else None,
                        "source_timestamp": str(source_row["timestamp"]),
                    }
                )
                yield _set_source_role(row, "auxiliary")

        with paths["extraction_detection_v1"].open(encoding="utf-8") as handle:
            for line in handle:
                source_row = json.loads(line)
                profile["extraction_detection_rows"] += 1
                required = {
                    "sample_id",
                    "access_code",
                    "llm_output",
                    "is_prompt_extraction",
                }
                if set(source_row) != required:
                    raise ValueError(
                        "tensor_trust_raw:extraction_detection_v1 has an unexpected schema"
                    )
                if (
                    type(source_row["sample_id"]) is not int
                    or not isinstance(source_row["access_code"], str)
                    or type(source_row["is_prompt_extraction"]) is not bool
                ):
                    raise ValueError(
                        "tensor_trust_raw:extraction_detection_v1 has invalid field types"
                    )
                text = source_row["llm_output"]
                if not isinstance(text, str) or not text.strip():
                    profile["empty_extraction_output_rows_omitted"] += 1
                    continue
                row = _sample(
                    text=text,
                    label=None,
                    attack_type=None,
                    security_label="uncertain",
                    source="tensor_trust_raw",
                    source_split="extraction_detection_v1",
                    source_id=f"extraction:{source_row['sample_id']}",
                    group_id=f"tensor_trust_raw:extraction:{source_row['sample_id']}",
                    category="model_output_extraction_detection",
                    input_channel="model_output",
                    label_basis="source_output_extraction_annotation_not_injection_supervision",
                )
                row.update(
                    {
                        "source_access_code_sha256": hashlib.sha256(
                            source_row["access_code"].encode()
                        ).hexdigest(),
                        "source_is_prompt_extraction": source_row[
                            "is_prompt_extraction"
                        ],
                    }
                )
                yield _set_source_role(row, "auxiliary")

    return rows(), downloads, profile, None


def _browsesafe_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    dataset, downloads = _parquet_dataset("browsesafe")

    def rows() -> Iterator[dict]:
        for split in ("train", "test"):
            role = "candidate" if split == "train" else "dev_test"
            for index, source_row in enumerate(dataset[split]):
                label = source_row["label"]
                if label not in {"yes", "no"}:
                    raise ValueError(f"browsesafe:{split} has an unexpected label")
                text = source_row["content"]
                positive = label == "yes"
                digest = text_hash(text)
                row = _sample(
                    text=text,
                    label=int(positive),
                    attack_type=("indirect_prompt_injection" if positive else None),
                    source="browsesafe",
                    source_split=split,
                    source_id=f"{split}:{index}:{digest}",
                    group_id=f"browsesafe:{split}:{digest}",
                    input_channel="untrusted_content",
                    label_basis="benchmark_document_construction",
                )
                row.update(
                    {
                        "document_characters": len(text),
                        "document_granularity": "whole_document",
                        "known_attack_span": False,
                    }
                )
                yield _set_source_role(row, role)

    return rows(), downloads, {"positive_payload_spans_available": False}, None


def _hackaprompt_sample(source_row: dict, index: int) -> dict:
    text = source_row.get("user_input")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"hackaprompt:{index} has empty detector text")
    if (
        type(source_row.get("level")) is not int
        or type(source_row.get("correct")) is not bool
        or type(source_row.get("error")) is not bool
        or type(source_row.get("token_count")) is not int
        or not isinstance(source_row.get("model"), str)
        or not source_row["model"].strip()
        or not isinstance(source_row.get("dataset"), str)
        or not source_row["dataset"].strip()
        or (
            source_row.get("timestamp") is not None
            and not isinstance(source_row["timestamp"], str)
        )
        or (
            source_row.get("session_id") is not None
            and not isinstance(source_row["session_id"], str)
        )
        or (
            source_row.get("score") is not None
            and (
                type(source_row["score"]) not in {int, float}
                or not math.isfinite(source_row["score"])
            )
        )
    ):
        raise ValueError(f"hackaprompt:{index} has invalid metadata types")
    level = str(source_row["level"])
    session_id = source_row["session_id"]
    row = _sample(
        text=text,
        label=1,
        attack_type="direct_prompt_injection",
        source="hackaprompt",
        source_split="full",
        source_id=f"{index}:{text_hash(text)}",
        group_id=f"hackaprompt:{index}",
        split_group_id=f"hackaprompt:level:{level}",
        category=f"level:{level}",
        label_basis="competition_attack_attempt",
    )
    row.update(
        {
            "source_attack_success": source_row["correct"],
            "source_collection": source_row["dataset"],
            "source_error": source_row["error"],
            "source_level": source_row["level"],
            "source_model": source_row["model"],
            "source_score": source_row["score"],
            "source_session_id_sha256": (
                hashlib.sha256(session_id.encode()).hexdigest()
                if session_id is not None
                else None
            ),
            "source_timestamp": source_row["timestamp"],
            "source_token_count": source_row["token_count"],
        }
    )
    return _set_source_role(row, "candidate")


def _hackaprompt_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    filename, expected = FILES["hackaprompt"]["full"]
    path, digest = _download("hackaprompt", filename, expected)
    parquet = pq.ParquetFile(path)
    profile = {
        "empty_user_input_rows_omitted": 0,
        "rows_without_timestamp": 0,
        "rows_without_session_id": 0,
        "rows_without_score": 0,
        "source_error_rows": 0,
        "collections": {},
        "unique_session_ids": 0,
        "participant_identifier_available": False,
        "split_grouping": (
            "whole challenge level; session_id is retained by hash but is not "
            "claimed as participant identity"
        ),
    }

    def rows() -> Iterator[dict]:
        required = {
            "level",
            "user_input",
            "correct",
            "model",
            "token_count",
            "error",
            "score",
            "dataset",
            "timestamp",
            "session_id",
        }
        if not required <= set(parquet.schema_arrow.names):
            raise ValueError("hackaprompt has an unexpected schema")
        collections = Counter()
        sessions = set()
        source_rows = (
            source_row
            for batch in parquet.iter_batches(batch_size=8192)
            for source_row in batch.to_pylist()
        )
        for index, source_row in enumerate(source_rows):
            text = source_row["user_input"]
            if not isinstance(text, str) or not text.strip():
                profile["empty_user_input_rows_omitted"] += 1
                continue
            row = _hackaprompt_sample(source_row, index)
            collections[source_row["dataset"]] += 1
            if source_row["timestamp"] is None:
                profile["rows_without_timestamp"] += 1
            if source_row["session_id"] is None:
                profile["rows_without_session_id"] += 1
            else:
                sessions.add(source_row["session_id"])
            if source_row["score"] is None:
                profile["rows_without_score"] += 1
            if source_row["error"]:
                profile["source_error_rows"] += 1
            yield row
        profile["collections"] = dict(sorted(collections.items()))
        profile["unique_session_ids"] = len(sessions)

    return rows(), {filename: digest}, profile, None


def _wildjailbreak_sample(source_row: dict, split: str, index: int) -> dict:
    data_type = source_row.get("data_type")
    mapping = {
        "vanilla_benign": (0, "benign", "safe"),
        "vanilla_harmful": (0, "harmful_non_injection", "unsafe"),
        "adversarial_benign": (None, "uncertain", "safe"),
        "adversarial_harmful": (1, "direct_jailbreak", "unsafe"),
    }
    if data_type not in mapping:
        raise ValueError(f"wildjailbreak:{split} has an unexpected data_type")
    if split == "eval":
        expected_label = "1" if data_type == "adversarial_harmful" else "0"
        if source_row.get("label") != expected_label:
            raise ValueError(
                f"wildjailbreak:{split}:{index} label disagrees with data_type"
            )
    adversarial = data_type.startswith("adversarial_")
    text = source_row.get("adversarial" if adversarial else "vanilla")
    if not isinstance(text, str) or not text.strip():
        fallback = source_row.get("vanilla")
        if not adversarial or not isinstance(fallback, str) or not fallback.strip():
            raise ValueError(f"wildjailbreak:{split}:{index} has empty detector text")
        row = _sample(
            text=fallback,
            label=None,
            attack_type=None,
            security_label="uncertain",
            source="wildjailbreak",
            source_split=split,
            source_id=f"{split}:{index}:{text_hash(fallback)}",
            group_id=f"wildjailbreak:{split}:{index}",
            split_group_id=f"wildjailbreak:base:{text_hash(fallback)}",
            category=f"{data_type}:missing_adversarial_text",
            goal_policy_status="safe",
            label_basis="source_row_missing_adversarial_projection",
        )
        return _set_source_role(row, "uncertain")
    injection_label, security_label, goal_status = mapping[data_type]
    base = source_row.get("vanilla")
    base_hash = (
        text_hash(base) if isinstance(base, str) and base.strip() else text_hash(text)
    )
    row = _sample(
        text=text,
        label=injection_label,
        attack_type="direct_jailbreak" if injection_label else None,
        security_label=security_label,
        source="wildjailbreak",
        source_split=split,
        source_id=f"{split}:{index}:{text_hash(text)}",
        group_id=f"wildjailbreak:{split}:{index}",
        split_group_id=f"wildjailbreak:base:{base_hash}",
        category=data_type,
        goal_policy_status=goal_status,
        label_basis="four_way_source_construction",
    )
    tactics = source_row.get("tactics")
    if tactics:
        row["source_tactics"] = str(tactics)
    if source_row.get("label") not in {None, ""}:
        row["source_harmfulness_label"] = str(source_row["label"])
    if data_type == "adversarial_benign":
        return _set_source_role(row, "auxiliary")
    return _set_source_role(row, "candidate" if split == "train" else "dev_test")


def _wildjailbreak_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    paths, downloads = _download_files("wildjailbreak")
    profile = {"missing_adversarial_text_rows_uncertain": 0}

    def rows() -> Iterator[dict]:
        csv.field_size_limit(10_000_000)
        for split in ("train", "eval"):
            with paths[split].open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                required = (
                    {"data_type", "vanilla", "adversarial"}
                    if split == "train"
                    else {"data_type", "adversarial", "label"}
                )
                if not required <= set(reader.fieldnames or []):
                    raise ValueError(f"wildjailbreak:{split} has an unexpected schema")
                for index, source_row in enumerate(reader):
                    if (
                        source_row.get("data_type", "").startswith("adversarial_")
                        and not (source_row.get("adversarial") or "").strip()
                    ):
                        profile["missing_adversarial_text_rows_uncertain"] += 1
                    yield _wildjailbreak_sample(source_row, split, index)

    return rows(), downloads, profile, None


def _wildguard_sample(source_row: dict, split: str, index: int) -> dict:
    text = source_row.get("prompt")
    label = source_row.get("prompt_harm_label")
    adversarial = source_row.get("adversarial")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"wildguardmix:{split}:{index} has empty prompt")
    if type(adversarial) is not bool:
        raise ValueError(f"wildguardmix:{split}:{index} has invalid adversarial flag")
    if label == "harmful" and not adversarial:
        injection_label = 0
        security_label = "harmful_non_injection"
    elif label == "unharmful" and not adversarial:
        injection_label = 0
        security_label = "benign"
    elif label in {"harmful", "unharmful", None}:
        injection_label = None
        security_label = "uncertain"
    else:
        raise ValueError(f"wildguardmix:{split} has an unexpected prompt label")
    row = _sample(
        text=text,
        label=injection_label,
        attack_type=None,
        security_label=security_label,
        source="wildguardmix",
        source_split=split,
        source_id=f"{split}:{index}:{text_hash(text)}",
        group_id=f"wildguardmix:{split}:{text_hash(text)}",
        category=source_row.get("subcategory"),
        goal_policy_status=(
            "unsafe"
            if label == "harmful"
            else "safe"
            if label == "unharmful"
            else "unknown"
        ),
        label_basis=(
            "source_model_weak_prompt_harmfulness"
            if split == "train"
            else "three_human_annotators_prompt_harmfulness"
        ),
    )
    row["source_adversarial"] = adversarial
    if "prompt_harm_agreement" in source_row:
        row["source_prompt_harm_agreement"] = source_row["prompt_harm_agreement"]
    row["source_harmfulness_label"] = label
    eligible = label == "harmful" or (label == "unharmful" and not adversarial)
    role = (
        "dev_test"
        if split == "test" and eligible
        else "auxiliary"
        if label in {"harmful", "unharmful"}
        else "uncertain"
    )
    return _set_source_role(row, role)


def _wildguardmix_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    paths, downloads = _download_files("wildguardmix")
    profile = {"empty_prompt_rows_omitted": 0}

    def rows() -> Iterator[dict]:
        for split in ("train", "test"):
            parquet = pq.ParquetFile(paths[split])
            required = {"prompt", "prompt_harm_label", "adversarial"}
            if not required <= set(parquet.schema_arrow.names):
                raise ValueError(f"wildguardmix:{split} has an unexpected schema")
            source_rows = (
                source_row
                for batch in parquet.iter_batches(batch_size=8192)
                for source_row in batch.to_pylist()
            )
            for index, source_row in enumerate(source_rows):
                if (
                    not isinstance(source_row.get("prompt"), str)
                    or not source_row["prompt"].strip()
                ):
                    profile["empty_prompt_rows_omitted"] += 1
                    continue
                yield _wildguard_sample(source_row, split, index)

    return rows(), downloads, profile, None
