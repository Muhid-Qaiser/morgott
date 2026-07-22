from __future__ import annotations

import bz2
import csv
import hashlib
import io
import json
import math
import re
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from decimal import Decimal
from pathlib import Path

import ijson
from datasets import disable_progress_bars, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

from .data import (
    SOURCES,
    _csv_rows,
    _fetch,
    _github_raw,
    _sample,
    _set_source_role,
    atomic_write_text,
    file_sha256,
    text_hash,
)
from .routing import materialize_routing_views

FILES = {
    "gandalf": {
        "train": (
            "data/train-00000-of-00001-ded53be747ff55cd.parquet",
            "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
        ),
        "validation": (
            "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
            "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
        ),
        "test": (
            "data/test-00000-of-00001-bc92128b9288a6d1.parquet",
            "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
        ),
    },
    "llmail": {
        "phase1_labels": (
            "data/labelled_unique_submissions_phase1.json",
            "691dfa1595d2bd0e731069f233bd5448f7af1c0ebd9732bc6f12dd6cee446586",
        ),
        "phase2_labels": (
            "data/labelled_unique_submissions_phase2.json",
            "f89af984e345430c3b357903890e30867bf4676f4ef10c138cc7bad218e890b8",
        ),
        "phase1_raw": (
            "data/raw_submissions_phase1.jsonl",
            "a9c62eca699dd270fdfbbfbfcc1253f5e5017f6d5a34ff7cb0f2cbb80b7f7c0a",
        ),
        "phase2_raw": (
            "data/raw_submissions_phase2.jsonl",
            "a9207e1d893ccb74ca6f9cc5eecea433bc49c23a26bed88088afd385c7ab18b6",
        ),
        "false_positive_controls": (
            "data/emails_for_fp_tests.json",
            "4ddd950b5dbaa8548f5597c886d8e09a051ba07f80a9291fdcca9c2397d22abe",
        ),
    },
    "tensor_trust_raw": {
        "attacks_v2": (
            "raw-data/v2/raw_dump_attacks.jsonl.bz2",
            "87853cc8065a22156d15c4bdec777a8d35758749beb79769147d63a9644a73ce",
        ),
        "defenses_v2": (
            "raw-data/v2/raw_dump_defenses.jsonl.bz2",
            "cbdf52c469b2a57db61f885562bae1725cd56174cc201979e7bc042a680a5166",
        ),
        "extraction_detection_v1": (
            "detecting-extractions/v1/prompt_extraction_detection.jsonl",
            "af28d09554db4f8ed91d042005c3457f72c88e763e3787c3b1c76c1d9e8f8260",
        ),
    },
    "browsesafe": {
        "train": (
            "train.parquet",
            "430881fa53da3898956048676f58db894fc760cfa348ab264ee8a07c44a3d9fb",
        ),
        "test": (
            "test.parquet",
            "00cbad96b60fee46e016d79af6981fb221384c61f12cf28b4f04b5a6420573d0",
        ),
    },
    "hackaprompt": {
        "full": (
            "hackaprompt.parquet",
            "bedca308fbd71be57793930e4e4a0dcfbda2a27b6d0f2ad3191bb20a6a315928",
        ),
    },
    "wildjailbreak": {
        "train": (
            "train/train.tsv",
            "376719bfdb46ad1a19e7ba4f587f80cc7cb1368cc213ee647ef739c170550f7a",
        ),
        "eval": (
            "eval/eval.tsv",
            "eeb7e43aafa0151588f5cb8994b99adc0ee34a57819d92f43a6084f0b7fe4fa4",
        ),
    },
    "wildguardmix": {
        "train": (
            "train/wildguard_train.parquet",
            "02ecea8a724a9146a1e473a95a7cdf262adfe9c7d5408953ca86d2fcfbdc8953",
        ),
        "test": (
            "test/wildguard_test.parquet",
            "6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca",
        ),
    },
    "false_reject": {
        "train": (
            "train.jsonl",
            "0331899da03e9c2c232acffd8b086e5e57116e1f53986012743b9a3bea46f868",
        ),
        "test": (
            "test.jsonl",
            "644b243987b4d16f36b1b668b03a17fa49d18174fee80fbaeef94a53facc462d",
        ),
    },
    "coconot": {
        "train": (
            "pref/train-00000-of-00001.parquet",
            "136ac18a54fbfa98472eabb77369de89c556ea05626584445f381238c287e104",
        ),
        "dev_test": (
            "contrast/test-00000-of-00001.parquet",
            "d2d4f9ea33eac017cfdd2b56669e417e979933418276083d3acb28be170a588f",
        ),
    },
    "jbb_benign": {
        "dev_test": (
            "data/benign-behaviors.csv",
            "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
        ),
    },
    "lmsys_arena": {
        "train": (
            "data/train-00000-of-00001-cced8514c7ed782a.parquet",
            "3726a6352e9bfc34e206460646f6e5e99bb837751966a671ddd30c7f64e5b06e",
        ),
    },
}


def _download(source: str, filename: str, expected_sha256: str) -> tuple[Path, str]:
    info = SOURCES[source]
    try:
        path = Path(
            hf_hub_download(
                info["repo"],
                filename,
                repo_type="dataset",
                revision=info["revision"],
            )
        )
    except GatedRepoError as error:
        if info.get("gated"):
            raise RuntimeError(
                f"{source}: access gate not accepted or token unavailable"
            ) from error
        raise
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"{source}:{filename} does not match its pinned digest")
    return path, digest


def _parquet_dataset(source: str) -> tuple[dict, dict[str, str]]:
    datasets = {}
    downloads = {}
    for split, (filename, expected) in FILES[source].items():
        path, digest = _download(source, filename, expected)
        downloads[filename] = digest
        datasets[split] = load_dataset(
            "parquet", data_files={split: str(path)}, split=split
        )
    return datasets, downloads


def _verified_archive(source: str) -> tuple[bytes, str]:
    info = SOURCES[source]
    return _fetch(
        info["archive_url"],
        max_bytes=info["bytes"],
        expected_bytes=info["bytes"],
        expected_sha256=info["sha256"],
    )


def _gandalf_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
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

    return rows(), downloads, {}


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


def _llmail_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    paths = {}
    downloads = {}
    for name, (filename, expected) in FILES["llmail"].items():
        path, digest = _download("llmail", filename, expected)
        paths[name] = path
        downloads[filename] = digest
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

    return rows(), downloads, profile


def _tensor_trust_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    paths = {}
    downloads = {}
    for name, (filename, expected) in FILES["tensor_trust_raw"].items():
        path, digest = _download("tensor_trust_raw", filename, expected)
        paths[name] = path
        downloads[filename] = digest
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
        with bz2.open(paths["attacks_v2"], "rt", encoding="utf-8") as handle:
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

        with bz2.open(paths["defenses_v2"], "rt", encoding="utf-8") as handle:
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

    return rows(), downloads, profile


def _browsesafe_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
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

    return rows(), downloads, {"positive_payload_spans_available": False}


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


def _hackaprompt_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    filename, expected = FILES["hackaprompt"]["full"]
    path, digest = _download("hackaprompt", filename, expected)
    dataset = load_dataset("parquet", data_files={"full": str(path)})["full"]
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
        if not required <= set(dataset.column_names):
            raise ValueError("hackaprompt has an unexpected schema")
        collections = Counter()
        sessions = set()
        for index, source_row in enumerate(dataset):
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

    return rows(), {filename: digest}, profile


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


def _wildjailbreak_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    paths = {}
    downloads = {}
    for split, (filename, expected) in FILES["wildjailbreak"].items():
        path, digest = _download("wildjailbreak", filename, expected)
        paths[split] = path
        downloads[filename] = digest
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

    return rows(), downloads, profile


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


def _wildguard_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    dataset, downloads = _parquet_dataset("wildguardmix")
    profile = {"empty_prompt_rows_omitted": 0}

    def rows() -> Iterator[dict]:
        for split in ("train", "test"):
            required = {"prompt", "prompt_harm_label", "adversarial"}
            if not required <= set(dataset[split].column_names):
                raise ValueError(f"wildguardmix:{split} has an unexpected schema")
            for index, source_row in enumerate(dataset[split]):
                if (
                    not isinstance(source_row.get("prompt"), str)
                    or not source_row["prompt"].strip()
                ):
                    profile["empty_prompt_rows_omitted"] += 1
                    continue
                yield _wildguard_sample(source_row, split, index)

    return rows(), downloads, profile


def _taskmaster_split_group(dialog: dict, collection: str) -> str:
    conversation_id = dialog.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError(f"taskmaster:{collection} has no conversation id")
    instruction_id = dialog.get("instruction_id")
    if isinstance(instruction_id, str) and instruction_id:
        return f"taskmaster:{collection}:instruction:{instruction_id}"
    instructions = dialog.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return f"taskmaster:{collection}:instructions:{text_hash(instructions)}"
    scenario = dialog.get("scenario")
    if isinstance(scenario, str) and scenario:
        return f"taskmaster:{collection}:scenario:{text_hash(scenario)}"
    return f"taskmaster:{conversation_id}"


def _taskmaster_sample(
    dialog: dict,
    turn: dict,
    *,
    collection: str,
    source_file: str,
    source_split: str,
    split_group_id: str,
    record_index: int,
    role: str,
    domain: str,
) -> dict:
    conversation_id = dialog.get("conversation_id")
    turn_index = turn.get("index")
    speaker = turn.get("speaker")
    text = turn.get("text")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError(
            f"taskmaster:{collection}:{record_index} has no conversation id"
        )
    if type(turn_index) is not int:
        raise ValueError(f"taskmaster:{conversation_id} has an invalid turn index")
    if not isinstance(speaker, str) or speaker.casefold() not in {"user", "assistant"}:
        raise ValueError(
            f"taskmaster:{conversation_id}:{turn_index} has invalid speaker"
        )
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"taskmaster:{conversation_id}:{turn_index} has empty text")
    speaker = speaker.casefold()
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="taskmaster",
        source_split=source_split,
        source_id=f"{source_file}:{record_index}:{conversation_id}:{turn_index}",
        group_id=f"taskmaster:{conversation_id}",
        split_group_id=split_group_id,
        category=domain,
        input_channel="direct_user" if speaker == "user" else "model_output",
        label_basis="bounded_task_dialogue_collection",
    )
    row.update(
        {
            "source_collection": collection,
            "source_conversation_id": conversation_id,
            "source_domain": domain,
            "source_file": source_file,
            "source_language": "en",
            "source_record_index": record_index,
            "source_speaker": speaker,
            "source_turn_index": turn_index,
        }
    )
    instruction_id = dialog.get("instruction_id")
    scenario = dialog.get("scenario")
    if isinstance(instruction_id, str) and instruction_id:
        row["source_instruction_id"] = instruction_id
    if isinstance(scenario, str) and scenario:
        row["source_scenario"] = scenario
    return _set_source_role(row, role)


def _taskmaster_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    data, digest = _verified_archive("taskmaster")
    profile = {
        "projection": "all non-empty user and assistant turn text from Taskmaster 1-3",
        "excluded": [
            "annotations, API calls, task instructions, and reward data",
            "Taskmaster-3 transformed language-model split files",
            "Taskmaster-4 because it is outside the selected Taskmaster 1-3 release",
        ],
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()

            def one_member(suffix: str) -> tarfile.TarInfo:
                matches = [member for member in members if member.name.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f"taskmaster archive is missing {suffix}")
                return matches[0]

            tm1_splits = {}
            for split in ("train", "dev", "test"):
                member = one_member(f"/TM-1-2019/train-dev-test/{split}.csv")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"taskmaster archive cannot read {member.name}")
                for csv_row in csv.reader(io.TextIOWrapper(handle, encoding="utf-8")):
                    if csv_row and csv_row[0]:
                        tm1_splits[csv_row[0]] = split

            selected = []
            for member in members:
                name = member.name
                if name.endswith("/TM-1-2019/self-dialogs.json"):
                    selected.append((member, "tm1_self", None))
                elif name.endswith("/TM-1-2019/woz-dialogs.json"):
                    selected.append((member, "tm1_woz", None))
                elif match := re.search(r"/TM-2-2020/data/([^/]+)\.json$", name):
                    selected.append((member, "tm2", match.group(1)))
                elif re.search(r"/TM-3-2020/data/data_\d\d\.json$", name):
                    selected.append((member, "tm3", "movie-tickets"))
            if len(selected) != 29:
                raise ValueError("taskmaster archive has an unexpected data-file set")

            for member, collection, file_domain in selected:
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"taskmaster archive cannot read {member.name}")
                source_file = member.name.split("/", 1)[-1]
                for record_index, dialog in enumerate(ijson.items(handle, "item")):
                    if not isinstance(dialog, dict) or not isinstance(
                        dialog.get("utterances"), list
                    ):
                        raise ValueError(f"taskmaster:{source_file} has invalid dialog")
                    counts[f"dialogs:{collection}"] += 1
                    conversation_id = dialog.get("conversation_id")
                    if collection == "tm1_self":
                        if conversation_id not in tm1_splits:
                            raise ValueError("taskmaster tm1 self-dialog has no split")
                        split = tm1_splits[conversation_id]
                        source_split = f"tm1_self:{split}"
                        role = "candidate" if split == "train" else "dev_test"
                    else:
                        source_split = collection
                        role = "candidate"
                    instruction_id = dialog.get("instruction_id")
                    domain = file_domain or (
                        re.sub(r"[- ]\d+$", "", instruction_id)
                        if isinstance(instruction_id, str) and instruction_id
                        else collection
                    )
                    split_group_id = _taskmaster_split_group(dialog, collection)
                    for turn in dialog["utterances"]:
                        counts["raw_turns"] += 1
                        if not isinstance(turn, dict):
                            raise ValueError(
                                f"taskmaster:{source_file} has invalid turn"
                            )
                        speaker = turn.get("speaker")
                        text = turn.get("text")
                        if speaker == "":
                            counts["empty_speaker_turns_omitted"] += 1
                            continue
                        if isinstance(text, str) and not text.strip():
                            counts["empty_text_turns_omitted"] += 1
                            continue
                        yield _taskmaster_sample(
                            dialog,
                            turn,
                            collection=collection,
                            source_file=source_file,
                            source_split=source_split,
                            split_group_id=split_group_id,
                            record_index=record_index,
                            role=role,
                            domain=domain,
                        )
                        counts[f"retained_speaker:{str(speaker).casefold()}"] += 1
        profile.update(
            {
                "raw_dialogs": {
                    key.removeprefix("dialogs:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("dialogs:")
                },
                "raw_turns": counts["raw_turns"],
                "retained_speakers": {
                    key.removeprefix("retained_speaker:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("retained_speaker:")
                },
                "empty_speaker_turns_omitted": counts["empty_speaker_turns_omitted"],
                "empty_text_turns_omitted": counts["empty_text_turns_omitted"],
            }
        )

    return rows(), {"taskmaster.tar.gz": digest}, profile


def _banking77_sample(source_row: dict, split: str, index: int) -> dict:
    text = source_row.get("text")
    intent = source_row.get("category")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"banking77:{split}:{index} has empty text")
    if not isinstance(intent, str) or not intent:
        raise ValueError(f"banking77:{split}:{index} has no intent")
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="banking77",
        source_split=split,
        source_id=f"{split}:{index}",
        group_id=f"banking77:{split}:{index}",
        category=intent,
        label_basis="banking_assistant_intent_collection",
    )
    row["source_intent"] = intent
    row["source_language"] = "en"
    return _set_source_role(row, "candidate" if split == "train" else "dev_test")


def _banking77_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    expected = {
        "banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        "banking_data/test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    }
    contents = {}
    downloads = {}
    for filename, expected_digest in expected.items():
        data, digest = _github_raw("banking77", filename)
        if digest != expected_digest:
            raise ValueError(f"banking77:{filename} does not match its pinned digest")
        contents[filename] = data
        downloads[filename] = digest
    datasets = {
        split: _csv_rows(contents[f"banking_data/{split}.csv"], {"text", "category"})
        for split in ("train", "test")
    }

    def rows() -> Iterator[dict]:
        for split in ("train", "test"):
            for index, source_row in enumerate(datasets[split]):
                yield _banking77_sample(source_row, split, index)

    profile = {
        "projection": "all non-empty English online-banking queries",
        "raw_rows": {
            split: len(source_rows) for split, source_rows in datasets.items()
        },
        "lineage_limit": "the source exposes no conversation lineage; each query is a singleton group",
    }
    return rows(), downloads, profile


def _false_reject_sample(source_row: dict, split: str, index: int) -> dict:
    text = source_row.get("prompt")
    category = source_row.get("category_text")
    category_id = source_row.get("category")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"false_reject:{split}:{index} has empty prompt")
    if not isinstance(category, str) or not category or type(category_id) is not int:
        raise ValueError(f"false_reject:{split}:{index} has invalid category")
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="false_reject",
        source_split=split,
        source_id=f"{split}:{index}:{text_hash(text)}",
        group_id=f"false_reject:{split}:{text_hash(text)}",
        category=category,
        goal_policy_status="safe",
        label_basis=(
            "human_validated_benign_overrefusal_test"
            if split == "test"
            else "multi_agent_generated_benign_weak_label"
        ),
    )
    row["source_category_id"] = category_id
    row["source_language"] = "en"
    return _set_source_role(row, "dev_test" if split == "test" else "candidate")


def _false_reject_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    paths = {}
    downloads = {}
    for split, (filename, expected) in FILES["false_reject"].items():
        path, digest = _download("false_reject", filename, expected)
        paths[split] = path
        downloads[filename] = digest
    profile = {
        "projection": "prompt only",
        "excluded": "generated standard and chain-of-thought response fields",
        "raw_rows": {},
    }

    def rows() -> Iterator[dict]:
        for split in ("train", "test"):
            count = 0
            with paths[split].open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if not line.strip():
                        raise ValueError(f"false_reject:{split}:{index} has blank row")
                    source_row = json.loads(line)
                    if not isinstance(source_row, dict):
                        raise ValueError(
                            f"false_reject:{split}:{index} has invalid row"
                        )
                    yield _false_reject_sample(source_row, split, index)
                    count += 1
            profile["raw_rows"][split] = count

    return rows(), downloads, profile


def _schema_guided_dialogue_sample(
    dialog: dict,
    turn: dict,
    *,
    split: str,
    source_file: str,
    turn_index: int,
) -> dict:
    dialogue_id = dialog.get("dialogue_id")
    services = dialog.get("services")
    speaker = turn.get("speaker")
    text = turn.get("utterance")
    if not isinstance(dialogue_id, str) or not dialogue_id:
        raise ValueError(f"schema_guided_dialogue:{source_file} has no dialogue id")
    if (
        not isinstance(services, list)
        or not services
        or not all(isinstance(service, str) and service for service in services)
    ):
        raise ValueError(f"schema_guided_dialogue:{dialogue_id} has invalid services")
    if speaker not in {"USER", "SYSTEM"}:
        raise ValueError(f"schema_guided_dialogue:{dialogue_id} has invalid speaker")
    domains = sorted({service.split("_", 1)[0].casefold() for service in services})
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="schema_guided_dialogue",
        source_split=split,
        source_id=f"{split}:{dialogue_id}:{turn_index}",
        group_id=f"schema_guided_dialogue:{split}:{dialogue_id}",
        category=domains[0] if len(domains) == 1 else "multi_domain",
        input_channel="direct_user" if speaker == "USER" else "model_output",
        label_basis="simulated_task_dialogue_with_crowdworker_utterances",
    )
    row.update(
        {
            "source_dialogue_id": dialogue_id,
            "source_file": source_file,
            "source_language": "en",
            "source_services": services,
            "source_speaker": speaker.casefold(),
            "source_turn_index": turn_index,
        }
    )
    return _set_source_role(row, "candidate" if split == "train" else "dev_test")


def _schema_guided_dialogue_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    data, digest = _verified_archive("schema_guided_dialogue")
    profile = {
        "projection": "all non-empty user and system utterances",
        "excluded": "schemas, dialogue frames, slot values, actions, and service results",
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            selected = []
            for member in archive.getmembers():
                match = re.fullmatch(
                    r"[^/]+/(train|dev|test)/dialogues_\d+\.json", member.name
                )
                if match:
                    selected.append((member, match.group(1)))
            if len(selected) != 181:
                raise ValueError(
                    "schema_guided_dialogue archive has an unexpected data-file set"
                )
            for member, split in sorted(selected, key=lambda item: item[0].name):
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(
                        f"schema_guided_dialogue cannot read {member.name}"
                    )
                source_file = member.name.split("/", 1)[-1]
                for dialog in ijson.items(handle, "item"):
                    if not isinstance(dialog, dict) or not isinstance(
                        dialog.get("turns"), list
                    ):
                        raise ValueError(
                            f"schema_guided_dialogue:{source_file} has invalid dialogue"
                        )
                    counts[f"dialogs:{split}"] += 1
                    for turn_index, turn in enumerate(dialog["turns"]):
                        counts[f"raw_turns:{split}"] += 1
                        if not isinstance(turn, dict):
                            raise ValueError(
                                f"schema_guided_dialogue:{source_file} has invalid turn"
                            )
                        text = turn.get("utterance")
                        if isinstance(text, str) and not text.strip():
                            counts["empty_turns_omitted"] += 1
                            continue
                        yield _schema_guided_dialogue_sample(
                            dialog,
                            turn,
                            split=split,
                            source_file=source_file,
                            turn_index=turn_index,
                        )
        profile.update(
            {
                "raw_dialogs": {
                    split: counts[f"dialogs:{split}"]
                    for split in ("train", "dev", "test")
                },
                "raw_turns": {
                    split: counts[f"raw_turns:{split}"]
                    for split in ("train", "dev", "test")
                },
                "empty_turns_omitted": counts["empty_turns_omitted"],
            }
        )

    return rows(), {"schema_guided_dialogue.tar.gz": digest}, profile


def _massive_sample(source_row: dict, index: int) -> dict:
    source_id = source_row.get("id")
    split = source_row.get("partition")
    text = source_row.get("utt")
    locale = source_row.get("locale")
    scenario = source_row.get("scenario")
    intent = source_row.get("intent")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"massive_en:{index} has no id")
    if split not in {"train", "dev", "test"} or locale != "en-US":
        raise ValueError(f"massive_en:{source_id} has invalid split or locale")
    if not isinstance(scenario, str) or not isinstance(intent, str):
        raise ValueError(f"massive_en:{source_id} has invalid intent metadata")
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="massive_en",
        source_split=split,
        source_id=f"{split}:{source_id}",
        group_id=f"massive:{source_id}",
        category=intent,
        label_basis="localized_voice_assistant_intent_collection",
    )
    row.update(
        {
            "source_intent": intent,
            "source_language": locale,
            "source_massive_id": source_id,
            "source_scenario": scenario,
            "source_worker_sha256": hashlib.sha256(
                str(source_row.get("worker_id")).encode()
            ).hexdigest(),
        }
    )
    return _set_source_role(row, "candidate" if split == "train" else "dev_test")


def _massive_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    data, digest = _verified_archive("massive_en")
    profile = {
        "projection": "all non-empty en-US utterances",
        "excluded": "slot-annotated utterance and raw worker identifier",
        "language_scope": "English only; translations retain shared IDs upstream",
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            handle = archive.extractfile("1.1/data/en-US.jsonl")
            if handle is None:
                raise ValueError("massive_en archive has no en-US data")
            for index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError(f"massive_en:{index} has blank row")
                source_row = json.loads(line)
                if not isinstance(source_row, dict):
                    raise ValueError(f"massive_en:{index} has invalid row")
                row = _massive_sample(source_row, index)
                counts[row["source_split"]] += 1
                yield row
        profile["raw_rows"] = dict(sorted(counts.items()))

    return rows(), {"amazon-massive-dataset-1.1.tar.gz": digest}, profile


def _coconot_sample(source_row: dict, split: str, index: int) -> dict:
    source_id = source_row.get("id")
    text = source_row.get("prompt")
    category = source_row.get("category")
    subcategory = source_row.get("subcategory")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"coconot:{split}:{index} has no id")
    if not isinstance(category, str) or not isinstance(subcategory, str):
        raise ValueError(f"coconot:{source_id} has invalid taxonomy")
    source_split = "pref:train" if split == "train" else "contrast:test"
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="coconot",
        source_split=source_split,
        source_id=f"{source_split}:{source_id}",
        group_id=f"coconot:{source_id}",
        category=category,
        goal_policy_status="safe",
        label_basis=(
            "safe_to_comply_preference_train_weak_label"
            if split == "train"
            else "human_verified_safe_to_comply_contrast"
        ),
    )
    row.update(
        {
            "source_coconot_id": source_id,
            "source_language": "en",
            "source_subcategory": subcategory,
        }
    )
    for field in ("chosen_model", "rejected_model"):
        if isinstance(source_row.get(field), str):
            row[f"source_{field}"] = source_row[field]
    role = "candidate" if split == "train" else "dev_test"
    return _set_source_role(row, role)


def _coconot_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    datasets, downloads = _parquet_dataset("coconot")

    def rows() -> Iterator[dict]:
        for split in ("train", "dev_test"):
            for index, source_row in enumerate(datasets[split]):
                yield _coconot_sample(source_row, split, index)

    profile = {
        "projection": "safe-to-comply prompt only",
        "excluded": "chosen, rejected, and contrast response text",
        "raw_rows": {role: len(dataset) for role, dataset in datasets.items()},
    }
    return rows(), downloads, profile


def _jbb_benign_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    filename, expected = FILES["jbb_benign"]["dev_test"]
    path, digest = _download("jbb_benign", filename, expected)
    source_rows = _csv_rows(
        path.read_bytes(), {"Index", "Goal", "Target", "Behavior", "Category", "Source"}
    )

    def rows() -> Iterator[dict]:
        for index, source_row in enumerate(source_rows):
            source_id = source_row["Index"]
            row = _sample(
                text=source_row["Goal"],
                label=0,
                attack_type=None,
                source="jbb_benign",
                source_split="benign",
                source_id=f"benign:{source_id}",
                group_id=f"jbb_benign:{source_id}",
                category=source_row["Category"],
                goal_policy_status="safe",
                label_basis="curated_thematically_matched_benign_behavior",
            )
            row.update(
                {
                    "source_behavior": source_row["Behavior"],
                    "source_behavior_source": source_row["Source"],
                    "source_index": source_id,
                    "source_language": "en",
                    "source_record_index": index,
                }
            )
            yield _set_source_role(row, "dev_test")

    profile = {
        "projection": "benign behavior goal only",
        "excluded": "matched affirmative target response",
        "raw_rows": len(source_rows),
    }
    return rows(), {filename: digest}, profile


def _lmsys_arena_safety(source_row: dict) -> tuple[bool, dict[str, float | bool]]:
    moderation = source_row.get("openai_moderation")
    toxic = source_row.get("toxic_chat_tag")
    if not isinstance(moderation, dict) or not isinstance(toxic, dict):
        raise ValueError("lmsys_arena row has no moderation metadata")
    roberta = toxic.get("roberta-large")
    t5 = toxic.get("t5-large")
    if not isinstance(roberta, dict) or not isinstance(t5, dict):
        raise ValueError("lmsys_arena row has invalid toxic-chat metadata")
    flags = (
        moderation.get("flagged"),
        roberta.get("flagged"),
        t5.get("flagged"),
    )
    if not all(type(flag) is bool for flag in flags):
        raise ValueError("lmsys_arena row has invalid moderation flags")
    category_scores = moderation.get("category_scores")
    if not isinstance(category_scores, dict) or not category_scores:
        raise ValueError("lmsys_arena row has no moderation scores")
    return not any(flags), {
        "source_openai_moderation_flagged": flags[0],
        "source_openai_moderation_max_score": max(
            float(score) for score in category_scores.values()
        ),
        "source_toxic_roberta_flagged": flags[1],
        "source_toxic_roberta_probability": float(roberta["probability"]),
        "source_toxic_t5_flagged": flags[2],
        "source_toxic_t5_score": float(t5["score"]),
    }


def _lmsys_arena_sample(
    source_row: dict,
    message: dict,
    safety: dict[str, float | bool],
    *,
    row_index: int,
    side: str,
    message_index: int,
    judge_sha256: str,
) -> dict:
    question_id = source_row.get("question_id")
    role = message.get("role")
    text = message.get("content")
    model = source_row.get(f"model_{side}")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError(f"lmsys_arena:{row_index} has no question id")
    if role not in {"user", "assistant"} or not isinstance(model, str):
        raise ValueError(f"lmsys_arena:{question_id} has invalid conversation schema")
    is_user = role == "user"
    user_prompt_safe = not any(
        safety[field]
        for field in (
            "source_openai_moderation_flagged",
            "source_toxic_roberta_flagged",
            "source_toxic_t5_flagged",
        )
    )
    if user_prompt_safe:
        label = 0
        security_label = None
        toxicity = "not_toxic" if is_user else "unknown"
        label_basis = (
            "three_automated_user_prompt_flags_unflagged_weak_benign"
            if is_user
            else "model_output_from_unflagged_user_prompt_weak_benign"
        )
        source_role = "candidate"
    elif is_user:
        label = None
        security_label = "uncertain"
        toxicity = "unknown"
        label_basis = "automated_user_prompt_safety_flags_unverified"
        source_role = "uncertain"
    else:
        label = None
        security_label = "uncertain"
        toxicity = "unknown"
        label_basis = "model_output_from_flagged_user_prompt_unlabelled"
        source_role = "uncertain"
    row = _sample(
        text=text,
        label=label,
        attack_type=None,
        source="lmsys_arena",
        source_split="train",
        source_id=f"{row_index}:{question_id}:{side}:{message_index}",
        group_id=f"lmsys_arena:judge:{judge_sha256}",
        category="real_world_chat",
        input_channel="direct_user" if is_user else "model_output",
        label_basis=label_basis,
        security_label=security_label,
        toxicity=toxicity,
    )
    row.update(
        {
            **safety,
            "source_anonymized": source_row.get("anony"),
            "source_content_license": (
                "CC-BY-4.0" if role == "user" else "CC-BY-NC-4.0"
            ),
            "source_judge_sha256": judge_sha256,
            "source_language": "en",
            "source_message_index": message_index,
            "source_model": model,
            "source_pair_side": side,
            "source_question_id": question_id,
            "source_role_name": role,
            "source_safety_scope": "user_prompts",
            "source_timestamp": source_row.get("tstamp"),
            "source_turn_count": source_row.get("turn"),
            "source_winner": source_row.get("winner"),
        }
    )
    return _set_source_role(row, source_role)


def _lmsys_arena_rows() -> tuple[Iterator[dict], dict[str, str], dict]:
    datasets, downloads = _parquet_dataset("lmsys_arena")
    profile = {
        "projection": (
            "non-empty English messages from unflagged prompts as weak-benign candidates; "
            "flagged user prompts and their assistant messages retained as uncertain"
        ),
        "excluded": [
            "non-English conversations",
            "raw anonymized judge identifier",
            "full moderation category-score vector",
        ],
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        for row_index, source_row in enumerate(datasets["train"]):
            counts["raw_conversations"] += 1
            safe, safety = _lmsys_arena_safety(source_row)
            if source_row.get("language") != "English":
                counts["non_english_conversations_omitted"] += 1
                continue
            if not safe:
                counts["flagged_conversations_retained"] += 1
            judge = source_row.get("judge")
            if not isinstance(judge, str) or not judge:
                raise ValueError(f"lmsys_arena:{row_index} has no judge lineage")
            judge_sha256 = hashlib.sha256(judge.encode()).hexdigest()
            counts["retained_conversations"] += 1
            for side in ("a", "b"):
                conversation = source_row.get(f"conversation_{side}")
                if not isinstance(conversation, list):
                    raise ValueError(
                        f"lmsys_arena:{row_index} has invalid conversation"
                    )
                for message_index, message in enumerate(conversation):
                    if not isinstance(message, dict):
                        raise ValueError(f"lmsys_arena:{row_index} has invalid message")
                    text = message.get("content")
                    if isinstance(text, str) and not text.strip():
                        counts["empty_messages_omitted"] += 1
                        continue
                    output = _lmsys_arena_sample(
                        source_row,
                        message,
                        safety,
                        row_index=row_index,
                        side=side,
                        message_index=message_index,
                        judge_sha256=judge_sha256,
                    )
                    counts[f"retained_role:{message.get('role')}"] += 1
                    yield output
        profile.update(
            {
                "raw_conversations": counts["raw_conversations"],
                "retained_conversations": counts["retained_conversations"],
                "non_english_conversations_omitted": counts[
                    "non_english_conversations_omitted"
                ],
                "flagged_conversations_retained": counts[
                    "flagged_conversations_retained"
                ],
                "empty_messages_omitted": counts["empty_messages_omitted"],
                "retained_roles": {
                    role.removeprefix("retained_role:"): value
                    for role, value in sorted(counts.items())
                    if role.startswith("retained_role:")
                },
            }
        )

    return rows(), downloads, profile


LOADERS = {
    "gandalf": _gandalf_rows,
    "llmail": _llmail_rows,
    "tensor_trust_raw": _tensor_trust_rows,
    "browsesafe": _browsesafe_rows,
    "hackaprompt": _hackaprompt_rows,
    "wildjailbreak": _wildjailbreak_rows,
    "wildguardmix": _wildguard_rows,
    "taskmaster": _taskmaster_rows,
    "banking77": _banking77_rows,
    "false_reject": _false_reject_rows,
    "schema_guided_dialogue": _schema_guided_dialogue_rows,
    "massive_en": _massive_rows,
    "coconot": _coconot_rows,
    "jbb_benign": _jbb_benign_rows,
    "lmsys_arena": _lmsys_arena_rows,
}


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


def build_corpus(data_dir: Path, *, core_manifest_path: Path) -> dict:
    disable_progress_bars()
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    source_dir = data_dir / "sources"
    source_outputs = {}
    source_profiles = {}
    downloads = {}
    for source, loader in LOADERS.items():
        rows, source_downloads, source_profile = loader()
        summary = _consume_source(source_dir / f"{source}.jsonl", rows)
        summary["path"] = str((source_dir / f"{source}.jsonl").relative_to(data_dir))
        source_outputs[source] = summary
        source_profiles[source] = source_profile
        downloads[source] = source_downloads

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
            "routing": routing_quarantine,
        },
        "routing_deduplication": routing_stats,
    }
    atomic_write_text(
        data_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


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
