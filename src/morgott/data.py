from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from datasets import disable_progress_bars

SOURCES = {
    "toxic_chat": {
        "repo": "lmsys/toxic-chat",
        "revision": "29df8e4dba60e1f4af4b4075c0705c5b313548a8",
        "license": "CC-BY-NC-4.0",
        "url": "https://huggingface.co/datasets/lmsys/toxic-chat",
        "use": "train and same-source test; explicit jailbreak label",
    },
    "prompt_injections": {
        "repo": "deepset/prompt-injections",
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/deepset/prompt-injections",
        "use": "train and same-source test; direct prompt-injection label",
    },
    "xstest": {
        "revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
        "license": "CC-BY-4.0",
        "url": "https://github.com/paul-rottger/xstest",
        "use": "hard-negative test; safe and unsafe requests are not attacks",
    },
    "multi_turn": {
        "repo": "tom-gibbs/multi-turn_jailbreak_attack_datasets",
        "revision": "e3b30257c4d6be5438ea19f0989ac82c24234fe4",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets",
        "use": "out-of-source obfuscated-jailbreak test grouped by goal",
    },
    "oasst1": {
        "repo": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "use": "multilingual weak injection controls; auxiliary for broad routing",
    },
    "harmbench": {
        "revision": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        "license": "MIT",
        "url": "https://github.com/centerforaisafety/HarmBench",
        "use": "held-out harmful-goal non-injection negatives",
    },
    "do_not_answer": {
        "revision": "460703484df354958a5e1cd7378a38fcb94a2f3e",
        "license": "CC-BY-NC-SA-4.0",
        "url": "https://github.com/libr-ai/do-not-answer",
        "use": "held-out harmful-goal non-injection negatives",
    },
    "bipia": {
        "revision": "a004b69ec0dd446e0afd461d98cb5e96e120a5d0",
        "license": "MIT attacks; mixed benchmark context licenses",
        "url": "https://github.com/microsoft/BIPIA",
        "use": "channel-specific indirect-injection train/test with clean-context controls",
    },
    "notinject": {
        "revision": "1b5751e88bf7475acbedfc8eda795ce060307c84",
        "license": "MIT",
        "url": "https://github.com/leolee99/PIGuard",
        "use": "locked trigger-word hard negatives for measuring over-defense",
    },
    "jailbreaks_over_time": {
        "revision": "94a2e998282301d545f92177e3fff8aab11fb0dd",
        "license": "MIT",
        "url": "https://github.com/wagner-group/JailbreaksOverTime",
        "use": "source-held-out temporal distribution-shift evaluation only",
    },
    "tensor_trust": {
        "revision": "747a75e096761ebc01bd3970158827326b4add23",
        "license": "public research release; no explicit standard dataset license",
        "url": "https://github.com/HumanCompatibleAI/tensor-trust-data",
        "use": "human prompt-injection robustness evaluation only; never training",
    },
    "nemotron_agentic_ipi": {
        "repo": "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "revision": "d738d4f361cc38bb4d7a42b9066776dade5332f5",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "use": "synthetic successful agentic indirect-injection evaluation only; never direct-user training",
        "path": "train.jsonl",
        "bytes": 11_703_379,
        "sha256": "3329da17564a7eb287e2730fc7d6956e1f4fe51e8950ac4f110b3c37e78cf3b9",
    },
    "gandalf": {
        "repo": "Lakera/gandalf_ignore_instructions",
        "revision": "04737b65e90a6794ec227012e4a255a7def6344b",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        "use": "human direct-injection data; official train only for fitting",
    },
    "llmail": {
        "repo": "microsoft/llmail-inject-challenge",
        "revision": "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/microsoft/llmail-inject-challenge",
        "use": "human adaptive email injection; phase 1 fit, phase 2 evaluation",
    },
    "tensor_trust_raw": {
        "repo": "qxcv/tensor-trust",
        "revision": "4de2b2fe01ba0cb6fbf7cbb9f1a3fabaf8157372",
        "license": "no standard dataset license declared",
        "url": "https://huggingface.co/datasets/qxcv/tensor-trust",
        "use": "human game attack attempts; grouped development data",
    },
    "browsesafe": {
        "repo": "perplexity-ai/browsesafe-bench",
        "revision": "b506fb5bc7fd4472c8738055a67a0ef6406afdc9",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/perplexity-ai/browsesafe-bench",
        "use": "whole-document browser injection train and official test",
    },
    "hackaprompt": {
        "repo": "hackaprompt/hackaprompt-dataset",
        "revision": "25b87fbedfb86840abaf8cd09af7a029208a971a",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset",
        "use": "gated human direct attack attempts; user_input only",
        "gated": True,
    },
    "wildjailbreak": {
        "repo": "allenai/wildjailbreak",
        "revision": "5ddc12a7894f842b0619b8e1c7ee496b198af009",
        "license": "ODC-BY",
        "url": "https://huggingface.co/datasets/allenai/wildjailbreak",
        "use": "gated four-way harmful/benign and adversarial contrast data",
        "gated": True,
    },
    "wildguardmix": {
        "repo": "allenai/wildguardmix",
        "revision": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
        "license": "ODC-BY",
        "url": "https://huggingface.co/datasets/allenai/wildguardmix",
        "use": "gated prompt harmfulness data for the routing target",
        "gated": True,
    },
    "harper_valley_bank": {
        "revision": "0bd721e877c4a85d8c13ff837e68661ea6200a98",
        "license": "CC-BY-4.0",
        "url": "https://github.com/cricketclub/gridspace-stanford-harper-valley",
        "use": "simulated human-human banking calls; caller and agent channels retained separately",
    },
    "tatqa": {
        "revision": "870accc41953dcde885aabeb963d94aabdc0fbc3",
        "license": "CC-BY-4.0",
        "url": "https://github.com/NExTplusplus/TAT-QA",
        "use": "finance-QA questions and report contexts; task construction is not safety adjudication",
    },
    "financebench": {
        "revision": "cc39aeb4afdf33909ee1412188bf89035950c2eb",
        "license": "no explicit dataset license declared; public open-source sample",
        "url": "https://github.com/patronus-ai/financebench",
        "use": "150 public finance-QA examples as development-only hard-benign diagnostics",
    },
    "mind2web": {
        "repo": "osunlp/Mind2Web",
        "revision": "17ece8eb89862368edc0cc806acee6fca5163474",
        "conversion_revision": "eabe74c3532cf3a35ff02913cece5341bd1ca0d5",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/osunlp/Mind2Web",
        "use": "confirmed official training tasks only, after local secret and PII quarantine",
    },
    "swebench_verified": {
        "repo": "SWE-bench/SWE-bench_Verified",
        "revision": "91aa3ed51b709be6457e12d00300a6a596d4c6a3",
        "license": "no dataset-level license declared; upstream repositories vary",
        "url": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified",
        "use": "human-verified software issue statements as a dev-test-only long-benign FPR slice",
    },
    "taskmaster": {
        "revision": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "license": "CC-BY-4.0",
        "url": "https://github.com/google-research-datasets/Taskmaster",
        "use": "English task-oriented user and assistant turns for benign routing balance",
        "archive_url": "https://codeload.github.com/google-research-datasets/Taskmaster/tar.gz/d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "bytes": 138_453_808,
        "sha256": "c7e4774798ace96e3b413a9beef6d2a706458d66c4e87ef1061e9426db1a3c46",
    },
    "banking77": {
        "revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "license": "CC-BY-4.0",
        "url": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "use": "English online-banking intent queries as finance hard negatives",
    },
    "false_reject": {
        "repo": "AmazonScience/FalseReject",
        "revision": "493ba967714ea54c6f01067e1f61e389cc2c9b3e",
        "license": "CC-BY-NC-4.0",
        "url": "https://huggingface.co/datasets/AmazonScience/FalseReject",
        "use": "hard-benign prompts; generated candidates and human test held out",
    },
    "schema_guided_dialogue": {
        "revision": "e852981ae34990f4358979625854259302feaa78",
        "license": "CC-BY-SA-4.0",
        "url": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
        "use": "English crowdworker task-dialogue turns for benign routing balance",
        "archive_url": "https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/tar.gz/e852981ae34990f4358979625854259302feaa78",
        "bytes": 36_792_911,
        "sha256": "ff97a9ab52b4cc9f25e1a093c96431512465e8377f9a1f57dc710a10484d2188",
    },
    "massive_en": {
        "revision": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/AmazonScience/massive",
        "use": "English voice-assistant utterances for benign intent coverage",
        "archive_url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        "bytes": 40_251_390,
        "sha256": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    },
    "coconot": {
        "repo": "allenai/coconot",
        "revision": "2cbe16aabf9069f17e48c8daad8aeabc29469eb7",
        "license": "ODC-BY-1.0 + component licenses",
        "url": "https://huggingface.co/datasets/allenai/coconot",
        "use": "safe-to-comply prompts for weak development and hard-benign evaluation",
    },
    "jbb_benign": {
        "repo": "JailbreakBench/JBB-Behaviors",
        "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        "use": "curated benign behaviors thematically matched to misuse requests",
    },
    "lmsys_arena": {
        "repo": "lmsys/chatbot_arena_conversations",
        "revision": "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
        "license": "CC-BY-4.0 prompts; CC-BY-NC-4.0 model outputs",
        "url": "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations",
        "use": "English Arena messages with weak-benign candidates and flagged conversations retained as uncertain",
        "gated": True,
    },
    "agentic_boundary_pairs": {
        "repo": "3nesdeniz/agentic-prompt-injection-boundary-pairs",
        "revision": "a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs",
        "use": "auxiliary paired instruction-subversion training and authorization diagnostics",
    },
}

MAX_DOWNLOAD_BYTES = 60_000_000

SECURITY_LABELS = {
    "benign",
    "direct_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "harmful_non_injection",
    "uncertain",
}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def split_is_validation(row: dict) -> bool:
    group = row.get("split_group_id", row.get("group_id"))
    if not isinstance(group, str) or not group:
        raise ValueError("row has no split group")
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:2]) % 5 == 0


def materialize_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = []
    validation = []
    for row in rows:
        target = validation if split_is_validation(row) else train
        row["data_role"] = "validation" if target is validation else "train"
        target.append(row)
    return train, validation


def manifest_output_path(data_dir: Path, output: dict) -> Path:
    return data_dir / output["path"]


def iter_verified_jsonl(path: Path, expected_sha256: str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                yield json.loads(line)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{path.name} changed: expected {expected_sha256}, got {actual}"
        )


def read_verified_jsonl(path: Path, expected_sha256: str) -> list[dict]:
    return list(iter_verified_jsonl(path, expected_sha256))


def _sample(
    *,
    text: str,
    label: int | None,
    attack_type: str | None,
    source: str,
    source_split: str,
    source_id: str,
    group_id: str,
    split_group_id: str | None = None,
    category: str | None = None,
    goal_policy_status: str = "unknown",
    input_channel: str = "direct_user",
    label_basis: str = "source_annotation",
    security_label: str | None = None,
    toxicity: str = "unknown",
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source}:{source_id} has empty text")
    if label not in (0, 1, None):
        raise ValueError(f"{source}:{source_id} has invalid injection label")
    if goal_policy_status not in {"safe", "unsafe", "unknown"}:
        raise ValueError(f"{source}:{source_id} has invalid goal policy status")
    if input_channel not in {
        "direct_user",
        "model_output",
        "trusted_instruction",
        "untrusted_content",
    }:
        raise ValueError(f"{source}:{source_id} has invalid input channel")
    if label is None and security_label != "uncertain":
        raise ValueError(
            f"{source}:{source_id} requires security_label=uncertain for an unknown "
            "injection label"
        )
    if security_label is None:
        if label:
            if attack_type in {
                "jailbreak",
                "direct_jailbreak",
                "obfuscated_jailbreak",
            }:
                security_label = "direct_jailbreak"
            elif attack_type == "indirect_prompt_injection":
                security_label = "indirect_prompt_injection"
            else:
                security_label = "direct_prompt_injection"
        elif goal_policy_status == "unsafe" or toxicity == "toxic":
            security_label = "harmful_non_injection"
        else:
            security_label = "benign"
    if security_label not in SECURITY_LABELS:
        raise ValueError(f"{source}:{source_id} has invalid security label")
    injection_security_labels = {
        "direct_jailbreak",
        "direct_prompt_injection",
        "indirect_prompt_injection",
    }
    if (security_label in injection_security_labels) != (label == 1):
        raise ValueError(
            f"{source}:{source_id} has inconsistent injection and security labels"
        )
    if toxicity not in {"toxic", "not_toxic", "unknown"}:
        raise ValueError(f"{source}:{source_id} has invalid toxicity")
    if security_label == "benign" and (
        goal_policy_status == "unsafe" or toxicity == "toxic"
    ):
        raise ValueError(f"{source}:{source_id} has contradictory benign annotations")
    info = SOURCES[source]
    normalized_hash = text_hash(text)
    split_group = str(split_group_id or group_id)
    security_tags = []
    if security_label == "benign":
        security_tags.append("benign")
    elif security_label == "uncertain":
        security_tags.append("uncertain")
    else:
        security_tags.append(security_label)
    if security_label in injection_security_labels:
        security_tags.append("instruction_subversion")
    if goal_policy_status == "unsafe":
        security_tags.append("harmful_intent")
    if toxicity == "toxic":
        security_tags.append("toxic")
    security_tags = sorted(set(security_tags))
    routing_label = int(security_label != "benign" or toxicity == "toxic")
    origin = {
        "source": source,
        "source_revision": info["revision"],
        "source_split": source_split,
        "source_id": str(source_id),
        "group_id": str(group_id),
        "split_group_id": split_group,
        "label_basis": label_basis,
        "license": info["license"],
        "injection_label": label,
        "routing_label": routing_label,
        "security_label": security_label,
        "security_tags": security_tags,
        "toxicity": toxicity,
        "attack_type": attack_type,
        "goal_policy_status": goal_policy_status,
        "category": category,
        "input_channel": input_channel,
    }
    return {
        "schema_version": 5,
        "id": f"{source}:{source_id}",
        "text": text,
        "label": label,
        "injection_label": label,
        "routing_label": routing_label,
        "security_label": security_label,
        "security_tags": security_tags,
        "injection_subtype_training_eligible": label is not None,
        "toxicity": toxicity,
        "attack_type": attack_type,
        "goal_policy_status": goal_policy_status,
        "category": category,
        "input_channel": input_channel,
        "label_basis": label_basis,
        "source": source,
        "source_split": source_split,
        "source_id": str(source_id),
        "group_id": str(group_id),
        "split_group_id": split_group,
        "source_revision": info["revision"],
        "license": info["license"],
        "normalized_text_sha256": normalized_hash,
        "origins": [origin],
    }


def _set_source_role(row: dict, role: str) -> dict:
    if role not in {"candidate", "dev_test", "auxiliary", "uncertain"}:
        raise ValueError(f"invalid source role: {role}")
    routing_training_eligible = role in {"candidate", "dev_test"}
    row["source_role"] = role
    row["routing_training_eligible"] = routing_training_eligible
    row["origins"][-1].update(
        {
            "source_role": role,
            "routing_training_eligible": routing_training_eligible,
            **{key: value for key, value in row.items() if key.startswith("source_")},
        }
    )
    return row


def _fetch(
    url: str,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "morgott/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                data = response.read(max_bytes + 1)
        except (
            ConnectionError,
            TimeoutError,
            http.client.IncompleteRead,
            urllib.error.URLError,
        ):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
            continue
        if len(data) > max_bytes:
            raise ValueError(f"download exceeded {max_bytes} bytes: {url}")
        digest = hashlib.sha256(data).hexdigest()
        if (
            content_length is not None
            and len(data) != int(content_length)
            or expected_bytes is not None
            and len(data) != expected_bytes
            or expected_sha256 is not None
            and digest != expected_sha256
        ):
            if attempt == 2:
                raise ValueError(f"download does not match pinned metadata: {url}")
            time.sleep(2**attempt)
            continue
        return data, digest
    raise AssertionError("unreachable")


def _csv_rows(data: bytes, required: set[str]) -> list[dict[str, str]]:
    csv.field_size_limit(10_000_000)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    columns = set(reader.fieldnames or [])
    if not required <= columns:
        raise ValueError(f"missing CSV columns: {sorted(required - columns)}")
    return list(reader)


def _github_raw(source: str, path: str) -> tuple[bytes, str]:
    revision = SOURCES[source]["revision"]
    return _fetch(
        f"https://raw.githubusercontent.com/{SOURCES[source]['url'].split('github.com/')[1]}/"
        f"{revision}/{path}"
    )


def deduplicate(
    rows: Iterable[dict],
    blocked: set[str] | dict[str, set[tuple[object, ...]]] | None = None,
    *,
    quarantine: list[dict] | None = None,
    label_fields: tuple[str, ...] = ("label", "routing_label"),
    blocked_reason: str = "exact_reference_overlap",
) -> tuple[list[dict], dict[str, int]]:
    blocked = blocked or set()
    groups: dict[str, list[dict]] = {}
    stats = {
        "blocked_by_reference": 0,
        "blocked_label_conflicts": 0,
        "duplicates": 0,
        "label_conflicts": 0,
    }
    for row in rows:
        key = row.get("normalized_text_sha256") or text_hash(row["text"])
        if key in blocked:
            stats["blocked_by_reference"] += 1
            blocked_labels = blocked.get(key) if isinstance(blocked, dict) else None
            row_labels = tuple(row.get(field, row["label"]) for field in label_fields)
            if blocked_labels is not None and row_labels not in blocked_labels:
                stats["blocked_label_conflicts"] += 1
            if quarantine is not None:
                quarantined = dict(row)
                quarantined["data_role"] = "quarantine"
                quarantined["quarantine_reason"] = blocked_reason
                quarantine.append(quarantined)
            continue
        groups.setdefault(key, []).append(row)

    kept = []
    for matches in groups.values():
        labels = {
            tuple(row.get(field, row["label"]) for field in label_fields)
            for row in matches
        }
        if len(labels) > 1:
            stats["label_conflicts"] += len(matches)
            if quarantine is not None:
                for row in matches:
                    quarantined = dict(row)
                    quarantined["data_role"] = "quarantine"
                    quarantined["quarantine_reason"] = "exact_label_conflict"
                    quarantine.append(quarantined)
            continue
        representative = dict(matches[0])
        origins = []
        seen_origins = set()
        for row in matches:
            for origin in row.get("origins", []):
                key = json.dumps(origin, sort_keys=True, separators=(",", ":"))
                if key not in seen_origins:
                    seen_origins.add(key)
                    origins.append(origin)
        if origins:
            representative["origins"] = origins
        annotation_fields = (
            "injection_label",
            "security_label",
            "security_tags",
            "toxicity",
            "goal_policy_status",
            "input_channel",
            "attack_type",
        )
        if any(field in row for row in matches for field in annotation_fields):
            disagreement = [
                field
                for field in annotation_fields
                if any(
                    row.get(field) != matches[0].get(field)
                    or type(row.get(field)) is not type(matches[0].get(field))
                    for row in matches[1:]
                )
            ]
            representative["security_tags"] = sorted(
                {tag for row in matches for tag in row.get("security_tags", [])}
            )
            injection_disagreement = {
                "injection_label",
                "security_label",
                "input_channel",
                "attack_type",
            } & set(disagreement)
            representative["injection_subtype_training_eligible"] = (
                not injection_disagreement
                and all(
                    row.get("injection_subtype_training_eligible", False)
                    for row in matches
                )
            )
            if disagreement:
                representative["annotation_disagreement_fields"] = disagreement
                neutral_values = {
                    "security_label": "uncertain",
                    "toxicity": "unknown",
                    "goal_policy_status": "unknown",
                    "input_channel": "mixed",
                    "attack_type": None,
                }
                for field, value in neutral_values.items():
                    if field in disagreement:
                        representative[field] = value
                if "injection_label" in disagreement:
                    representative["injection_label"] = None
                    representative["label"] = None
                if "routing_label" in representative:
                    representative["routing_label"] = int(
                        representative.get("security_label") != "benign"
                        or representative.get("toxicity") == "toxic"
                    )
        kept.append(representative)
        stats["duplicates"] += len(matches) - 1
    return kept, stats


@contextmanager
def _atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # The dotted prefix marks strays left by a killed process so
        # scripts/azsync.sh can exclude them from pushes.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".tmp-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            yield handle
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with _atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def atomic_write_text(path: Path, value: str) -> None:
    with _atomic_text_writer(path) as handle:
        handle.write(value)


def _write_json(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _consume_source_rows(
    path: Path, rows: Iterable[dict], *, source: str | None = None
) -> dict:
    """Stream canonical rows to a shard and return its manifest summary."""

    counts = Counter()
    row_ids = set()
    with _atomic_text_writer(path) as handle:
        for row in rows:
            if row["id"] in row_ids:
                where = f" in {source}" if source else ""
                raise ValueError(f"duplicate canonical row id{where}: {row['id']}")
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
    return {
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


def _set_core_routing_role(source: str, row: dict) -> None:
    """Assign only source-supported broad-routing supervision."""

    if "source_role" in row:
        return
    eligible = True
    if source == "toxic_chat":
        eligible = row["injection_label"] == 1 or row["toxicity"] == "toxic"
    elif source in {"prompt_injections", "jailbreaks_over_time"}:
        eligible = row["injection_label"] == 1
    elif source == "bipia":
        eligible = row["injection_label"] == 1
    if not eligible:
        _set_source_role(row, "auxiliary")
        return
    candidate = source in {"toxic_chat", "prompt_injections", "bipia"} and row[
        "source_split"
    ].startswith("train")
    _set_source_role(row, "candidate" if candidate else "dev_test")


def build_dataset(data_dir: Path = Path("data"), *, manifest_path: Path) -> dict:
    # Imported at call time: sources.core depends on this module's contract
    # helpers, so a top-level import would be circular.
    from .sources.core import (
        _load_bipia,
        _load_do_not_answer,
        _load_harmbench,
        _load_jailbreaks_over_time,
        _load_multi_turn,
        _load_nemotron_agentic_ipi,
        _load_notinject,
        _load_oasst1,
        _load_prompt_injections,
        _load_tensor_trust,
        _load_toxic_chat,
        _load_xstest,
        _oasst_position_stress,
    )

    disable_progress_bars()
    source_dir = data_dir / "sources"
    injection_dir = data_dir / "views" / "injection"
    audit_dir = data_dir / "audits"
    quarantine_dir = data_dir / "quarantine"

    toxic = _load_toxic_chat()
    injections = _load_prompt_injections()
    oasst1, oasst1_profile = _load_oasst1()
    xstest, xstest_sha = _load_xstest()
    harmbench, harmbench_sha = _load_harmbench()
    do_not_answer, do_not_answer_sha = _load_do_not_answer()
    multi_turn, multi_turn_sha, multi_turn_profile = _load_multi_turn()
    bipia, bipia_sha = _load_bipia()
    notinject, notinject_sha = _load_notinject()
    jailbreaks_over_time, jailbreaks_over_time_sha = _load_jailbreaks_over_time()
    tensor_trust, tensor_trust_sha = _load_tensor_trust()
    nemotron_agentic_ipi, nemotron_agentic_ipi_sha, nemotron_agentic_ipi_profile = (
        _load_nemotron_agentic_ipi()
    )

    source_rows = {
        "toxic_chat": toxic["train"] + toxic["test"],
        "prompt_injections": injections["train"] + injections["test"],
        "oasst1": oasst1["source"],
        "xstest": xstest,
        "harmbench": harmbench,
        "do_not_answer": [
            row
            for language in ("english", "chinese", "translated_english")
            for row in do_not_answer[language]
        ],
        "multi_turn": [
            row
            for family in ("harmful", "semi_benign", "benign", "complete_harmful")
            for row in multi_turn[family]
        ],
        "bipia": [
            row
            for split in ("train", "test")
            for kind in ("payload", "context", "clean")
            for row in bipia[split][kind]
        ],
        "notinject": notinject,
        "jailbreaks_over_time": jailbreaks_over_time,
        "tensor_trust": tensor_trust["attack"] + tensor_trust["context"],
        "nemotron_agentic_ipi": nemotron_agentic_ipi,
    }
    train_raw = toxic["train"] + injections["train"] + oasst1["train"]
    indirect_train_raw = (
        bipia["train"]["payload"] + bipia["train"]["context"] + bipia["train"]["clean"]
    )
    source_outputs = {}
    for name, rows in source_rows.items():
        for row in rows:
            _set_core_routing_role(name, row)
        path = source_dir / f"{name}.jsonl"
        source_outputs[name] = {
            "path": str(path.relative_to(data_dir)),
            **_consume_source_rows(path, rows, source=name),
        }

    quarantine = []
    train, train_dedup = deduplicate(
        train_raw, quarantine=quarantine, label_fields=("label",)
    )
    direct_train, direct_validation = materialize_split(train)
    indirect_train, indirect_train_dedup = deduplicate(
        indirect_train_raw, quarantine=quarantine, label_fields=("label",)
    )
    indirect_train_rows, indirect_validation = materialize_split(indirect_train)
    training_labels: dict[str, set[tuple[int]]] = {}
    indirect_training_labels: dict[str, set[tuple[int]]] = {}
    for row in train:
        training_labels.setdefault(row["normalized_text_sha256"], set()).add(
            (row["label"],)
        )
    for row in indirect_train:
        indirect_training_labels.setdefault(row["normalized_text_sha256"], set()).add(
            (row["label"],)
        )

    eval_sets = {
        "toxic_chat": toxic["test"],
        "prompt_injections": injections["test"],
        "xstest": xstest,
        "oasst1_chat": oasst1["validation"],
        "oasst1_position_stress": _oasst_position_stress(oasst1["validation"]),
        "harmbench": harmbench,
        "do_not_answer": do_not_answer["english"],
        "do_not_answer_chinese": do_not_answer["chinese"],
        "do_not_answer_translated_english": do_not_answer["translated_english"],
        "multi_turn": multi_turn["harmful"],
        "multi_turn_semi_benign": multi_turn["semi_benign"],
        "multi_turn_benign": multi_turn["benign"],
        "bipia_clean_context": bipia["test"]["clean"],
        "bipia_payload": bipia["test"]["payload"],
        "bipia_context": bipia["test"]["context"],
        "notinject": notinject,
        "jailbreaks_over_time": jailbreaks_over_time,
        "tensor_trust_attack": tensor_trust["attack"],
        "tensor_trust_context": tensor_trust["context"],
        "nemotron_agentic_ipi": nemotron_agentic_ipi,
    }
    dedup = {
        "train": train_dedup,
        "indirect_train": indirect_train_dedup,
    }
    output_rows = {}
    for name, rows in eval_sets.items():
        blocked = training_labels
        if rows and any(row["input_channel"] == "untrusted_content" for row in rows):
            blocked = dict(training_labels)
            for key, labels in indirect_training_labels.items():
                blocked.setdefault(key, set()).update(labels)
        output_rows[name], dedup[name] = deduplicate(
            rows,
            blocked,
            quarantine=quarantine,
            label_fields=("label",),
            blocked_reason="exact_train_overlap",
        )
        for row in output_rows[name]:
            row["data_role"] = "dev_test"

    outputs = {}
    for name, rows in output_rows.items():
        path = injection_dir / f"{name}.jsonl"
        outputs[name] = {
            "path": str(path.relative_to(data_dir)),
            "rows": len(rows),
            "positive": sum(row["label"] for row in rows),
            "negative": sum(not row["label"] for row in rows),
            "routing_non_benign": sum(row["routing_label"] for row in rows),
            "routing_benign": sum(not row["routing_label"] for row in rows),
            "security_labels": dict(
                sorted(Counter(row["security_label"] for row in rows).items())
            ),
            "sha256": _write_jsonl(path, rows),
        }

    split_rows = {
        "direct_train": direct_train,
        "direct_validation": direct_validation,
        "indirect_train": indirect_train_rows,
        "indirect_validation": indirect_validation,
    }
    split_outputs = {}
    for name, rows in split_rows.items():
        path = injection_dir / f"{name}.jsonl"
        split_outputs[name] = {
            "path": str(path.relative_to(data_dir)),
            "rows": len(rows),
            "positive": sum(row["label"] for row in rows),
            "negative": sum(not row["label"] for row in rows),
            "routing_non_benign": sum(row["routing_label"] for row in rows),
            "routing_benign": sum(not row["routing_label"] for row in rows),
            "sha256": _write_jsonl(path, rows),
        }

    quarantine_path = quarantine_dir / "injection.jsonl"
    quarantine_output = {
        "path": str(quarantine_path.relative_to(data_dir)),
        "rows": len(quarantine),
        "reasons": dict(
            sorted(Counter(row["quarantine_reason"] for row in quarantine).items())
        ),
        "sha256": _write_jsonl(quarantine_path, quarantine),
    }

    from .overlap import NEAR_METHOD, audit_near_overlaps

    near_overlaps = audit_near_overlaps(
        {"direct_train": direct_train, "indirect_train": indirect_train_rows},
        {
            "direct_validation": direct_validation,
            "indirect_validation": indirect_validation,
            **output_rows,
        },
    )
    near_overlap_path = audit_dir / "injection_near_overlap.jsonl"
    near_overlap_output = {
        "path": str(near_overlap_path.relative_to(data_dir)),
        "rows": len(near_overlaps),
        "candidate_rows_by_dataset": dict(
            sorted(Counter(row["candidate_dataset"] for row in near_overlaps).items())
        ),
        "sha256": _write_jsonl(near_overlap_path, near_overlaps),
    }
    active_sources = sorted(source_outputs)

    manifest = {
        "schema_version": 5,
        "canonical_row_schema_version": 5,
        "targets": {
            "injection_label": (
                "channel-scoped jailbreak or prompt-injection attack attempt; retained "
                "for the existing sensors"
            ),
            "routing_label": (
                "0 only for a source-supported benign row; 1 for injection, jailbreak, "
                "harmful non-injection, toxic, or uncertain content requiring a "
                "downstream decision"
            ),
            "security_tags": (
                "independent tags for instruction subversion, subtype, harmful intent, "
                "toxicity, benignness, or unresolved labels; use masked multi-task "
                "losses rather than forcing mutually exclusive classes"
            ),
            "injection_subtype_training_eligible": (
                "false when injection subtype is unknown or exact-duplicate injection "
                "annotations disagree; routing, harmfulness, or toxicity supervision "
                "may still be valid"
            ),
        },
        "sources": {source: SOURCES[source] for source in active_sources},
        "download_sha256": {
            "xstest_prompts.csv": xstest_sha,
            "harmbench_behaviors_text_all.csv": harmbench_sha,
            **{
                f"do_not_answer/{path}": digest
                for path, digest in do_not_answer_sha.items()
            },
            **{f"multi_turn/{path}": digest for path, digest in multi_turn_sha.items()},
            **{f"bipia/{path}": digest for path, digest in bipia_sha.items()},
            **{f"notinject/{path}": digest for path, digest in notinject_sha.items()},
            "jailbreaks_over_time/jailbreaksovertime.json": jailbreaks_over_time_sha,
            **{
                f"tensor_trust/{path}": digest
                for path, digest in tensor_trust_sha.items()
            },
            "nemotron_agentic_ipi/train.jsonl": nemotron_agentic_ipi_sha,
        },
        "source_profiles": {
            "multi_turn": multi_turn_profile,
            "nemotron_agentic_ipi": nemotron_agentic_ipi_profile,
            "oasst1": oasst1_profile,
        },
        "source_outputs": source_outputs,
        "deduplication": dedup,
        "injection_views": {**split_outputs, **outputs},
        "quarantines": {"injection": quarantine_output},
        "audits": {
            "injection_near_overlap": {
                "method": NEAR_METHOD,
                "scope": "materialized train rows against validation and dev-test rows",
                "action": (
                    "audited but not removed from the retained baseline; new rows "
                    "matching locked evaluation are quarantined before training"
                ),
                **near_overlap_output,
            }
        },
    }
    _write_json(manifest_path, manifest)
    return manifest
