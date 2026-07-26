from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from datasets import disable_progress_bars, load_dataset

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_toxic_chat() -> dict[str, list[dict]]:
    info = SOURCES["toxic_chat"]
    dataset = load_dataset(info["repo"], "toxicchat0124", revision=info["revision"])
    output: dict[str, list[dict]] = {}
    for split in ("train", "test"):
        output[split] = []
        for row in dataset[split]:
            if (
                type(row["jailbreaking"]) is not int
                or row["jailbreaking"] not in (0, 1)
                or type(row["toxicity"]) is not int
                or row["toxicity"] not in (0, 1)
            ):
                raise ValueError(f"toxic_chat:{split} has invalid binary labels")
            sample = _sample(
                text=row["user_input"],
                label=row["jailbreaking"],
                attack_type=("direct_jailbreak" if row["jailbreaking"] else None),
                source="toxic_chat",
                source_split=split,
                source_id=row["conv_id"],
                group_id=f"toxic_chat:{row['conv_id']}",
                category="toxic" if row["toxicity"] else "not_toxic",
                goal_policy_status=("unsafe" if row["toxicity"] else "unknown"),
                toxicity="toxic" if row["toxicity"] else "not_toxic",
                label_basis="official_jailbreak_and_toxicity_annotations",
            )
            sample.update(
                {
                    "source_human_annotation": row.get("human_annotation"),
                    "source_openai_moderation": row.get("openai_moderation"),
                }
            )
            output[split].append(sample)
    return output


def _load_prompt_injections() -> dict[str, list[dict]]:
    info = SOURCES["prompt_injections"]
    dataset = load_dataset(info["repo"], revision=info["revision"])
    output: dict[str, list[dict]] = {}
    for split in ("train", "test"):
        output[split] = []
        for row in dataset[split]:
            digest = text_hash(row["text"])
            output[split].append(
                _sample(
                    text=row["text"],
                    label=row["label"],
                    attack_type="direct_prompt_injection" if row["label"] else None,
                    source="prompt_injections",
                    source_split=split,
                    source_id=digest,
                    group_id=f"prompt_injections:{digest}",
                )
            )
    return output


def _load_xstest() -> tuple[list[dict], str]:
    revision = SOURCES["xstest"]["revision"]
    url = (
        "https://raw.githubusercontent.com/paul-rottger/xstest/"
        f"{revision}/xstest_prompts.csv"
    )
    data, digest = _fetch(url)
    rows = _csv_rows(data, {"id", "prompt", "type", "label"})
    return [
        _sample(
            text=row["prompt"],
            label=0,
            attack_type=None,
            source="xstest",
            source_split="test",
            source_id=row["id"],
            group_id=f"xstest:{row['id']}",
            category=row["type"],
            goal_policy_status=row["label"],
            label_basis="benchmark_nonattack",
        )
        for row in rows
    ], digest


def _load_multi_turn() -> tuple[dict[str, list[dict]], dict[str, str], dict]:
    revision = SOURCES["multi_turn"]["revision"]
    specs = {
        "harmful": ("Harmful Dataset.csv", 1, "benchmark_attack"),
        "semi_benign": (
            "Semi-Benign Dataset.csv",
            0,
            "generated_semantically_benign_control_with_harmful_terms",
        ),
        "benign": (
            "Completely-Benign Dataset.csv",
            0,
            "generated_benign_control",
        ),
    }
    output: dict[str, list[dict]] = {}
    downloads = {}
    for family, (filename, label, label_basis) in specs.items():
        url = (
            "https://huggingface.co/datasets/"
            "tom-gibbs/multi-turn_jailbreak_attack_datasets/resolve/"
            f"{revision}/{filename.replace(' ', '%20')}"
        )
        data, downloads[filename] = _fetch(url)
        rows = _csv_rows(
            data,
            {
                "Goal ID",
                "Goal",
                "Prompt",
                "Multi-turn conversation",
                "Input-cipher",
                "Output-cipher",
            },
        )
        output[family] = []
        for row in rows:
            prompt_digest = text_hash(row["Prompt"])
            technique = "/".join(
                value for value in (row["Input-cipher"], row["Output-cipher"]) if value
            )
            sample = _sample(
                text=row["Prompt"],
                label=label,
                attack_type="obfuscated_jailbreak" if label else None,
                source="multi_turn",
                source_split=family,
                source_id=f"{family}:{row['Goal ID']}:{prompt_digest}",
                group_id=f"multi_turn:{family}:{row['Goal ID']}",
                category=technique,
                goal_policy_status="unsafe" if label else "safe",
                label_basis=label_basis,
            )
            sample.update(
                {
                    "source_goal_sha256": hashlib.sha256(
                        row["Goal"].encode()
                    ).hexdigest(),
                    "source_multi_turn_conversation_sha256": hashlib.sha256(
                        row["Multi-turn conversation"].encode()
                    ).hexdigest(),
                }
            )
            output[family].append(sample)

    complete_filename = "Complete Harmful Dataset.csv"
    complete_url = (
        "https://huggingface.co/datasets/"
        "tom-gibbs/multi-turn_jailbreak_attack_datasets/resolve/"
        f"{revision}/Complete%20Harmful%20Dataset.csv"
    )
    complete_data, downloads[complete_filename] = _fetch(complete_url)
    complete_rows = _csv_rows(
        complete_data,
        {
            "Example ID",
            "Goal ID",
            "Goal",
            "Prompt",
            "Multi-turn conversation",
            "Single-turn conversation",
            "Decoded responses",
            "Model",
            "Input-cipher",
            "Output-cipher",
            "Jailbroken",
            "UTQ",
        },
    )
    output["complete_harmful"] = []
    for row in complete_rows:
        sample = _sample(
            text=row["Prompt"],
            label=1,
            attack_type="obfuscated_jailbreak",
            source="multi_turn",
            source_split="complete_harmful",
            source_id=f"complete:{row['Example ID']}:{text_hash(row['Prompt'])}",
            group_id=f"multi_turn:harmful:{row['Goal ID']}",
            category="/".join(
                value for value in (row["Input-cipher"], row["Output-cipher"]) if value
            ),
            goal_policy_status="unsafe",
            label_basis="benchmark_attack_with_target_specific_outcome",
        )
        sample.update(
            {
                "source_jailbroken_outcome": row["Jailbroken"],
                "source_model": row["Model"],
                "source_response_sha256": hashlib.sha256(
                    row["Decoded responses"].encode()
                ).hexdigest(),
                "source_understood_question_outcome": row["UTQ"],
            }
        )
        output["complete_harmful"].append(_set_source_role(sample, "auxiliary"))
    profile = {
        "rows_by_file": {
            "Harmful Dataset.csv": len(output["harmful"]),
            "Semi-Benign Dataset.csv": len(output["semi_benign"]),
            "Completely-Benign Dataset.csv": len(output["benign"]),
            complete_filename: len(output["complete_harmful"]),
        },
        "projection": (
            "Prompt is detector text; goals, multi-turn histories, and target responses "
            "are retained by digest and remain reproducible from pinned source files"
        ),
    }
    return output, downloads, profile


def _load_oasst1() -> tuple[dict[str, list[dict]], dict]:
    info = SOURCES["oasst1"]
    dataset = load_dataset(info["repo"], revision=info["revision"])
    output: dict[str, list[dict]] = {"train": [], "validation": [], "source": []}
    profile = {
        "rows_by_split_and_role": {},
        "selected_weak_injection_control_rows": {},
    }
    for split in ("train", "validation"):
        selected = 0
        counts = Counter()
        for row in dataset[split]:
            if type(row["deleted"]) is not bool or (
                row["review_result"] is not None
                and type(row["review_result"]) is not bool
            ):
                raise ValueError(f"oasst1:{split} has invalid review fields")
            accepted_prompter = (
                row["role"] == "prompter"
                and not row["deleted"]
                and row["review_result"] is True
                and bool(row["text"].strip())
            )
            sample = _sample(
                text=row["text"],
                label=0 if accepted_prompter else None,
                attack_type=None,
                security_label="uncertain",
                source="oasst1",
                source_split=split,
                source_id=row["message_id"],
                group_id=f"oasst1:{row['message_tree_id']}",
                category=row["lang"] or "unknown_language",
                input_channel=(
                    "direct_user" if row["role"] == "prompter" else "model_output"
                ),
                label_basis=(
                    "weak_injection_nonattack:accepted_human_chat"
                    if accepted_prompter
                    else "official_message_without_benign_injection_supervision"
                ),
            )
            sample.update(
                {
                    "source_deleted": bool(row["deleted"]),
                    "source_detoxify": row.get("detoxify"),
                    "source_labels": row.get("labels"),
                    "source_language": row.get("lang"),
                    "source_message_role": row["role"],
                    "source_parent_id": row["parent_id"],
                    "source_review_count": row.get("review_count"),
                    "source_review_result": row["review_result"],
                }
            )
            role = (
                "auxiliary"
                if accepted_prompter
                else "uncertain"
                if row["role"] == "prompter"
                else "auxiliary"
            )
            _set_source_role(sample, role)
            output["source"].append(sample)
            counts[row["role"]] += 1
            if accepted_prompter:
                selected += 1
                output[split].append(sample)
        profile["rows_by_split_and_role"][split] = dict(sorted(counts.items()))
        profile["selected_weak_injection_control_rows"][split] = selected
    return output, profile


def _oasst_position_stress(rows: list[dict], limit: int = 500) -> list[dict]:
    output = []
    left = rows[::2]
    right = reversed(rows[1::2])
    for first, second in zip(left, right):
        if first["group_id"] == second["group_id"]:
            continue
        output.append(
            _set_source_role(
                _sample(
                    text=f"{first['text']}\n\n---\n\n{second['text']}",
                    label=0,
                    attack_type=None,
                    source="oasst1",
                    source_split="validation_position_stress",
                    source_id=f"{first['source_id']}:{second['source_id']}",
                    group_id=f"oasst1_position:{first['group_id']}:{second['group_id']}",
                    category=f"{first['category']}+{second['category']}",
                    label_basis="weak_nonattack:two_accepted_human_prompts",
                ),
                "dev_test",
            )
        )
        if len(output) == limit:
            break
    return output


def _github_raw(source: str, path: str) -> tuple[bytes, str]:
    revision = SOURCES[source]["revision"]
    return _fetch(
        f"https://raw.githubusercontent.com/{SOURCES[source]['url'].split('github.com/')[1]}/"
        f"{revision}/{path}"
    )


def _load_harmbench() -> tuple[list[dict], str]:
    data, digest = _github_raw(
        "harmbench", "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    rows = _csv_rows(
        data,
        {
            "Behavior",
            "BehaviorID",
            "ContextString",
            "FunctionalCategory",
            "SemanticCategory",
        },
    )
    output = []
    for row in rows:
        context = row["ContextString"].strip()
        text = f"{context}\n\n{row['Behavior']}" if context else row["Behavior"]
        output.append(
            _sample(
                text=text,
                label=0,
                attack_type=None,
                source="harmbench",
                source_split="test",
                source_id=row["BehaviorID"],
                group_id=f"harmbench:{row['BehaviorID']}",
                category=row["SemanticCategory"],
                goal_policy_status="unsafe",
                label_basis="task_definition:harmful_goal_not_injection",
            )
        )
    return output, digest


def _load_do_not_answer() -> tuple[dict[str, list[dict]], dict[str, str]]:
    data, english_digest = _github_raw(
        "do_not_answer", "datasets/Instruction/do_not_answer_en.csv"
    )
    rows = _csv_rows(data, {"id", "risk_area", "types_of_harm", "question"})
    output = {}
    output["english"] = [
        _sample(
            text=row["question"],
            label=0,
            attack_type=None,
            source="do_not_answer",
            source_split="test",
            source_id=row["id"],
            group_id=f"do_not_answer:{row['id']}",
            category=row["risk_area"],
            goal_policy_status="unsafe",
            label_basis="task_definition:harmful_goal_not_injection",
        )
        for row in rows
    ]

    data, chinese_digest = _github_raw("do_not_answer", "datasets/data_zh.csv")
    rows = _csv_rows(
        data,
        {
            "id",
            "risk_area",
            "types_of_harm",
            "question_type",
            "zh_question",
            "en_question",
        },
    )
    output["chinese"] = []
    output["translated_english"] = []
    for index, row in enumerate(rows):
        pair_id = f"{row['id']}:{index}"
        for language, field, name in (
            ("zh", "zh_question", "chinese"),
            ("en", "en_question", "translated_english"),
        ):
            sample = _sample(
                text=row[field],
                label=0,
                attack_type=None,
                source="do_not_answer",
                source_split=f"test_{language}",
                source_id=f"{pair_id}:{language}",
                group_id=f"do_not_answer:paired:{pair_id}",
                split_group_id=f"do_not_answer:paired:{pair_id}",
                category=row["risk_area"],
                goal_policy_status="unsafe",
                label_basis="task_definition:paired_harmful_goal_not_injection",
            )
            sample.update(
                {
                    "source_language": language,
                    "source_question_type": row["question_type"],
                }
            )
            output[name].append(sample)
    return output, {
        "datasets/Instruction/do_not_answer_en.csv": english_digest,
        "datasets/data_zh.csv": chinese_digest,
    }


def _jsonl_bytes(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _insert_attack(context: str, attack: str, position: str) -> str:
    if position == "start":
        return f"{attack}\n\n{context}"
    if position == "middle":
        midpoint = len(context) // 2
        return f"{context[:midpoint]}\n\n{attack}\n\n{context[midpoint:]}"
    return f"{context}\n\n{attack}"


def _bipia_contexts(contents: dict[str, bytes], split: str) -> dict[str, list[tuple]]:
    text_rows = {
        task: _jsonl_bytes(contents[f"benchmark/{task}/{split}.jsonl"])
        for task in ("email", "table")
    }
    text_contexts = []
    for index in range(max(len(rows) for rows in text_rows.values())):
        for task, rows in text_rows.items():
            if index < len(rows):
                text_contexts.append((task, f"{task}:{index}", rows[index]["context"]))
    return {
        "text": text_contexts,
        "code": [
            ("code", f"code:{index}", "\n".join(row["context"]))
            for index, row in enumerate(
                _jsonl_bytes(contents[f"benchmark/code/{split}.jsonl"])
            )
        ],
    }


def _bipia_split(contents: dict[str, bytes], split: str) -> dict[str, list[dict]]:
    contexts = _bipia_contexts(contents, split)
    payload_specs = []
    for kind in ("text", "code"):
        attacks = json.loads(contents[f"benchmark/{kind}_attack_{split}.json"])
        for category, variants in attacks.items():
            for index, attack in enumerate(variants):
                source_id = f"{split}:{kind}:{category}:{index}"
                payload_specs.append((kind, category, source_id, attack))

    payloads = []
    poisoned = []
    for index, (kind, category, source_id, attack) in enumerate(payload_specs):
        task, context_id, context = contexts[kind][index % len(contexts[kind])]
        group_id = f"bipia:{source_id}"
        split_group_id = f"bipia:{split}:context:{context_id}"
        payloads.append(
            _sample(
                text=attack,
                label=1,
                attack_type="indirect_prompt_injection",
                source="bipia",
                source_split=f"{split}_payload",
                source_id=source_id,
                group_id=group_id,
                split_group_id=split_group_id,
                category=f"{kind}/{category}",
                input_channel="untrusted_content",
                label_basis="benchmark_attack_construction",
            )
        )
        for position in ("start", "middle", "end"):
            poisoned.append(
                _sample(
                    text=_insert_attack(context, attack, position),
                    label=1,
                    attack_type="indirect_prompt_injection",
                    source="bipia",
                    source_split=f"{split}_poisoned_context",
                    source_id=f"{source_id}:{task}:{context_id}:{position}",
                    group_id=group_id,
                    split_group_id=split_group_id,
                    category=f"{kind}/{task}/{position}",
                    input_channel="untrusted_content",
                    label_basis="benchmark_attack_construction",
                )
            )

    clean = []
    for kind_contexts in contexts.values():
        for task, context_id, context in kind_contexts:
            clean.append(
                _sample(
                    text=context,
                    label=0,
                    attack_type=None,
                    source="bipia",
                    source_split=f"{split}_clean_context",
                    source_id=f"{split}:clean:{context_id}",
                    group_id=f"bipia:{split}:clean:{context_id}",
                    split_group_id=f"bipia:{split}:context:{context_id}",
                    category=task,
                    input_channel="untrusted_content",
                    label_basis="benchmark_clean_context",
                )
            )
    return {"payload": payloads, "context": poisoned, "clean": clean}


def _load_bipia() -> tuple[dict[str, dict[str, list[dict]]], dict[str, str]]:
    paths = tuple(
        path
        for split in ("train", "test")
        for path in (
            f"benchmark/text_attack_{split}.json",
            f"benchmark/code_attack_{split}.json",
            f"benchmark/email/{split}.jsonl",
            f"benchmark/table/{split}.jsonl",
            f"benchmark/code/{split}.jsonl",
        )
    )
    downloads = {}
    contents = {}
    for path in paths:
        contents[path], downloads[path] = _github_raw("bipia", path)

    return {
        split: _bipia_split(contents, split) for split in ("train", "test")
    }, downloads


def _load_notinject() -> tuple[list[dict], dict[str, str]]:
    output = []
    downloads = {}
    for word_count in ("one", "two", "three"):
        path = f"datasets/NotInject_{word_count}.json"
        data, downloads[path] = _github_raw("notinject", path)
        for index, row in enumerate(json.loads(data)):
            output.append(
                _sample(
                    text=row["prompt"],
                    label=0,
                    attack_type=None,
                    source="notinject",
                    source_split="test",
                    source_id=f"{word_count}:{index}",
                    group_id=f"notinject:{word_count}:{index}",
                    category=row["category"],
                    goal_policy_status="safe",
                    label_basis="benchmark_nonattack:injection_trigger_words",
                )
            )
    return output, downloads


def _load_jailbreaks_over_time() -> tuple[list[dict], str]:
    data, digest = _github_raw("jailbreaks_over_time", "data/jailbreaksovertime.json")
    rows = json.loads(data)
    return [
        _sample(
            text=row["prompt"],
            label=row["label"],
            attack_type="direct_jailbreak" if row["label"] else None,
            source="jailbreaks_over_time",
            source_split="temporal_test",
            source_id=row["uid"],
            group_id=f"jailbreaks_over_time:{row['uid']}",
            category=row["source"],
            goal_policy_status="unknown",
            label_basis="source_temporal_annotation",
        )
        for row in rows
        if isinstance(row.get("prompt"), str) and row["prompt"].strip()
    ], digest


def _load_tensor_trust() -> tuple[dict[str, list[dict]], dict[str, str]]:
    attacks = []
    contexts = []
    downloads = {}
    for benchmark in ("extraction", "hijacking"):
        path = (
            f"benchmarks/{benchmark}-robustness/v1/{benchmark}_robustness_dataset.jsonl"
        )
        data, downloads[path] = _github_raw("tensor_trust", path)
        for row in _jsonl_bytes(data):
            lineage_id = f"{benchmark}:{row['sample_id']}"
            attack_type = (
                "prompt_extraction" if benchmark == "extraction" else "prompt_hijacking"
            )
            common = {
                "label": 1,
                "attack_type": attack_type,
                "source": "tensor_trust",
                "group_id": f"tensor_trust:{lineage_id}",
                "category": benchmark,
                "label_basis": "benchmark_human_attack",
            }
            attacks.append(
                _sample(
                    text=row["attack"],
                    source_split=f"{benchmark}_attack_test",
                    source_id=f"{lineage_id}:attack",
                    **common,
                )
            )
            contexts.append(
                _sample(
                    text=f"{row['pre_prompt']}\n{row['attack']}\n{row['post_prompt']}",
                    source_split=f"{benchmark}_context_test",
                    source_id=f"{lineage_id}:context",
                    input_channel="untrusted_content",
                    security_label="indirect_prompt_injection",
                    **common,
                )
            )
    return {"attack": attacks, "context": contexts}, downloads


def _parse_nemotron_agentic_ipi(data: bytes) -> tuple[list[dict], dict]:
    expected_fields = {
        "agent_ref",
        "attack_category",
        "domain",
        "environment",
        "id",
        "injection",
        "injection_vector",
        "license",
        "required_tools",
        "responses_create_params",
        "target_tool",
        "used_in",
        "verifier_config",
    }
    expected_injection_fields = {
        "category",
        "goal",
        "injection_text",
        "target_args",
        "target_tool",
        "vector",
        "verification_type",
    }
    rows = _jsonl_bytes(data)
    output = []
    seen_ids = set()
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("unexpected Nemotron Agentic IPI schema")
        injection = row["injection"]
        if (
            not isinstance(injection, dict)
            or set(injection) != expected_injection_fields
        ):
            raise ValueError("unexpected Nemotron Agentic IPI injection schema")
        if (
            row["license"] != "CC BY 4.0"
            or row["agent_ref"]
            != {
                "type": "responses_api_agents",
                "name": "indirect_prompt_injection_simple_agent",
            }
            or row["verifier_config"]
            != {"type": "trace_analysis", "mode": "agentic_ipi"}
            or row["used_in"] != ["ultra_v3"]
        ):
            raise ValueError("unexpected Nemotron Agentic IPI provenance")
        if (
            not isinstance(row["id"], int)
            or row["id"] in seen_ids
            or not isinstance(row["environment"], dict)
            or not isinstance(row["required_tools"], list)
            or len(row["required_tools"]) != 1
            or not isinstance(row["required_tools"][0], str)
            or row["target_tool"] in row["required_tools"]
        ):
            raise ValueError("invalid Nemotron Agentic IPI task structure")
        request = row["responses_create_params"]
        messages = request.get("input") if isinstance(request, dict) else None
        if (
            not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
            or [message.get("role") for message in messages] != ["system", "user"]
        ):
            raise ValueError("unexpected Nemotron Agentic IPI request structure")
        if (
            injection["target_tool"] != row["target_tool"]
            or injection["category"] != row["attack_category"]
            or injection["vector"] != row["injection_vector"]
        ):
            raise ValueError("inconsistent Nemotron Agentic IPI attack metadata")
        if not all(
            isinstance(row[field], str) and row[field]
            for field in (
                "domain",
                "attack_category",
                "injection_vector",
                "target_tool",
            )
        ) or not isinstance(injection["injection_text"], str):
            raise ValueError("invalid Nemotron Agentic IPI grouping metadata")
        seen_ids.add(row["id"])
        text = injection["injection_text"]
        sample = _sample(
            text=text,
            label=1,
            attack_type="indirect_prompt_injection",
            source="nemotron_agentic_ipi",
            source_split="synthetic_success_test",
            source_id=str(row["id"]),
            group_id=f"nemotron_agentic_ipi:text:{text_hash(text)}",
            category=row["attack_category"],
            goal_policy_status="unsafe",
            input_channel="untrusted_content",
            label_basis="fully_synthetic_attack_success_against_source_defender",
        )
        sample.update(
            {
                "source_domain": row["domain"],
                "source_attack_category": row["attack_category"],
                "source_injection_vector": row["injection_vector"],
                "source_target_tool": row["target_tool"],
            }
        )
        output.append(sample)

    profile = {
        "raw_rows": len(rows),
        "normalized_unique_injection_texts": len(
            {normalize_text(row["injection"]["injection_text"]) for row in rows}
        ),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "attack_categories": dict(
            sorted(Counter(row["attack_category"] for row in rows).items())
        ),
        "injection_vectors": dict(
            sorted(Counter(row["injection_vector"] for row in rows).items())
        ),
        "target_tools": dict(
            sorted(Counter(row["target_tool"] for row in rows).items())
        ),
        "projection": "injection.injection_text plus non-content grouping metadata",
        "excluded_content": [
            "environment",
            "injection.goal",
            "injection.target_args",
            "responses_create_params.input",
            "responses_create_params.tools",
        ],
    }
    return output, profile


def _load_nemotron_agentic_ipi() -> tuple[list[dict], str, dict]:
    info = SOURCES["nemotron_agentic_ipi"]
    data, digest = _fetch(
        f"https://huggingface.co/datasets/{info['repo']}/resolve/"
        f"{info['revision']}/{info['path']}",
        expected_bytes=info["bytes"],
        expected_sha256=info["sha256"],
    )
    rows, profile = _parse_nemotron_agentic_ipi(data)
    return rows, digest, profile


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
                if len(
                    {
                        json.dumps(
                            row.get(field), sort_keys=True, separators=(",", ":")
                        )
                        for row in matches
                    }
                )
                > 1
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


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


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
        row_ids = set()
        for row in rows:
            if row["id"] in row_ids:
                raise ValueError(f"duplicate canonical row id in {name}: {row['id']}")
            row_ids.add(row["id"])
            _set_core_routing_role(name, row)
        path = source_dir / f"{name}.jsonl"
        role_counts = Counter(row["source_role"] for row in rows)
        source_outputs[name] = {
            "path": str(path.relative_to(data_dir)),
            "rows": len(rows),
            "roles": dict(sorted(role_counts.items())),
            "security_labels": dict(
                sorted(Counter(row["security_label"] for row in rows).items())
            ),
            "routing_benign": sum(not row["routing_label"] for row in rows),
            "routing_non_benign": sum(row["routing_label"] for row in rows),
            "sha256": _write_jsonl(path, rows),
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
