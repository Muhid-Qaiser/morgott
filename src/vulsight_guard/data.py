from __future__ import annotations

import csv
import hashlib
import io
import json
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
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
        "use": "multilingual human-chat hard negatives; train/test separated by official split",
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
}

MAX_DOWNLOAD_BYTES = 60_000_000


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(
    *,
    text: str,
    label: int,
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
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source}:{source_id} has empty text")
    info = SOURCES[source]
    return {
        "schema_version": 1,
        "id": f"{source}:{source_id}",
        "text": text,
        "label": int(label),
        "attack_type": attack_type,
        "goal_policy_status": goal_policy_status,
        "category": category,
        "input_channel": input_channel,
        "label_basis": label_basis,
        "source": source,
        "source_split": source_split,
        "source_id": str(source_id),
        "group_id": str(group_id),
        "split_group_id": str(split_group_id or group_id),
        "source_revision": info["revision"],
        "license": info["license"],
    }


def _fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "vulsight-agent-guard/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"download exceeded {MAX_DOWNLOAD_BYTES} bytes: {url}")
    return data, hashlib.sha256(data).hexdigest()


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
        output[split] = [
            _sample(
                text=row["user_input"],
                label=row["jailbreaking"],
                attack_type="jailbreak" if row["jailbreaking"] else None,
                source="toxic_chat",
                source_split=split,
                source_id=row["conv_id"],
                group_id=f"toxic_chat:{row['conv_id']}",
                category="toxic" if row["toxicity"] else "not_toxic",
                goal_policy_status="unsafe" if row["toxicity"] else "unknown",
            )
            for row in dataset[split]
        ]
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


def _load_multi_turn() -> tuple[list[dict], str]:
    revision = SOURCES["multi_turn"]["revision"]
    url = (
        "https://huggingface.co/datasets/"
        "tom-gibbs/multi-turn_jailbreak_attack_datasets/resolve/"
        f"{revision}/Harmful%20Dataset.csv"
    )
    data, digest = _fetch(url)
    rows = _csv_rows(
        data,
        {"Goal ID", "Prompt", "Input-cipher", "Output-cipher"},
    )
    output = []
    for row in rows:
        prompt_digest = text_hash(row["Prompt"])
        technique = "/".join(
            value for value in (row["Input-cipher"], row["Output-cipher"]) if value
        )
        output.append(
            _sample(
                text=row["Prompt"],
                label=1,
                attack_type="obfuscated_jailbreak",
                source="multi_turn",
                source_split="test",
                source_id=f"{row['Goal ID']}:{prompt_digest}",
                group_id=f"multi_turn:{row['Goal ID']}",
                category=technique,
                goal_policy_status="unsafe",
                label_basis="benchmark_attack",
            )
        )
    return output, digest


def _load_oasst1() -> dict[str, list[dict]]:
    info = SOURCES["oasst1"]
    dataset = load_dataset(info["repo"], revision=info["revision"])
    output: dict[str, list[dict]] = {}
    for split in ("train", "validation"):
        output[split] = [
            _sample(
                text=row["text"],
                label=0,
                attack_type=None,
                source="oasst1",
                source_split=split,
                source_id=row["message_id"],
                group_id=f"oasst1:{row['message_tree_id']}",
                category=row["lang"] or "unknown_language",
                label_basis="weak_nonattack:accepted_human_chat",
            )
            for row in dataset[split]
            if row["role"] == "prompter"
            and not row["deleted"]
            and row["review_result"] is True
            and row["text"].strip()
        ]
    return output


def _oasst_position_stress(rows: list[dict], limit: int = 500) -> list[dict]:
    output = []
    left = rows[::2]
    right = reversed(rows[1::2])
    for first, second in zip(left, right):
        if first["group_id"] == second["group_id"]:
            continue
        output.append(
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


def _load_do_not_answer() -> tuple[list[dict], str]:
    data, digest = _github_raw(
        "do_not_answer", "datasets/Instruction/do_not_answer_en.csv"
    )
    rows = _csv_rows(data, {"id", "risk_area", "types_of_harm", "question"})
    return [
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
    ], digest


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
            attack_type="jailbreak" if row["label"] else None,
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
            source_id = f"{benchmark}:{row['sample_id']}"
            attack_type = (
                "prompt_extraction" if benchmark == "extraction" else "prompt_hijacking"
            )
            common = {
                "label": 1,
                "attack_type": attack_type,
                "source": "tensor_trust",
                "source_id": source_id,
                "group_id": f"tensor_trust:{source_id}",
                "category": benchmark,
                "label_basis": "benchmark_human_attack",
            }
            attacks.append(
                _sample(
                    text=row["attack"],
                    source_split=f"{benchmark}_attack_test",
                    **common,
                )
            )
            contexts.append(
                _sample(
                    text=f"{row['pre_prompt']}\n{row['attack']}\n{row['post_prompt']}",
                    source_split=f"{benchmark}_context_test",
                    input_channel="untrusted_content",
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
                "domain": row["domain"],
                "attack_category": row["attack_category"],
                "injection_vector": row["injection_vector"],
                "target_tool": row["target_tool"],
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
        f"{info['revision']}/{info['path']}"
    )
    if len(data) != info["bytes"] or digest != info["sha256"]:
        raise ValueError("Nemotron Agentic IPI artifact does not match pinned metadata")
    rows, profile = _parse_nemotron_agentic_ipi(data)
    return rows, digest, profile


def deduplicate(
    rows: Iterable[dict], blocked: set[str] | None = None
) -> tuple[list[dict], dict[str, int]]:
    blocked = blocked or set()
    groups: dict[str, list[dict]] = {}
    stats = {"blocked_by_train": 0, "duplicates": 0, "label_conflicts": 0}
    for row in rows:
        key = normalize_text(row["text"])
        if key in blocked:
            stats["blocked_by_train"] += 1
            continue
        groups.setdefault(key, []).append(row)

    kept = []
    for matches in groups.values():
        if len({row["label"] for row in matches}) > 1:
            stats["label_conflicts"] += len(matches)
            continue
        kept.append(matches[0])
        stats["duplicates"] += len(matches) - 1
    return kept, stats


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(data_dir: Path = Path("data")) -> dict:
    disable_progress_bars()
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    toxic = _load_toxic_chat()
    injections = _load_prompt_injections()
    oasst1 = _load_oasst1()
    xstest, xstest_sha = _load_xstest()
    harmbench, harmbench_sha = _load_harmbench()
    do_not_answer, do_not_answer_sha = _load_do_not_answer()
    multi_turn, multi_turn_sha = _load_multi_turn()
    bipia, bipia_sha = _load_bipia()
    notinject, notinject_sha = _load_notinject()
    jailbreaks_over_time, jailbreaks_over_time_sha = _load_jailbreaks_over_time()
    tensor_trust, tensor_trust_sha = _load_tensor_trust()
    nemotron_agentic_ipi, nemotron_agentic_ipi_sha, nemotron_agentic_ipi_profile = (
        _load_nemotron_agentic_ipi()
    )

    train_raw = toxic["train"] + injections["train"] + oasst1["train"]
    train, train_dedup = deduplicate(train_raw)
    train_keys = {normalize_text(row["text"]) for row in train}
    indirect_train_raw = (
        bipia["train"]["payload"] + bipia["train"]["context"] + bipia["train"]["clean"]
    )
    indirect_train, indirect_train_dedup = deduplicate(indirect_train_raw)
    indirect_train_keys = {normalize_text(row["text"]) for row in indirect_train}

    eval_sets = {
        "toxic_chat": toxic["test"],
        "prompt_injections": injections["test"],
        "xstest": xstest,
        "oasst1_chat": oasst1["validation"],
        "oasst1_position_stress": _oasst_position_stress(oasst1["validation"]),
        "harmbench": harmbench,
        "do_not_answer": do_not_answer,
        "multi_turn": multi_turn,
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
    output_rows = {"train": train, "indirect_train": indirect_train}
    for name, rows in eval_sets.items():
        blocked = train_keys
        if rows and any(row["input_channel"] == "untrusted_content" for row in rows):
            blocked = train_keys | indirect_train_keys
        output_rows[name], dedup[name] = deduplicate(rows, blocked)

    outputs = {}
    for name, rows in output_rows.items():
        path = processed / f"{name}.jsonl"
        outputs[name] = {
            "path": str(path),
            "rows": len(rows),
            "positive": sum(row["label"] for row in rows),
            "negative": sum(not row["label"] for row in rows),
            "sha256": _write_jsonl(path, rows),
        }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": "channel-scoped jailbreak or prompt-injection attack attempt",
        "sources": SOURCES,
        "download_sha256": {
            "xstest_prompts.csv": xstest_sha,
            "harmbench_behaviors_text_all.csv": harmbench_sha,
            "do_not_answer_en.csv": do_not_answer_sha,
            "multi_turn_harmful.csv": multi_turn_sha,
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
            "nemotron_agentic_ipi": nemotron_agentic_ipi_profile,
        },
        "deduplication": dedup,
        "outputs": outputs,
        "license_note": (
            "ToxicChat is CC-BY-NC-4.0 and Do-Not-Answer is CC-BY-NC-SA-4.0. "
            "Treat this consolidated corpus and derived model as research-only until "
            "commercial-use data is substituted."
        ),
    }
    manifest_path = processed / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
