"""First-generation core source loaders consumed by `morgott.data.build_dataset`.

These loaders keep their historical structured return shapes because
`build_dataset` derives the legacy injection views from them; the newer
`LOADERS` adapters in this package feed `morgott.corpus` instead.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from datasets import load_dataset

from ..data import (
    SOURCES,
    _csv_rows,
    _fetch,
    _github_raw,
    _sample,
    _set_source_role,
    normalize_text,
    text_hash,
)

# Pinned digests, keyed "source/path", for raw fetches that historically
# carried no pre-fetch pin. Values mirror the download_sha256 entries in
# data/manifest.json (they stay out of SOURCES, which is serialized into the
# manifest verbatim). _fetch verifies them on every read, download or cache.
_PINNED_SHA256 = {
    "xstest/xstest_prompts.csv": (
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
    ),
    "harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv": (
        "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
    ),
    "do_not_answer/datasets/Instruction/do_not_answer_en.csv": (
        "06acfa39a06a1b33d1f264ce41b4f7a95812010c594fb733ae4717ee0a4544fc"
    ),
    "do_not_answer/datasets/data_zh.csv": (
        "cec83aa3c0f141f1c65b88ab99b953e6acd7d1badec630a0a76f70044a07fec3"
    ),
    "bipia/benchmark/text_attack_train.json": (
        "63f95d3e67eac4178cdabdbdaf192cd05f2b6ed0702b578f1d30d556e5155670"
    ),
    "bipia/benchmark/text_attack_test.json": (
        "75750e7b4e8b34e8f9d88d89b357aeaaf02bd07f9e493ccd37eda74a0cd7c7f8"
    ),
    "bipia/benchmark/code_attack_train.json": (
        "fe515080b6da2b5c0b67ca7ba2a3b3c2ce57c48f45733c475ebc813cba118484"
    ),
    "bipia/benchmark/code_attack_test.json": (
        "892545c5aaec0645b1ded65dc7816b3d70e9ef4eadcba2301a7a3db93676b6e0"
    ),
    "bipia/benchmark/email/train.jsonl": (
        "82207193cb8ce06713eeb7c33ca0716446613512e2ad9303b302ba14d425ddd4"
    ),
    "bipia/benchmark/email/test.jsonl": (
        "217b403faaa1d0cb12c24892bea39f4a0b9e2819919ac7e2b902006e28278cdb"
    ),
    "bipia/benchmark/table/train.jsonl": (
        "ae1c3ff25733fb3ff1a35bda3009716f4d25a17ae6833c73521be0411d576494"
    ),
    "bipia/benchmark/table/test.jsonl": (
        "3d5eaab192b80bada762e3cd7655583a14e5888ba8b845a01f0434e4ccc4291e"
    ),
    "bipia/benchmark/code/train.jsonl": (
        "5e6879b621a5cefe265a5b41b7ed0be632536f658f6097baf85a28b9abcb3115"
    ),
    "bipia/benchmark/code/test.jsonl": (
        "ed40b84f541352fb45753032a80750922de8d5e0b70997924df3d04cf942e058"
    ),
    "multi_turn/Harmful Dataset.csv": (
        "9b84edb9ebbf695b4884a897284022c4f75e382a577359bd031c6ec902a45f17"
    ),
    "multi_turn/Semi-Benign Dataset.csv": (
        "9acd4d02efaf7cdf79ac52841541e07bf405e7d9257884af7b55793fe0c839da"
    ),
    "multi_turn/Completely-Benign Dataset.csv": (
        "4efb90f0b4c8d54ad66353acd89bf4008413c80b9bf45ddc812b1f36847c7398"
    ),
    "multi_turn/Complete Harmful Dataset.csv": (
        "c0c037832b43c3ccc7386a70e50a1c51b4e5fb30128fd9beb252369492807304"
    ),
    "notinject/datasets/NotInject_one.json": (
        "69b535596d95102424e9c5946944feb4f2d596687eb8213f2ecad75478e5ffdd"
    ),
    "notinject/datasets/NotInject_two.json": (
        "6043d94e75b48d8e7682d25dc79eaf45359e1e561ce520e3b8fd5625a91060c6"
    ),
    "notinject/datasets/NotInject_three.json": (
        "ef01eff0d761d2e34571b3fdbcec08c30cd93efe8d0e1a2eb5c2baeb1873b070"
    ),
    "jailbreaks_over_time/data/jailbreaksovertime.json": (
        "110fb269a3af1bd9891d583b576563ddde658a980efee5718be50dd8fc5ad001"
    ),
    "tensor_trust/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl": (
        "4090f9f57b6df69cc32d7669ac991e3fd53e6e2bb616a5e6f45dac79c183ee4c"
    ),
    "tensor_trust/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl": (
        "2ad3c2343bbf43cb1924f3ec5d6f830864ce3c879e9832b01d93a991e7e2e680"
    ),
}


def _pinned_github_raw(source: str, path: str) -> tuple[bytes, str]:
    return _github_raw(source, path, expected_sha256=_PINNED_SHA256[f"{source}/{path}"])


def _load_toxic_chat() -> dict[str, list[dict]]:
    info = SOURCES["toxic_chat"]
    dataset = load_dataset(info["repo"], "toxicchat0124", revision=info["revision"])
    output: dict[str, list[dict]] = {}
    for split in ("train", "test"):
        output[split] = []
        for row in dataset[split].to_list():
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
        for row in dataset[split].to_list():
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
    data, digest = _pinned_github_raw("xstest", "xstest_prompts.csv")
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
        data, downloads[filename] = _fetch(
            url, expected_sha256=_PINNED_SHA256[f"multi_turn/{filename}"]
        )
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
    complete_data, downloads[complete_filename] = _fetch(
        complete_url,
        expected_sha256=_PINNED_SHA256[f"multi_turn/{complete_filename}"],
    )
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
        # ponytail: to_list() materializes the split (~0.5GB transient RSS
        # for train), revert to row iteration if a repinned dump outgrows
        # the box.
        for row in dataset[split].to_list():
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


def _load_harmbench() -> tuple[list[dict], str]:
    data, digest = _pinned_github_raw(
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
    data, english_digest = _pinned_github_raw(
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

    data, chinese_digest = _pinned_github_raw("do_not_answer", "datasets/data_zh.csv")
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
        contents[path], downloads[path] = _pinned_github_raw("bipia", path)

    return {
        split: _bipia_split(contents, split) for split in ("train", "test")
    }, downloads


def _load_notinject() -> tuple[list[dict], dict[str, str]]:
    output = []
    downloads = {}
    for word_count in ("one", "two", "three"):
        path = f"datasets/NotInject_{word_count}.json"
        data, downloads[path] = _pinned_github_raw("notinject", path)
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
    data, digest = _pinned_github_raw(
        "jailbreaks_over_time", "data/jailbreaksovertime.json"
    )
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
        data, downloads[path] = _pinned_github_raw("tensor_trust", path)
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
