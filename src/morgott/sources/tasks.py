from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from collections.abc import Iterator

import ijson
import pyarrow.parquet as pq

from ..data import (
    SOURCES,
    _csv_rows,
    _github_raw,
    _sample,
    _set_source_role,
    text_hash,
)
from ._shared import (
    _SENSITIVE_TEXT_PATTERNS as _SENSITIVE_TEXT_PATTERNS,
)
from ._shared import (
    FILES,
    _download,
    _download_files,
    _parquet_dataset,
    _sensitive_quarantine,
    _verified_archive,
)
from ._shared import (
    _sensitive_text_reasons as _sensitive_text_reasons,
)

_PROVIDER_EGRESS_LICENSES = frozenset(
    {
        "Apache-2.0",
        "CC-BY-4.0",
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-4.0 prompts; CC-BY-NC-4.0 model outputs",
        "MIT",
        "ODC-BY",
    }
)


def _public_declared_license(value: object) -> bool:
    return isinstance(value, str) and value.strip() in _PROVIDER_EGRESS_LICENSES


def _mind2web_sample(source_row: dict) -> dict:
    annotation_id = source_row.get("annotation_id")
    text = source_row.get("confirmed_task")
    website = source_row.get("website")
    domain = source_row.get("domain")
    subdomain = source_row.get("subdomain")
    if (
        not isinstance(annotation_id, str)
        or not annotation_id
        or not isinstance(text, str)
        or not text.strip()
        or not isinstance(website, str)
        or not website
        or not isinstance(domain, str)
        or not domain
        or not isinstance(subdomain, str)
        or not subdomain
    ):
        raise ValueError("mind2web training row has invalid task lineage")
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="mind2web",
        source_split="train",
        source_id=annotation_id,
        group_id=f"mind2web:{annotation_id}",
        split_group_id=f"mind2web:{annotation_id}",
        category=f"{domain}/{subdomain}",
        input_channel="direct_user",
        label_basis="bounded_web_agent_task_collection_not_safety_annotation",
    )
    row.update(
        {
            "source_annotation_id": annotation_id,
            "source_domain": domain,
            "source_language": "en",
            "source_subdomain": subdomain,
            "source_website": website,
        }
    )
    return row


def _mind2web_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    conversion_revision = SOURCES["mind2web"]["conversion_revision"]
    paths, downloads = _download_files("mind2web", revision=conversion_revision)
    downloads = {
        f"conversion/{conversion_revision}/{filename}": digest
        for filename, digest in downloads.items()
    }
    columns = ("website", "domain", "subdomain", "annotation_id", "confirmed_task")
    accepted = []
    quarantined = []
    reason_counts = Counter()
    annotation_ids = set()
    for path in paths.values():
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=128):
            for source_row in batch.to_pylist():
                row = _mind2web_sample(source_row)
                annotation_id = row["source_annotation_id"]
                if annotation_id in annotation_ids:
                    raise ValueError(
                        f"mind2web has duplicate annotation id: {annotation_id}"
                    )
                annotation_ids.add(annotation_id)
                quarantine_row = _sensitive_quarantine(row)
                if quarantine_row is not None:
                    quarantined.append(quarantine_row)
                    reason_counts.update(
                        quarantine_row["source_sensitive_text_reasons"]
                    )
                else:
                    accepted.append(_set_source_role(row, "candidate"))
    if len(annotation_ids) != 1_009:
        raise ValueError(
            f"mind2web expected 1009 official training tasks, found {len(annotation_ids)}"
        )
    profile = {
        "projection": (
            "confirmed_task plus annotation, website, domain, and subdomain lineage "
            "from all 1009 official training tasks"
        ),
        "excluded": (
            "protected test data, raw and cleaned HTML, action representations, actions, "
            "DOM snapshots, HAR or network data, storage and session files, and traces"
        ),
        "conversion": (
            "complete Hugging Face Parquet conversion of the pinned official training "
            "revision; only the five permitted columns are read"
        ),
        "privacy_check": (
            "high-precision local regex screening for credentials, contact details, "
            "payment data, identifiers, and explicit street addresses; suspicious raw "
            "tasks are quarantined without redaction"
        ),
        "privacy_limit": (
            "pattern checks reduce obvious exposure but do not prove that retained task "
            "text contains no personal data"
        ),
        "official_training_tasks": len(annotation_ids),
        "retained_tasks": len(accepted),
        "quarantined_tasks": len(quarantined),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
    }
    return iter(accepted), downloads, profile, iter(quarantined)


def _swebench_verified_sample(source_row: dict) -> dict:
    fields = (
        "repo",
        "instance_id",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
        "environment_setup_commit",
        "difficulty",
    )
    if any(
        not isinstance(source_row.get(field), str) or not source_row[field].strip()
        for field in fields
    ):
        raise ValueError("swebench_verified row has invalid task lineage")
    repo = source_row["repo"]
    instance_id = source_row["instance_id"]
    row = _sample(
        text=source_row["problem_statement"],
        label=0,
        attack_type=None,
        source="swebench_verified",
        source_split="test",
        source_id=instance_id,
        group_id=f"swebench_verified:{instance_id}",
        split_group_id=f"swebench_verified:repo:{repo}",
        category=repo,
        input_channel="direct_user",
        label_basis="human_verified_solvable_software_issue_not_safety_annotation",
    )
    row.update(
        {
            "source_base_commit": source_row["base_commit"],
            "source_created_at": source_row["created_at"],
            "source_difficulty": source_row["difficulty"],
            "source_environment_setup_commit": source_row["environment_setup_commit"],
            "source_instance_id": instance_id,
            "source_language": "en",
            "source_repository": repo,
            "source_version": source_row["version"],
        }
    )
    return _set_source_role(row, "dev_test")


def _swebench_verified_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    filename, expected = FILES["swebench_verified"]["dev_test"]
    path, digest = _download("swebench_verified", filename, expected)
    columns = (
        "repo",
        "instance_id",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
        "environment_setup_commit",
        "difficulty",
    )
    accepted = []
    quarantined = []
    reason_counts = Counter()
    instance_ids = set()
    repositories = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=128):
        for source_row in batch.to_pylist():
            row = _swebench_verified_sample(source_row)
            instance_id = row["source_instance_id"]
            if instance_id in instance_ids:
                raise ValueError(
                    f"swebench_verified has duplicate instance id: {instance_id}"
                )
            instance_ids.add(instance_id)
            repositories[row["source_repository"]] += 1
            quarantine_row = _sensitive_quarantine(row)
            if quarantine_row is not None:
                quarantined.append(quarantine_row)
                reason_counts.update(quarantine_row["source_sensitive_text_reasons"])
            else:
                accepted.append(row)
    if len(instance_ids) != 500:
        raise ValueError(
            f"swebench_verified expected 500 official test tasks, found {len(instance_ids)}"
        )
    lengths = [len(row["text"]) for row in accepted]
    profile = {
        "projection": "problem_statement plus repository, instance, commit, time, version, and difficulty lineage",
        "excluded": "patch, test patch, test names, hints, and all repository contents",
        "label_limit": "human validation establishes a legitimate solvable software task, not a safety annotation",
        "evaluation_limit": "benign FPR slice only; no same-format attack arm or aggregate performance claim",
        "privacy_check": "the same local secret and PII screen used for Mind2Web; suspicious issue text is quarantined without redaction",
        "privacy_limit": "pattern checks reduce obvious exposure but do not prove that retained public issue text contains no personal data",
        "official_test_tasks": len(instance_ids),
        "retained_tasks": len(accepted),
        "quarantined_tasks": len(quarantined),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "official_tasks_by_repository": dict(sorted(repositories.items())),
        "retained_length_characters": {
            "at_least_1024": sum(length >= 1_024 for length in lengths),
            "at_least_2048": sum(length >= 2_048 for length in lengths),
            "at_least_4096": sum(length >= 4_096 for length in lengths),
            "maximum": max(lengths),
        },
    }
    return iter(accepted), {filename: digest}, profile, iter(quarantined)


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


def _taskmaster_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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

    return rows(), {"taskmaster.tar.gz": digest}, profile, None


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


def _banking77_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    expected = {
        "banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        "banking_data/test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    }
    contents = {}
    downloads = {}
    for filename, expected_digest in expected.items():
        data, digest = _github_raw(
            "banking77", filename, expected_sha256=expected_digest
        )
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
    return rows(), downloads, profile, None


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


def _false_reject_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    paths, downloads = _download_files("false_reject")
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

    return rows(), downloads, profile, None


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


def _schema_guided_dialogue_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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

    return rows(), {"schema_guided_dialogue.tar.gz": digest}, profile, None


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


def _massive_en_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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

    return rows(), {"amazon-massive-dataset-1.1.tar.gz": digest}, profile, None


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


def _coconot_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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
    return rows(), downloads, profile, None


def _jbb_benign_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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
    return rows(), {filename: digest}, profile, None


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


def _lmsys_arena_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
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

    return rows(), downloads, profile, None
