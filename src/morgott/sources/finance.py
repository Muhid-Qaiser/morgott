from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import ijson

from ..data import SOURCES, _sample, _set_source_role
from ._shared import _github_pinned


def _harper_checkout() -> tuple[tempfile.TemporaryDirectory, Path, str]:
    temporary = tempfile.TemporaryDirectory(prefix="morgott-harper-")
    root = Path(temporary.name) / "repository"
    info = SOURCES["harper_valley_bank"]
    commands = (
        ("git", "init", "--quiet", str(root)),
        ("git", "-C", str(root), "remote", "add", "origin", info["url"]),
        ("git", "-C", str(root), "sparse-checkout", "init", "--cone"),
        (
            "git",
            "-C",
            str(root),
            "sparse-checkout",
            "set",
            "data/metadata",
            "data/transcript",
        ),
        (
            "git",
            "-C",
            str(root),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            info["revision"],
        ),
        (
            "git",
            "-C",
            str(root),
            "checkout",
            "--quiet",
            "--detach",
            "FETCH_HEAD",
        ),
    )
    try:
        for command in commands:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        revision = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if revision != info["revision"]:
            raise ValueError(
                "harper_valley_bank checkout did not resolve the pinned commit"
            )
        paths = sorted(
            [
                *root.glob("data/metadata/*.json"),
                *root.glob("data/transcript/*.json"),
            ],
            key=lambda path: path.relative_to(root).as_posix(),
        )
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        actual = digest.hexdigest()
        expected = "70782a602ebaca43930dbeef8b160675cfb8bbca0df21b5b12e46236c33cd9ec"
        if actual != expected:
            raise ValueError(
                "harper_valley_bank projected tree has an unexpected digest"
            )
        return temporary, root, actual
    except BaseException:
        temporary.cleanup()
        raise


def _harper_has_lexical_content(text: str) -> bool:
    without_markers = re.sub(r"\[[^\]]+\]|<unk>", " ", text, flags=re.IGNORECASE)
    return any(character.isalnum() for character in without_markers)


def _harper_valley_bank_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    temporary, root, digest = _harper_checkout()
    metadata_paths = {path.stem: path for path in root.glob("data/metadata/*.json")}
    transcript_paths = {path.stem: path for path in root.glob("data/transcript/*.json")}
    if len(metadata_paths) != 1_446 or metadata_paths.keys() != transcript_paths.keys():
        temporary.cleanup()
        raise ValueError("harper_valley_bank has an unexpected conversation file set")
    profile = {
        "projection": (
            "all meaningful human-corrected caller and agent transcript segments; "
            "caller is direct_user and agent is model_output"
        ),
        "collection_limit": (
            "human-human but simulated banking calls with eight bounded intents and "
            "deliberately limited vocabulary"
        ),
        "excluded": (
            "audio, machine transcripts, timestamps, model dialog acts and emotions, "
            "names in task metadata, survey data, and empty or marker-only segments"
        ),
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        sessions = Counter()
        intents = Counter()
        try:
            for sid in sorted(transcript_paths):
                metadata = json.loads(metadata_paths[sid].read_text(encoding="utf-8"))
                transcript = json.loads(
                    transcript_paths[sid].read_text(encoding="utf-8")
                )
                if metadata.get("sid") != sid or not isinstance(transcript, list):
                    raise ValueError(f"harper_valley_bank:{sid} has invalid lineage")
                session = metadata.get("session")
                tasks = metadata.get("tasks")
                if (
                    not isinstance(session, str)
                    or not session
                    or not isinstance(tasks, list)
                ):
                    raise ValueError(f"harper_valley_bank:{sid} has invalid metadata")
                task_types = list(
                    dict.fromkeys(
                        task.get("task_type")
                        for task in tasks
                        if isinstance(task, dict)
                        and isinstance(task.get("task_type"), str)
                        and task["task_type"]
                    )
                )
                if not task_types:
                    raise ValueError(f"harper_valley_bank:{sid} has no task intent")
                sessions[session] += 1
                intents.update(task_types)
                for segment_index, segment in enumerate(transcript):
                    counts["raw_segments"] += 1
                    if not isinstance(segment, dict):
                        raise ValueError(
                            f"harper_valley_bank:{sid}:{segment_index} is invalid"
                        )
                    text = segment.get("human_transcript")
                    if not isinstance(text, str) or not text.strip():
                        counts["empty_segments_omitted"] += 1
                        continue
                    if not _harper_has_lexical_content(text):
                        counts["marker_only_segments_omitted"] += 1
                        continue
                    speaker_role = segment.get("speaker_role")
                    turn_index = segment.get("index")
                    party = (
                        metadata.get(speaker_role)
                        if isinstance(speaker_role, str)
                        else None
                    )
                    if (
                        speaker_role not in {"caller", "agent"}
                        or type(turn_index) is not int
                        or not isinstance(party, dict)
                        or type(party.get("speaker_id")) is not int
                    ):
                        raise ValueError(
                            f"harper_valley_bank:{sid}:{segment_index} has invalid speaker data"
                        )
                    row = _sample(
                        text=text,
                        label=0,
                        attack_type=None,
                        source="harper_valley_bank",
                        source_split="complete_corpus",
                        source_id=f"{sid}:{turn_index}",
                        group_id=f"harper_valley_bank:{sid}",
                        split_group_id=f"harper_valley_bank:{sid}",
                        category="+".join(task_types),
                        input_channel=(
                            "direct_user"
                            if speaker_role == "caller"
                            else "model_output"
                        ),
                        label_basis=(
                            "simulated_banking_dialogue_task_construction_not_safety_annotation"
                        ),
                    )
                    row.update(
                        {
                            "source_conversation_id": sid,
                            "source_intents": task_types,
                            "source_language": "en",
                            "source_session": session,
                            "source_speaker_id": party["speaker_id"],
                            "source_speaker_role": speaker_role,
                            "source_transcript_kind": "human_corrected",
                            "source_turn_index": turn_index,
                        }
                    )
                    counts[f"retained_role:{speaker_role}"] += 1
                    yield _set_source_role(row, "candidate")
        finally:
            temporary.cleanup()
        profile.update(
            {
                "conversations": len(transcript_paths),
                "raw_segments": counts["raw_segments"],
                "retained_roles": {
                    key.removeprefix("retained_role:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("retained_role:")
                },
                "empty_segments_omitted": counts["empty_segments_omitted"],
                "marker_only_segments_omitted": counts["marker_only_segments_omitted"],
                "sessions": dict(sorted(sessions.items())),
                "intents": dict(sorted(intents.items())),
            }
        )

    return rows(), {"projected_git_tree": digest}, profile, None


def _tatqa_table_text(table: list[list[str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    for row in table:
        if not isinstance(row, list) or not all(isinstance(cell, str) for cell in row):
            raise ValueError("tatqa table has a non-string cell")
        writer.writerow(row)
    return output.getvalue().rstrip("\n")


def _tatqa_sample(
    *,
    text: str,
    split: str,
    source_id: str,
    context_id: str,
    category: str,
    input_channel: str,
    metadata: dict,
) -> dict:
    if not text.strip():
        raise ValueError(f"tatqa:{split}:{source_id} has empty detector text")
    row = _sample(
        text=text,
        label=0,
        attack_type=None,
        source="tatqa",
        source_split=split,
        source_id=f"{category}:{source_id}",
        group_id=f"tatqa:{context_id}",
        split_group_id=f"tatqa:{context_id}",
        category=category,
        input_channel=input_channel,
        label_basis="finance_qa_task_construction_not_human_safety_annotation",
    )
    row.update(
        {
            "source_context_id": context_id,
            "source_language": "en",
            **metadata,
        }
    )
    role = "candidate" if split == "train" else "dev_test"
    return _set_source_role(row, role)


def _tatqa_rows() -> tuple[Iterator[dict], dict[str, str], dict, Iterator[dict] | None]:
    expected = {
        "dataset_raw/tatqa_dataset_train.json": (
            "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69"
        ),
        "dataset_raw/tatqa_dataset_dev.json": (
            "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"
        ),
        "dataset_raw/tatqa_dataset_test.json": (
            "6efcf044cedeba3661eb70b1b93595673fd3f3dfcc1f78288ec5115682e7a96c"
        ),
    }
    contents = {}
    downloads = {}
    for filename, expected_digest in expected.items():
        data, digest = _github_pinned("tatqa", filename, expected_digest)
        split = filename.removeprefix("dataset_raw/tatqa_dataset_").removesuffix(
            ".json"
        )
        contents[split] = data
        downloads[filename] = digest
    profile = {
        "projection": (
            "human-written questions, report paragraphs, and reversible TSV table "
            "serializations from the official raw JSON"
        ),
        "excluded": (
            "answers, derivations, reasoning programs, answer types, supporting-fact "
            "labels, comparison flags, and scales"
        ),
        "label_limit": (
            "finance-QA task construction supports task relevance, not independent "
            "prompt-safety adjudication"
        ),
        "lineage_limit": (
            "the release exposes hybrid context IDs but no stable financial-report ID; "
            "all material from one hybrid context is grouped together"
        ),
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        for split in ("train", "dev", "test"):
            for context_index, context in enumerate(
                ijson.items(io.BytesIO(contents[split]), "item")
            ):
                if not isinstance(context, dict):
                    raise ValueError(
                        f"tatqa:{split}:{context_index} has invalid context"
                    )
                table = context.get("table")
                paragraphs = context.get("paragraphs")
                questions = context.get("questions")
                if (
                    not isinstance(table, dict)
                    or not isinstance(paragraphs, list)
                    or not isinstance(questions, list)
                ):
                    raise ValueError(
                        f"tatqa:{split}:{context_index} has invalid schema"
                    )
                context_id = table.get("uid")
                table_rows = table.get("table")
                if not isinstance(context_id, str) or not isinstance(table_rows, list):
                    raise ValueError(f"tatqa:{split}:{context_index} has invalid table")
                counts[f"contexts:{split}"] += 1
                for question in questions:
                    if not isinstance(question, dict):
                        raise ValueError(
                            f"tatqa:{split}:{context_id} has invalid question"
                        )
                    uid = question.get("uid")
                    text = question.get("question")
                    order = question.get("order")
                    if (
                        not isinstance(uid, str)
                        or not isinstance(text, str)
                        or type(order) is not int
                    ):
                        raise ValueError(
                            f"tatqa:{split}:{context_id} has invalid question fields"
                        )
                    yield _tatqa_sample(
                        text=text,
                        split=split,
                        source_id=uid,
                        context_id=context_id,
                        category="financial_question",
                        input_channel="direct_user",
                        metadata={
                            "source_question_order": order,
                            "source_question_uid": uid,
                        },
                    )
                    counts[f"questions:{split}"] += 1
                for paragraph in paragraphs:
                    if not isinstance(paragraph, dict):
                        raise ValueError(
                            f"tatqa:{split}:{context_id} has invalid paragraph"
                        )
                    uid = paragraph.get("uid")
                    text = paragraph.get("text")
                    order = paragraph.get("order")
                    if (
                        not isinstance(uid, str)
                        or not isinstance(text, str)
                        or type(order) is not int
                    ):
                        raise ValueError(
                            f"tatqa:{split}:{context_id} has invalid paragraph fields"
                        )
                    yield _tatqa_sample(
                        text=text,
                        split=split,
                        source_id=uid,
                        context_id=context_id,
                        category="financial_report_paragraph",
                        input_channel="untrusted_content",
                        metadata={
                            "source_paragraph_order": order,
                            "source_paragraph_uid": uid,
                        },
                    )
                    counts[f"paragraphs:{split}"] += 1
                yield _tatqa_sample(
                    text=_tatqa_table_text(table_rows),
                    split=split,
                    source_id=context_id,
                    context_id=context_id,
                    category="financial_report_table",
                    input_channel="untrusted_content",
                    metadata={
                        "source_serialization": "tsv",
                        "source_table_uid": context_id,
                    },
                )
                counts[f"tables:{split}"] += 1
        profile["official_split_counts"] = {
            split: {
                kind: counts[f"{kind}:{split}"]
                for kind in ("contexts", "questions", "paragraphs", "tables")
            }
            for split in ("train", "dev", "test")
        }

    return rows(), downloads, profile, None


def _financebench_rows() -> tuple[
    Iterator[dict], dict[str, str], dict, Iterator[dict] | None
]:
    filename = "data/financebench_open_source.jsonl"
    expected = "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877"
    data, digest = _github_pinned("financebench", filename, expected)
    source_rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(source_rows) != 150:
        raise ValueError("financebench public sample no longer has 150 examples")
    profile = {
        "projection": (
            "all 150 public questions and their annotated evidence_text passages as a "
            "development-only diagnostic"
        ),
        "excluded": (
            "answers, justifications, question reasoning, domain question numbers, "
            "full-page evidence text, PDFs, model results, and vector stores"
        ),
        "label_limit": (
            "expert annotations establish financial-QA correctness, not independent "
            "safety adjudication"
        ),
        "evaluation_limit": (
            "repeated development diagnostic only; it is not a prospective final test"
        ),
    }

    def rows() -> Iterator[dict]:
        counts = Counter()
        documents = set()
        for index, source_row in enumerate(source_rows):
            financebench_id = source_row.get("financebench_id")
            question = source_row.get("question")
            document = source_row.get("doc_name")
            company = source_row.get("company")
            question_type = source_row.get("question_type")
            evidence = source_row.get("evidence")
            if (
                not isinstance(financebench_id, str)
                or not isinstance(question, str)
                or not question.strip()
                or not isinstance(document, str)
                or not document
                or not isinstance(company, str)
                or not isinstance(question_type, str)
                or not isinstance(evidence, list)
                or source_row.get("dataset_subset_label") != "OPEN_SOURCE"
            ):
                raise ValueError(
                    f"financebench:{index} has invalid public-example schema"
                )
            documents.add(document)
            common = {
                "source_company": company,
                "source_document_name": document,
                "source_financebench_id": financebench_id,
                "source_language": "en",
            }
            row = _sample(
                text=question,
                label=0,
                attack_type=None,
                source="financebench",
                source_split="open_source_evaluation",
                source_id=f"question:{financebench_id}",
                group_id=f"financebench:{document}",
                split_group_id=f"financebench:{document}",
                category="financial_question",
                input_channel="direct_user",
                label_basis=(
                    "finance_qa_correctness_annotation_not_independent_safety_adjudication"
                ),
            )
            row.update({**common, "source_question_type": question_type})
            counts["questions"] += 1
            yield _set_source_role(row, "dev_test")
            for evidence_index, passage in enumerate(evidence):
                if not isinstance(passage, dict):
                    raise ValueError(
                        f"financebench:{financebench_id} has invalid evidence"
                    )
                text = passage.get("evidence_text")
                page = passage.get("evidence_page_num")
                evidence_document = passage.get("doc_name")
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or type(page) is not int
                    or evidence_document != document
                ):
                    raise ValueError(
                        f"financebench:{financebench_id}:{evidence_index} has invalid evidence fields"
                    )
                row = _sample(
                    text=text,
                    label=0,
                    attack_type=None,
                    source="financebench",
                    source_split="open_source_evaluation",
                    source_id=f"evidence:{financebench_id}:{evidence_index}",
                    group_id=f"financebench:{document}",
                    split_group_id=f"financebench:{document}",
                    category="financial_evidence_passage",
                    input_channel="untrusted_content",
                    label_basis=(
                        "finance_qa_correctness_annotation_not_independent_safety_adjudication"
                    ),
                )
                row.update(
                    {
                        **common,
                        "source_evidence_index": evidence_index,
                        "source_evidence_page_number": page,
                    }
                )
                counts["evidence_passages"] += 1
                yield _set_source_role(row, "dev_test")
        profile.update(
            {
                "documents": len(documents),
                "questions": counts["questions"],
                "evidence_passages": counts["evidence_passages"],
            }
        )

    return rows(), {filename: digest}, profile, None
