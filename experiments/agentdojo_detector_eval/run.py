#!/usr/bin/env python3
"""Freeze and score a pinned AgentDojo Banking detector diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import agentdojo
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.baseline_attacks import DirectAttack
from agentdojo.attacks.important_instructions_attacks import (
    ImportantInstructionsAttack,
)
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

from morgott.models.cascade import CascadeScanner
from morgott.models.deepseek_nooa import PROMPT_SHA256, REQUEST_SHA256
from morgott.models.downstream import THRESHOLD_SHA256
from morgott.models.mmbert.core import file_sha256
from morgott.models.mmbert.data import (
    OverlapGuard,
    canonical_rows,
    external_rows,
    filter_small_training_sets,
    matched_pairs,
    routing_views,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "agentdojo_detector_eval"
AGENTDOJO_VERSION = "0.1.35"
AGENTDOJO_TAG_REVISION = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
BENCHMARK_VERSION = "v1.2.2"
SUITE_NAME = "banking"
TARGET_PIPELINE_NAME = "gpt-4o-2024-05-13"
EXPECTED_TOOLS = {
    "get_balance",
    "get_iban",
    "get_most_recent_transactions",
    "get_scheduled_transactions",
    "get_user_info",
    "read_file",
    "schedule_transaction",
    "send_money",
    "update_password",
    "update_scheduled_transaction",
    "update_user_info",
}
SOURCE_PATHS = (
    "agent_pipeline/ground_truth_pipeline.py",
    "attacks/base_attacks.py",
    "attacks/baseline_attacks.py",
    "attacks/important_instructions_attacks.py",
    "base_tasks.py",
    "functions_runtime.py",
    "task_suite/load_suites.py",
    "task_suite/task_suite.py",
    "default_suites/v1/banking/injection_tasks.py",
    "default_suites/v1/banking/task_suite.py",
    "default_suites/v1/banking/user_tasks.py",
    "default_suites/v1_1_1/banking/user_tasks.py",
    "default_suites/v1_2/banking/injection_tasks.py",
    "default_suites/v1_2_2/banking/user_tasks.py",
    "data/suites/banking/environment.yaml",
    "data/suites/banking/injection_vectors.yaml",
)
ATTACK_NAMES = ("direct", "important_instructions")
PANEL_FIELDS = {
    "attack",
    "case_id",
    "fit_overlap",
    "fit_overlap_reason",
    "injection_task_id",
    "input_channel",
    "kind",
    "payload_chars",
    "payload_sha256",
    "text_chars",
    "text_sha256",
    "user_task_id",
    "vector_id",
}
BATCH_SIZE = 32


class _TargetName(BasePipelineElement):
    name = TARGET_PIPELINE_NAME

    def query(self, *args, **kwargs):
        raise NotImplementedError


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> dict:
    content = b"".join(_canonical_json(row) for row in rows)
    compressed = gzip.compress(content, compresslevel=9, mtime=0)
    _atomic_bytes(path, compressed)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "sha256": _sha256_bytes(compressed),
        "content_sha256": _sha256_bytes(content),
    }


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _task_key(item: tuple[str, object]) -> int:
    return int(item[0].rsplit("_", 1)[1])


def _message_trace(
    user_task, suite, injections: dict[str, str]
) -> list[tuple[str, str]]:
    environment = suite.load_and_inject_default_environment(injections)
    environment = user_task.init_environment(environment)
    _, _, _, responses, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT,
        FunctionsRuntime(suite.tools),
        environment,
    )
    return [
        (
            response["role"],
            (
                get_text_content_as_str(response["content"])
                if response.get("content") is not None
                else ""
            ),
        )
        for response in responses
    ]


def _source_contract() -> dict:
    installed = importlib.metadata.version("agentdojo")
    if installed != AGENTDOJO_VERSION:
        raise ValueError(f"expected AgentDojo {AGENTDOJO_VERSION}, got {installed}")
    package_root = Path(agentdojo.__file__).resolve().parent
    return {
        "project": "AgentDojo",
        "package_version": AGENTDOJO_VERSION,
        "tag_revision": AGENTDOJO_TAG_REVISION,
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "license": "MIT",
        "source_sha256": {
            relative: file_sha256(package_root / relative) for relative in SOURCE_PATHS
        },
    }


def _materialize() -> tuple[list[dict], dict[str, str], dict]:
    source = _source_contract()
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    if (
        len(suite.user_tasks) != 16
        or len(suite.injection_tasks) != 9
        or {tool.name for tool in suite.tools} != EXPECTED_TOOLS
    ):
        raise ValueError("pinned AgentDojo Banking surface changed")

    target = _TargetName()
    attacks = (
        DirectAttack(suite, target),
        ImportantInstructionsAttack(suite, target),
    )
    cases = []
    texts = {}
    baseline_traces = {}
    for user_task_id, user_task in sorted(suite.user_tasks.items(), key=_task_key):
        case_id = f"user:{user_task_id}"
        text = user_task.PROMPT
        texts[case_id] = text
        cases.append(
            {
                "attack": None,
                "case_id": case_id,
                "injection_task_id": None,
                "input_channel": "direct_user",
                "kind": "user_prompt",
                "payload_chars": None,
                "payload_sha256": None,
                "text_chars": len(text),
                "text_sha256": _sha256_bytes(text.encode()),
                "user_task_id": user_task_id,
                "vector_id": None,
            }
        )
        baseline_traces[user_task_id] = _message_trace(user_task, suite, {})

    for attack in attacks:
        for user_task_id, user_task in sorted(suite.user_tasks.items(), key=_task_key):
            baseline = baseline_traces[user_task_id]
            for injection_task_id, injection_task in sorted(
                suite.injection_tasks.items(), key=_task_key
            ):
                injections = attack.attack(user_task, injection_task)
                if len(injections) != 1:
                    raise ValueError(
                        "Banking case no longer exposes exactly one vector"
                    )
                vector_id, payload = next(iter(injections.items()))
                attacked = _message_trace(user_task, suite, injections)
                if len(attacked) != len(baseline):
                    raise ValueError("injection changed the ground-truth trace shape")
                changed = [
                    after
                    for before, after in zip(baseline, attacked, strict=True)
                    if before != after
                ]
                if len(changed) != 1 or changed[0][0] != "tool":
                    raise ValueError("could not isolate one injected tool-return text")
                text = changed[0][1]
                case_id = f"attack:{attack.name}:{user_task_id}:{injection_task_id}"
                texts[case_id] = text
                cases.append(
                    {
                        "attack": attack.name,
                        "case_id": case_id,
                        "injection_task_id": injection_task_id,
                        "input_channel": "untrusted_content",
                        "kind": "tool_return",
                        "payload_chars": len(payload),
                        "payload_sha256": _sha256_bytes(payload.encode()),
                        "text_chars": len(text),
                        "text_sha256": _sha256_bytes(text.encode()),
                        "user_task_id": user_task_id,
                        "vector_id": vector_id,
                    }
                )

    tool_cases = [case for case in cases if case["kind"] == "tool_return"]
    expected = {
        "cases": 304,
        "user_prompts": 16,
        "tool_returns": 288,
        "unique_user_prompts": 16,
        "unique_tool_returns": 72,
        "unique_payloads": 18,
    }
    observed = {
        "cases": len(cases),
        "user_prompts": len(cases) - len(tool_cases),
        "tool_returns": len(tool_cases),
        "unique_user_prompts": len(
            {case["text_sha256"] for case in cases if case["kind"] == "user_prompt"}
        ),
        "unique_tool_returns": len({case["text_sha256"] for case in tool_cases}),
        "unique_payloads": len({case["payload_sha256"] for case in tool_cases}),
    }
    if observed != expected or {case["attack"] for case in tool_cases} != set(
        ATTACK_NAMES
    ):
        raise ValueError(f"pinned AgentDojo projection changed: {observed}")
    return cases, texts, source


def _fit_references(counts: Counter):
    views = routing_views(ROOT / "data")
    external = external_rows(ROOT / "artifacts" / "mmbert" / "data")[0]

    def counted(name: str, rows):
        for row in rows:
            counts[name] += 1
            yield row

    return chain(
        counted("canonical_train", canonical_rows(*views["train"], split="train")),
        counted("promptshield_train", external["promptshield_train"]),
        counted(
            "matched_pairs",
            chain.from_iterable(
                matched_pairs(ROOT / "data-archive" / "matched_pairs_20260726.jsonl.gz")
            ),
        ),
    )


def _candidate_id(case: dict) -> str:
    return f"{case['input_channel']}:{case['text_sha256']}"


def _prepare(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.exists() or (output / "panel.jsonl.gz").exists():
        raise FileExistsError(
            "AgentDojo detector panel is write-once; use a fresh output"
        )
    cases, texts, source = _materialize()
    unique = {}
    for case in cases:
        identity = _candidate_id(case)
        text = texts[case["case_id"]]
        previous = unique.setdefault(
            identity,
            {
                "id": identity,
                "text": text,
                "source": "agentdojo",
                "label": int(case["kind"] == "tool_return"),
            },
        )
        if previous["text"] != text or previous["label"] != int(
            case["kind"] == "tool_return"
        ):
            raise ValueError("AgentDojo input hash collision")
    candidates = {
        "agentdojo_user": [row for row in unique.values() if row["label"] == 0],
        "agentdojo_tool": [row for row in unique.values() if row["label"] == 1],
    }
    reference_counts = Counter()
    guard = OverlapGuard(())
    _, removed = filter_small_training_sets(
        candidates,
        _fit_references(reference_counts),
        reference_guard=guard,
    )
    reasons = {identity: guard.reason(row) for identity, row in unique.items()}
    panel = [
        {
            **case,
            "fit_overlap": reasons[_candidate_id(case)] is not None,
            "fit_overlap_reason": reasons[_candidate_id(case)],
        }
        for case in cases
    ]
    panel_spec = _write_gzip_jsonl(output / "panel.jsonl.gz", panel)
    tool_cases = [case for case in panel if case["kind"] == "tool_return"]
    user_cases = [case for case in panel if case["kind"] == "user_prompt"]
    manifest = {
        "schema_version": 1,
        "purpose": "frozen external AgentDojo Banking advisory-cascade diagnostic",
        "source": source,
        "projection": {
            "attacks": list(ATTACK_NAMES),
            "rendered_model_name_source": TARGET_PIPELINE_NAME,
            "positive_input": (
                "complete serialized tool-return text whose value changes when the "
                "single reachable injection vector is populated"
            ),
            "negative_input": "unattacked source-authored user task prompt",
            "input_channels": {
                "tool_return": "untrusted_content",
                "user_prompt": "direct_user",
            },
            "stored_text": False,
        },
        "selection": {
            "fit_overlap_filter": (
                "normalized, audit-strict, and conservative near overlap against "
                "every full-LoRA fit candidate"
            ),
            "reference_rows": dict(sorted(reference_counts.items())),
            "removed_unique_inputs_by_reason": removed,
        },
        "population": {
            "cases": len(panel),
            "user_prompt_cases": len(user_cases),
            "tool_return_cases": len(tool_cases),
            "unique_user_prompts": len({case["text_sha256"] for case in user_cases}),
            "unique_tool_returns": len({case["text_sha256"] for case in tool_cases}),
            "unique_payloads": len({case["payload_sha256"] for case in tool_cases}),
            "fit_overlapping_cases": sum(case["fit_overlap"] for case in panel),
        },
        "analysis_contract": {
            "primary": (
                "non-overlap complete-tool-return restriction rates by attack, "
                "exact text, payload, user task, injection task, and vector"
            ),
            "legitimate_stress": (
                "restriction count on 16 non-overlap user prompts; not a production FPR"
            ),
            "threshold_selection": "none; use the registered cascade unchanged",
        },
        "cascade_contract": {
            "model_registry_sha256": file_sha256(ROOT / "model-artifacts.json"),
            "prompt_sha256": PROMPT_SHA256,
            "request_sha256": REQUEST_SHA256,
            "threshold_sha256": THRESHOLD_SHA256,
        },
        "overlap_contract": {
            path: file_sha256(ROOT / path)
            for path in (
                "data/manifest.json",
                "artifacts/mmbert/data/manifest.json",
                "data-archive/matched_pairs_20260726.jsonl.gz",
                "src/morgott/data.py",
                "src/morgott/normalization.py",
                "src/morgott/overlap.py",
                "src/morgott/models/mmbert/data.py",
            )
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "panel": panel_spec,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_panel(output: Path) -> tuple[list[dict], dict, str]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported AgentDojo detector manifest")
    panel_path = output / "panel.jsonl.gz"
    panel_bytes = panel_path.read_bytes()
    spec = manifest["panel"]
    if (
        _sha256_bytes(panel_bytes) != spec["sha256"]
        or _sha256_bytes(gzip.decompress(panel_bytes)) != spec["content_sha256"]
    ):
        raise ValueError("AgentDojo detector panel hash mismatch")
    panel = _read_jsonl(panel_path)
    if len(panel) != spec["rows"] or any(set(row) != PANEL_FIELDS for row in panel):
        raise ValueError("AgentDojo detector panel schema mismatch")
    case_ids = [row["case_id"] for row in panel]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("AgentDojo detector panel has duplicate cases")
    return panel, manifest, file_sha256(manifest_path)


def _validate_contract(manifest: dict) -> None:
    cascade = {
        "model_registry_sha256": file_sha256(ROOT / "model-artifacts.json"),
        "prompt_sha256": PROMPT_SHA256,
        "request_sha256": REQUEST_SHA256,
        "threshold_sha256": THRESHOLD_SHA256,
    }
    overlap = {path: file_sha256(ROOT / path) for path in manifest["overlap_contract"]}
    if (
        manifest["source"] != _source_contract()
        or manifest["cascade_contract"] != cascade
        or manifest["overlap_contract"] != overlap
        or manifest["runner_sha256"] != file_sha256(Path(__file__))
    ):
        raise ValueError("AgentDojo detector experiment contract changed")


def _result_rows(output: Path) -> list[dict]:
    compressed = output / "results.jsonl.gz"
    raw = output / "results.jsonl"
    if raw.exists():
        return _read_jsonl(raw)
    return _read_jsonl(compressed) if compressed.exists() else []


def _expected_inputs(panel: list[dict]) -> dict[str, dict]:
    expected = {}
    for row in panel:
        identity = _candidate_id(row)
        previous = expected.setdefault(identity, row)
        if previous["text_sha256"] != row["text_sha256"]:
            raise ValueError("AgentDojo input identity collision")
    return expected


def _validate_results(
    rows: list[dict], panel: list[dict], manifest_sha256: str
) -> dict[str, dict]:
    expected = _expected_inputs(panel)
    result = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"assessment", "input_id", "manifest_sha256", "schema_version"}
            or row["schema_version"] != 1
            or row["manifest_sha256"] != manifest_sha256
            or row["input_id"] not in expected
            or row["input_id"] in result
            or not isinstance(row["assessment"], dict)
        ):
            raise ValueError("invalid AgentDojo detector result row")
        assessment = row["assessment"]
        panel_row = expected[row["input_id"]]
        if (
            assessment.get("artifact_sha256") != panel_row["text_sha256"]
            or assessment.get("input_channel") != panel_row["input_channel"]
            or assessment.get("decision") != "allow"
            or assessment.get("advisory_only") is not True
            or assessment.get("prompt_sha256") != PROMPT_SHA256
            or assessment.get("provider_request_sha256") != REQUEST_SHA256
            or assessment.get("threshold_sha256") != THRESHOLD_SHA256
        ):
            raise ValueError("AgentDojo detector assessment identity mismatch")
        result[row["input_id"]] = row
    return result


def _validate_materialization(panel: list[dict]) -> dict[str, str]:
    current, texts, _ = _materialize()
    current_by_id = {row["case_id"]: row for row in current}
    if set(current_by_id) != {row["case_id"] for row in panel}:
        raise ValueError("AgentDojo detector case identities changed")
    for row in panel:
        source_row = current_by_id[row["case_id"]]
        if any(source_row[key] != row[key] for key in source_row):
            raise ValueError(f"AgentDojo detector case changed: {row['case_id']}")
        text = texts[row["case_id"]]
        if _sha256_bytes(text.encode()) != row["text_sha256"]:
            raise ValueError("AgentDojo detector source text hash changed")
    return texts


async def _run_async(output: Path) -> dict:
    panel, manifest, manifest_sha256 = _load_panel(output)
    _validate_contract(manifest)
    texts_by_case = _validate_materialization(panel)
    expected = _expected_inputs(panel)
    texts = {}
    for row in panel:
        identity = _candidate_id(row)
        text = texts_by_case[row["case_id"]]
        previous = texts.setdefault(identity, text)
        if previous != text:
            raise ValueError("AgentDojo detector text hash collision")
    existing_rows = _result_rows(output)
    existing = _validate_results(existing_rows, panel, manifest_sha256)
    pending = [
        identity
        for identity in expected
        if identity not in existing
        or existing[identity]["assessment"].get("complete") is not True
    ]
    raw_path = output / "results.jsonl"
    if pending:
        _atomic_bytes(
            raw_path,
            b"".join(
                _canonical_json(row)
                for row in existing_rows
                if row["assessment"].get("complete") is True
            ),
        )
    scanner = CascadeScanner.from_artifacts(
        manifest_path=ROOT / "model-artifacts.json",
    )
    try:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            assessments = await asyncio.gather(
                *(
                    scanner.assess_text(
                        texts[identity],
                        input_channel=expected[identity]["input_channel"],
                    )
                    for identity in batch
                )
            )
            with raw_path.open("a", encoding="utf-8") as handle:
                for identity, assessment in zip(batch, assessments, strict=True):
                    result = {
                        "assessment": asdict(assessment),
                        "input_id": identity,
                        "manifest_sha256": manifest_sha256,
                        "schema_version": 1,
                    }
                    handle.write(_canonical_json(result).decode())
                    existing[identity] = result
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        await scanner.aclose()
    if set(existing) != set(expected):
        raise AssertionError("AgentDojo detector scoring did not complete")
    ordered = [existing[identity] for identity in expected]
    result_spec = _write_gzip_jsonl(output / "results.jsonl.gz", ordered)
    raw_path.unlink(missing_ok=True)
    return {"pending_scored": len(pending), "results": result_spec}


def _rate(rows: list[dict]) -> dict:
    restricted = sum(row["restrict"] for row in rows)
    return {
        "cases": len(rows),
        "restricted": restricted,
        "restriction_rate": restricted / len(rows) if rows else None,
    }


def _unique_rate(rows: list[dict]) -> dict:
    unique = {}
    for row in rows:
        previous = unique.setdefault(row["input_id"], row["restrict"])
        if previous != row["restrict"]:
            raise ValueError("one AgentDojo input has inconsistent routes")
    return _rate([{"restrict": value} for value in unique.values()])


def _macro_rate(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["restrict"])
    rates = [sum(values) / len(values) for values in grouped.values()]
    return {
        "groups": len(rates),
        "mean_restriction_rate": statistics.fmean(rates) if rates else None,
        "minimum_restriction_rate": min(rates, default=None),
        "maximum_restriction_rate": max(rates, default=None),
    }


def _payload_summary(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["payload_sha256"]].append(row["restrict"])
    rates = [sum(values) / len(values) for values in grouped.values()]
    return {
        **_macro_rate(rows, "payload_sha256"),
        "all_contexts_restricted": sum(rate == 1 for rate in rates),
        "any_context_restricted": sum(rate > 0 for rate in rates),
    }


def _attack_summary(rows: list[dict]) -> dict:
    vectors = {}
    for vector in sorted({row["vector_id"] for row in rows}):
        selected = [row for row in rows if row["vector_id"] == vector]
        vectors[vector] = {
            "case_rate": _rate(selected),
            "unique_tool_return_rate": _unique_rate(selected),
        }
    return {
        "case_rate": _rate(rows),
        "unique_tool_return_rate": _unique_rate(rows),
        "payload_macro": _payload_summary(rows),
        "user_task_macro": _macro_rate(rows, "user_task_id"),
        "injection_task_macro": _macro_rate(rows, "injection_task_id"),
        "vectors": vectors,
    }


def _analyze(output: Path) -> dict:
    panel, manifest, manifest_sha256 = _load_panel(output)
    _validate_contract(manifest)
    results = _result_rows(output)
    by_id = _validate_results(results, panel, manifest_sha256)
    if set(by_id) != set(_expected_inputs(panel)):
        raise ValueError("AgentDojo detector result ledger is incomplete")
    joined = []
    for row in panel:
        input_id = _candidate_id(row)
        assessment = by_id[input_id]["assessment"]
        joined.append(
            {
                **row,
                "input_id": input_id,
                "restrict": assessment["advisory_route"] == "restrict",
            }
        )
    clean = [row for row in joined if not row["fit_overlap"]]
    attack_rows = [row for row in clean if row["kind"] == "tool_return"]
    user_rows = [row for row in clean if row["kind"] == "user_prompt"]
    assessments = [row["assessment"] for row in by_id.values()]
    reviewed = [
        review
        for assessment in assessments
        for review in assessment["reviewed_windows"]
    ]
    summary = {
        "schema_version": 1,
        "purpose": manifest["purpose"],
        "manifest_sha256": manifest_sha256,
        "panel_sha256": manifest["panel"]["sha256"],
        "result_ledger": {
            "path": str((output / "results.jsonl.gz").relative_to(ROOT)),
            "rows": len(results),
            "sha256": file_sha256(output / "results.jsonl.gz"),
            "content_sha256": _sha256_bytes(
                gzip.decompress((output / "results.jsonl.gz").read_bytes())
            ),
        },
        "primary": {
            attack: _attack_summary(
                [row for row in attack_rows if row["attack"] == attack]
            )
            for attack in ATTACK_NAMES
        },
        "legitimate_user_prompt_stress": _rate(user_rows),
        "overlap_audit": {
            "cases": len(joined),
            "overlapping_cases": len(joined) - len(clean),
            "reasons": dict(
                sorted(
                    Counter(
                        row["fit_overlap_reason"]
                        for row in joined
                        if row["fit_overlap_reason"] is not None
                    ).items()
                )
            ),
        },
        "cascade": {
            "unique_inputs": len(assessments),
            "complete_inputs": sum(
                assessment["complete"] for assessment in assessments
            ),
            "routes": dict(
                sorted(
                    Counter(
                        assessment["advisory_route"] for assessment in assessments
                    ).items()
                )
            ),
            "reasons": dict(
                sorted(
                    Counter(assessment["reason"] for assessment in assessments).items()
                )
            ),
            "deepseek_calls": sum(
                assessment["deepseek_calls"] for assessment in assessments
            ),
            "deepseek_failures": sum(
                assessment["deepseek_failures"] for assessment in assessments
            ),
            "reviewed_windows": len(reviewed),
            "input_tokens": sum(review["input_tokens"] or 0 for review in reviewed),
            "output_tokens": sum(review["output_tokens"] or 0 for review in reviewed),
            "local_latency_seconds": sum(
                assessment["local_latency_ms"] for assessment in assessments
            )
            / 1000,
            "provider_latency_seconds": sum(
                assessment["provider_latency_ms"] for assessment in assessments
            )
            / 1000,
        },
        "limitations": [
            "AgentDojo is an already-public, split-free development benchmark.",
            "The 288 attacked cases contain only 18 payloads and 72 exact tool-return texts.",
            "The 16 user prompts are a stress slice, not a production FPR denominator.",
            "This detector-only study does not measure agent behavior, task utility, authorization, or attack success.",
            "The fixed attacks do not support an adaptive robustness claim.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = _prepare(output)
    elif args.command == "run":
        if not args.allow_remote:
            raise ValueError("run requires explicit --allow-remote consent")
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        result = asyncio.run(_run_async(output))
    else:
        result = _analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
