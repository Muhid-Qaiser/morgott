#!/usr/bin/env python3
"""Collect selected-provider scores for reachable cascade windows.

The stage rehydrates text only in memory, reconstructs the registered
1,024-token windows with 128-token overlap, and persists no text or raw
provider response. Full-context-first behavior is simulated later from the
artifact and window ledgers; concurrency here changes transport only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiohttp

from experiments.pipeline_benchmark import local, metrics, providers, run
from morgott.models.mmbert.serving import MmbertRuntime

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pipeline_benchmark" / "20260816"
STAGE = "cascade_windows"
MAX_REMOTE_WINDOWS = 128


@dataclass(frozen=True, slots=True)
class WindowJob:
    job_id: str
    artifact_id: str
    artifact_text_sha256: str
    window_index: int
    char_start: int
    char_end: int
    text_sha256: str
    input_channel: str
    local_score: float
    endpoint: providers.Endpoint
    transport: providers.Transport
    text: str


def _sha256(path: Path) -> str:
    return local.file_sha256(path)


def _registered_preprocessor() -> MmbertRuntime:
    registry = run._json(local.MODEL_REGISTRY)
    entry = registry.get("models", {}).get(local.MODEL_KEY)
    serving = entry.get("serving") if isinstance(entry, dict) else None
    tokenizer = serving.get("tokenizer") if isinstance(serving, dict) else None
    if (
        not isinstance(tokenizer, dict)
        or serving.get("max_tokens") != local.MAX_TOKENS
        or serving.get("window_overlap") != local.WINDOW_OVERLAP
        or not isinstance(tokenizer.get("path"), str)
        or not isinstance(tokenizer.get("sha256"), str)
    ):
        raise ValueError("registered serving window contract changed")
    path = (ROOT / tokenizer["path"]).resolve()
    if not path.is_relative_to(ROOT) or _sha256(path) != tokenizer["sha256"]:
        raise ValueError("registered serving tokenizer identity changed")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "install the cascade extra to prepare provider windows"
        ) from error
    return MmbertRuntime(
        tokenizer=Tokenizer.from_file(str(path)),
        session=None,
        max_tokens=local.MAX_TOKENS,
        window_overlap=local.WINDOW_OVERLAP,
    )


def _approved_thresholds() -> tuple[dict[str, float], ...]:
    return tuple(
        {
            "direct_low": float(direct_low),
            "untrusted_low": float(untrusted_low),
            "local_high": float(local_high),
        }
        for direct_low in metrics.DIRECT_LOW_GRID
        for untrusted_low in metrics.UNTRUSTED_LOW_GRID
        for local_high in metrics.LOCAL_HIGH_GRID
    )


def _target_keys(
    summary: dict[str, Any], *, expected_panel_rows: int
) -> tuple[tuple[str, providers.Transport], ...]:
    logprob = summary.get("winners", {}).get("logprob")
    if not isinstance(logprob, dict):
        raise RuntimeError("the provisional logprob winner must exist")
    keys = [(logprob.get("provider"), logprob.get("transport"))]
    summaries = summary.get("providers")
    if not isinstance(summaries, dict):
        raise ValueError("provider summary has no completed providers")
    keys.extend(
        (value.get("provider"), value.get("transport"))
        for value in summaries.values()
        if isinstance(value, dict)
        and value.get("provider") in run.HEADLINE_EXPANSION
        and value.get("transport") == "strict_hard_verdict"
        and value.get("rows") == expected_panel_rows
    )
    if any(not isinstance(provider, str) for provider, _ in keys):
        raise ValueError("provider summary target is malformed")
    return tuple(dict.fromkeys(keys))


def _selected_targets(
    output: Path,
) -> tuple[tuple[providers.Endpoint, providers.Transport], ...]:
    summary = run._json(output / "provider_summary.json")
    expected_panel_rows = len(
        run._require_manifest(output)["roles"]["provider_panel_ids"]
    )
    requested = _target_keys(summary, expected_panel_rows=expected_panel_rows)
    candidates = {
        (endpoint.provider, transport): (endpoint, transport)
        for endpoint, transport in run._provider_candidates(output)
    }
    records = run._all_provider_records(output)
    targets = []
    for key in requested:
        winner = candidates.get(key)
        if winner is None or not run._canary_passed(records, str(key[0])):
            raise RuntimeError("selected provider is not canary eligible")
        if winner not in targets:
            targets.append(winner)
    if not any(transport == "strict_hard_verdict" for _, transport in targets):
        raise RuntimeError("no completed strict-hard provider is available")
    return tuple(targets)


def _reachable_window_indices(
    scores: list[float],
    input_channel: str,
    thresholds: tuple[dict[str, float], ...],
) -> set[int]:
    if input_channel not in {"direct_user", "untrusted_content"}:
        raise ValueError("input channel is invalid")
    if (
        not scores
        or len(scores) > MAX_REMOTE_WINDOWS
        or any(
            not isinstance(score, int | float) or not math.isfinite(score)
            for score in scores
        )
    ):
        raise ValueError("window scores violate the registered contract")
    result = set()
    low_key = "direct_low" if input_channel == "direct_user" else "untrusted_low"
    for profile in thresholds:
        high = profile["local_high"]
        if any(score >= high for score in scores):
            continue
        result.update(
            index
            for index, score in enumerate(scores)
            if profile[low_key] <= score < high
        )
    return result


def build_jobs(
    panel: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    texts: dict[str, str],
    *,
    safe_ids: set[str],
    thresholds: tuple[dict[str, float], ...],
    winners: tuple[tuple[providers.Endpoint, providers.Transport], ...],
    preprocessor: Any,
) -> list[WindowJob]:
    jobs = []
    for row in panel:
        artifact_id = row["panel_id"]
        score = scores.get(artifact_id)
        if artifact_id not in safe_ids or not isinstance(score, dict):
            continue
        window_scores = score.get("window_scores")
        if not isinstance(window_scores, list) or len(window_scores) <= 1:
            continue
        if (
            score.get("window_count") != len(window_scores)
            or score.get("text_sha256") != row["text_sha256"]
        ):
            raise ValueError("local window score identity changed")
        prepared = preprocessor.prepare(texts[artifact_id])
        if len(prepared.windows) != len(
            window_scores
        ) or prepared.token_count != score.get("token_count"):
            raise ValueError("registered window spans differ from local score order")
        indices = _reachable_window_indices(
            window_scores,
            row["input_channel"],
            thresholds,
        )
        for index in sorted(indices):
            window = prepared.windows[index]
            text = prepared.normalized_text[window.char_start : window.char_end]
            text_sha256 = hashlib.sha256(text.encode()).hexdigest()
            for endpoint, transport in winners:
                job_id = hashlib.sha256(
                    (
                        f"{STAGE}\0{endpoint.tag}\0{transport}\0{artifact_id}\0"
                        f"{index}\0{window.char_start}\0{window.char_end}\0"
                        f"{text_sha256}\0{window_scores[index]}"
                    ).encode()
                ).hexdigest()
                jobs.append(
                    WindowJob(
                        job_id=job_id,
                        artifact_id=artifact_id,
                        artifact_text_sha256=row["text_sha256"],
                        window_index=index,
                        char_start=window.char_start,
                        char_end=window.char_end,
                        text_sha256=text_sha256,
                        input_channel=row["input_channel"],
                        local_score=float(window_scores[index]),
                        endpoint=endpoint,
                        transport=transport,
                        text=text,
                    )
                )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("cascade-window job identities are not unique")
    return jobs


def _load_jobs(output: Path) -> tuple[list[WindowJob], dict[str, Any]]:
    manifest = run._require_manifest(output)
    panel = local.load_frozen_panel()
    score_path = output / "morgott_1024_scores.jsonl.gz"
    scores = {row["artifact_id"]: row for row in run._read_jsonl(score_path)}
    if len(scores) != len(panel):
        raise ValueError("complete local window scores are required")
    safe_ids = set(manifest["roles"]["provider_safe_calibration_panel_ids"]) | set(
        manifest["roles"]["provider_safe_evaluation_panel_ids"]
    )
    eligible_panel = [
        row
        for row in panel
        if row["panel_id"] in safe_ids
        and scores[row["panel_id"]].get("window_count", 0) > 1
    ]
    texts = local.load_frozen_texts(eligible_panel)
    thresholds = _approved_thresholds()
    winners = _selected_targets(output)
    jobs = build_jobs(
        eligible_panel,
        scores,
        texts,
        safe_ids=safe_ids,
        thresholds=thresholds,
        winners=winners,
        preprocessor=_registered_preprocessor(),
    )
    return jobs, {
        "manifest_sha256": _sha256(output / "manifest.json"),
        "local_scores_sha256": _sha256(score_path),
        "provider_summary_sha256": _sha256(output / "provider_summary.json"),
        "metrics_code_sha256": _sha256(Path(metrics.__file__)),
        "model_key": local.MODEL_KEY,
        "max_tokens": local.MAX_TOKENS,
        "window_overlap": local.WINDOW_OVERLAP,
    }


def plan(output: Path) -> dict[str, Any]:
    jobs, inputs = _load_jobs(output)
    result = {
        "stage": STAGE,
        "jobs": len(jobs),
        "artifacts": len({job.artifact_id for job in jobs}),
        "windows": len({(job.artifact_id, job.window_index) for job in jobs}),
        "providers": sorted({job.endpoint.provider for job in jobs}),
        "raw_text_persisted": False,
        "raw_responses_persisted": False,
        "full_context_first_simulated_later": True,
        "network_concurrency_changes_semantics": False,
        "inputs": inputs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _finalize(
    output: Path,
    jobs: list[WindowJob],
    inputs: dict[str, Any],
    *,
    concurrency: int,
) -> None:
    result_path = run._provider_result_path(output, STAGE)
    records = run._loaded_provider_records(result_path)
    if (
        len(records) != len(jobs)
        or len({row.get("job_id") for row in records}) != len(jobs)
        or {row.get("job_id") for row in records} != {job.job_id for job in jobs}
    ):
        raise RuntimeError("cascade-window ledger is incomplete or duplicated")
    run._atomic_json(
        output / "provider_cascade_windows_run.json",
        {
            "stage": STAGE,
            "concurrency": concurrency,
            "jobs": len(jobs),
            "artifacts": len({job.artifact_id for job in jobs}),
            "windows": len({(job.artifact_id, job.window_index) for job in jobs}),
            "providers": sorted({job.endpoint.provider for job in jobs}),
            "model": providers.MODEL,
            "full_context_first_simulated_later": True,
            "network_concurrency_changes_semantics": False,
            "recorded_total_spend_usd": str(run._recorded_spend(output)),
            "result_path": str(result_path.relative_to(ROOT)),
            "result_sha256": _sha256(result_path),
            "inputs": inputs,
        },
    )


async def execute(output: Path, *, concurrency: int) -> None:
    if concurrency not in {1, 4, 8}:
        raise ValueError("provider concurrency must be 1, 4, or 8")
    jobs, inputs = _load_jobs(output)
    result_path = run._provider_result_path(output, STAGE)
    completed = {row.get("job_id") for row in run._loaded_provider_records(result_path)}
    pending = [job for job in jobs if job.job_id not in completed]
    if not pending:
        _finalize(output, jobs, inputs, concurrency=concurrency)
        print("No pending cascade-window provider calls.")
        return
    maximum_estimate = sum(
        (
            providers.request_cost_ceiling(
                job.endpoint,
                input_bytes=len(
                    (
                        job.text
                        + providers.PROMPT.format(input_channel=job.input_channel)
                    ).encode()
                ),
            )
            for job in pending
        ),
        Decimal("0"),
    )
    run._reserve_budget(output, "provider:cascade-windows", maximum_estimate)

    queue: asyncio.Queue[WindowJob] = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    lock = asyncio.Lock()
    api_key = run._api_key()
    timeout = aiohttp.ClientTimeout(total=180, connect=20, sock_read=160)
    with result_path.open("a", encoding="utf-8") as handle:
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=concurrency),
        ) as session:

            async def worker() -> None:
                while True:
                    try:
                        job = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    row = {
                        "panel_id": job.artifact_id,
                        "input_channel": job.input_channel,
                        "text_sha256": job.text_sha256,
                    }
                    record = await run._call_provider(
                        session,
                        api_key,
                        endpoint=job.endpoint,
                        transport=job.transport,
                        row=row,
                        text=job.text,
                        stage=STAGE,
                    )
                    record.update(
                        {
                            "job_id": job.job_id,
                            "artifact_id": job.artifact_id,
                            "artifact_text_sha256": job.artifact_text_sha256,
                            "window_index": job.window_index,
                            "char_start": job.char_start,
                            "char_end": job.char_end,
                            "text_sha256": job.text_sha256,
                            "local_score": job.local_score,
                            "window_text_sha256": job.text_sha256,
                            "window_local_score": job.local_score,
                        }
                    )
                    async with lock:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    queue.task_done()

            await asyncio.gather(
                *(worker() for _ in range(min(concurrency, len(pending))))
            )

    spent_after = run._recorded_spend(output)
    providers.BudgetLedger(spent_usd=spent_after)
    _finalize(output, jobs, inputs, concurrency=concurrency)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "plan":
        plan(output)
    else:
        asyncio.run(execute(output, concurrency=args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
