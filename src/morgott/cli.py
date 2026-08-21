from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .corpus import build_corpus, rebuild_routing
from .models.cascade import ALLOWED_CHANNELS, CascadeScanner
from .models.detector import run_benchmark, scan
from .models.routing_baseline import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_PER_SOURCE_LABEL,
    run_routing_baseline,
)
from .policy import run_policy_ablation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morgott")
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data", help="download and consolidate pinned data")
    data.add_argument("--data-dir", type=Path, default=Path("data"))
    data.add_argument(
        "--routing-only",
        action="store_true",
        help="rebuild routing views from manifest-verified canonical source shards",
    )

    benchmark = subcommands.add_parser(
        "benchmark", help="train and evaluate the legacy shadow control"
    )
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))
    benchmark.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--reports-dir", type=Path, default=Path("reports"))

    routing_baseline = subcommands.add_parser(
        "routing-baseline",
        help="train the direct-user word n-gram routing control",
    )
    routing_baseline.add_argument("--data-dir", type=Path, default=Path("data"))
    routing_baseline.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts")
    )
    routing_baseline.add_argument("--reports-dir", type=Path, default=Path("reports"))
    routing_baseline.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    routing_baseline.add_argument(
        "--max-per-source-label",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE_LABEL,
    )

    demo = subcommands.add_parser("demo", help="run the action-policy ablation")
    demo.add_argument("--reports-dir", type=Path, default=Path("reports"))

    scan_parser = subcommands.add_parser("scan", help="score one input in shadow mode")
    scan_parser.add_argument("text")
    scan_parser.add_argument(
        "--channel",
        choices=("direct_user", "untrusted_content"),
        default="direct_user",
        help="trusted input provenance selecting the channel-specific sensor",
    )
    scan_parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/guard_bundle.joblib"),
        help="trusted local artifact produced by the benchmark command",
    )

    cascade = subcommands.add_parser(
        "cascade",
        help="run the maintained advisory mmBERT and DeepSeek cascade",
    )
    cascade.add_argument(
        "input",
        nargs="?",
        help="UTF-8 text file, or - for stdin",
    )
    cascade.add_argument(
        "--input-channel",
        choices=("direct_user", "untrusted_content"),
        help="trusted runtime provenance for this input (required without --jsonl)",
    )
    cascade.add_argument(
        "--jsonl",
        help=(
            "batch mode: JSONL file of {text, input_channel} records, or - for"
            " stdin; emits one result JSON per line and builds the scanner once"
        ),
    )
    cascade.add_argument(
        "--manifest",
        type=Path,
        default=Path("model-artifacts.json"),
    )
    # usage_error keeps argparse semantics (usage text, exit code 2) for the
    # post-parse required/mutual-exclusion checks in main.
    cascade.set_defaults(usage_error=cascade.error)
    return parser


async def _input_chunks(value: str):
    owned = value != "-"
    handle = Path(value).open(encoding="utf-8", newline="") if owned else sys.stdin
    try:
        while chunk := handle.read(1 << 20):
            yield chunk
    finally:
        if owned:
            handle.close()


def _require_path(value: str) -> None:
    # exists and not isdir, never is_file: FIFOs and /dev/stdin must keep
    # working, and isdir stays False for both.
    if value == "-":
        return
    if not os.path.exists(value):
        raise SystemExit(f"morgott cascade: input not found: {value}")
    if os.path.isdir(value):
        raise SystemExit(f"morgott cascade: input is a directory: {value}")
    # os.access never opens the file, so FIFOs and /dev/stdin keep working
    # and the eventual open still backstops any access() misjudgment.
    if not os.access(value, os.R_OK):
        raise SystemExit(f"morgott cascade: input is not readable: {value}")


async def _run_cascade(args) -> dict:
    _require_path(args.input)
    scanner = CascadeScanner.from_artifacts(
        manifest_path=args.manifest,
    )
    try:
        assessment = await scanner.assess_chunks(
            _input_chunks(args.input),
            input_channel=args.input_channel,
        )
        return asdict(assessment)
    finally:
        await scanner.aclose()


async def _run_cascade_jsonl(args) -> None:
    # ponytail: records are scored serially to keep input order, pipeline
    # across records if remote review latency ever dominates batch runs.
    _require_path(args.jsonl)
    scanner = CascadeScanner.from_artifacts(
        manifest_path=args.manifest,
    )
    try:
        owned = args.jsonl != "-"
        handle = Path(args.jsonl).open(encoding="utf-8") if owned else sys.stdin
        try:
            # ponytail: the first bad record aborts the whole batch (fail
            # closed) after earlier results were already emitted; add a
            # skip-with-error-record mode if large mixed batches show up.
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    text = record["text"]
                    input_channel = record["input_channel"]
                    if not isinstance(text, str) or not text:
                        raise ValueError("text must be a non-empty string")
                    if input_channel not in ALLOWED_CHANNELS:
                        raise ValueError(
                            "input_channel must be direct_user or untrusted_content"
                        )
                    # Lone surrogates from json escapes fail here, inside the
                    # line-numbered handler, not inside the scanner.
                    text.encode()
                except (KeyError, TypeError, ValueError) as error:
                    raise SystemExit(
                        f"morgott cascade: bad JSONL record on line {number}: {error}"
                    ) from error
                # Scanner-internal failures propagate as tracebacks, matching
                # the single-input path, instead of blaming the record.
                assessment = await scanner.assess_text(
                    text,
                    input_channel=input_channel,
                )
                print(json.dumps(asdict(assessment), sort_keys=True), flush=True)
        finally:
            if owned:
                handle.close()
    finally:
        await scanner.aclose()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "data":
        args.data_dir.mkdir(parents=True, exist_ok=True)
        canonical_manifest = args.data_dir / "manifest.json"
        if args.routing_only:
            result = rebuild_routing(args.data_dir)
        else:
            result = build_corpus(args.data_dir)
        summary = {
            "manifest": str(canonical_manifest),
            "sources": len(result["source_outputs"]),
            "routing_rows": {
                name: output["rows"] for name, output in result["routing_views"].items()
            },
        }
    elif args.command == "benchmark":
        result = run_benchmark(args.data_dir, args.artifacts_dir, args.reports_dir)
        summary = {
            "direct_threshold": result["training"]["threshold"],
            "indirect_threshold": result["indirect_training"]["threshold"],
            "report": str(args.reports_dir / "baseline.json"),
        }
    elif args.command == "routing-baseline":
        result = run_routing_baseline(
            args.data_dir,
            args.artifacts_dir,
            args.reports_dir,
            epochs=args.epochs,
            max_per_source_label=args.max_per_source_label,
        )
        summary = {
            "artifact": str(args.artifacts_dir / "routing_baseline.joblib"),
            "report": str(args.reports_dir / "routing-baseline.json"),
            "selected_rows": result["selection"]["selected_rows"],
        }
    elif args.command == "demo":
        result = run_policy_ablation(args.reports_dir)
        summary = {
            "unauthorized_actions_committed": result["reference_monitor"][
                "unauthorized_actions_committed"
            ],
            "report": str(args.reports_dir / "policy_ablation.md"),
        }
    elif args.command == "cascade":
        if args.jsonl is not None:
            if args.input is not None or args.input_channel is not None:
                args.usage_error(
                    "--jsonl replaces the input argument and --input-channel"
                )
            try:
                asyncio.run(_run_cascade_jsonl(args))
            except BrokenPipeError:
                # The consumer closed the pipe (e.g. | head). Point stdout at
                # devnull so interpreter shutdown does not raise again, then
                # exit nonzero per the python docs SIGPIPE pattern.
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
                raise SystemExit(1) from None
            return
        if args.input is None or args.input_channel is None:
            args.usage_error("input and --input-channel are required without --jsonl")
        summary = asyncio.run(_run_cascade(args))
    elif args.command == "scan":
        summary = scan(args.text, args.model, args.channel)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
