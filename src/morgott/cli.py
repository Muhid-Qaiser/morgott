from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .corpus import build_corpus, rebuild_routing
from .data import build_dataset
from .detector import run_benchmark, scan
from .policy import run_policy_ablation
from .routing_baseline import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_PER_SOURCE_LABEL,
    run_routing_baseline,
)


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

    scanner = subcommands.add_parser("scan", help="score one input in shadow mode")
    scanner.add_argument("text")
    scanner.add_argument(
        "--channel",
        choices=("direct_user", "untrusted_content"),
        default="direct_user",
        help="trusted input provenance selecting the channel-specific sensor",
    )
    scanner.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/guard_bundle.joblib"),
        help="trusted local artifact produced by the benchmark command",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "data":
        args.data_dir.mkdir(parents=True, exist_ok=True)
        canonical_manifest = args.data_dir / "manifest.json"
        if args.routing_only:
            result = rebuild_routing(args.data_dir)
        else:
            canonical_manifest.unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(
                dir=args.data_dir, prefix=".core-build-"
            ) as directory:
                core_manifest = Path(directory) / "manifest.json"
                build_dataset(args.data_dir, manifest_path=core_manifest)
                result = build_corpus(
                    args.data_dir,
                    core_manifest_path=core_manifest,
                )
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
            "report": str(args.reports_dir / "baseline.md"),
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
            "report": str(args.reports_dir / "routing-baseline.md"),
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
    else:
        summary = scan(args.text, args.model, args.channel)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
