from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_dataset
from .detector import run_benchmark, scan
from .policy import run_policy_ablation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morgott")
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data", help="download and consolidate pinned data")
    data.add_argument("--data-dir", type=Path, default=Path("data"))

    benchmark = subcommands.add_parser("benchmark", help="train and evaluate baselines")
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))
    benchmark.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--reports-dir", type=Path, default=Path("reports"))

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
        result = build_dataset(args.data_dir)
        summary = result["outputs"]
    elif args.command == "benchmark":
        result = run_benchmark(args.data_dir, args.artifacts_dir, args.reports_dir)
        summary = {
            "direct_threshold": result["training"]["threshold"],
            "indirect_threshold": result["indirect_training"]["threshold"],
            "report": str(args.reports_dir / "baseline.md"),
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
