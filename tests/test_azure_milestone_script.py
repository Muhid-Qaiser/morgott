from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

SERVICES = [
    "Virtual Machines",
    "Storage",
    "Container Registry",
    "Azure Container Apps",
    "Service Bus",
]


def _run_milestone(
    tmp_path: Path,
    services: list[str],
    *,
    portal_confirmed: bool = False,
    extra_rows: list[list[object]] | None = None,
) -> tuple[str, str]:
    fake_az = tmp_path / "az"
    fake_az.write_text(
        """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
if args[:2] == ["account", "show"]:
    print("waleed@vulsight.com" if "user.name" in args else "25d0cf2e-a75c-46f5-b26c-f57a48f96967")
elif args[:1] == ["rest"]:
    print(os.environ["FAKE_COST_RESPONSE"])
elif args[:2] == ["group", "show"]:
    print("")
elif args[:2] == ["group", "update"]:
    with open(os.environ["FAKE_AZ_LOG"], "a", encoding="utf-8") as log:
        log.write(" ".join(args) + "\\n")
else:
    raise SystemExit(f"unexpected az command: {args}")
""",
        encoding="utf-8",
    )
    fake_az.chmod(0o755)
    usage_date = int(datetime.now(UTC).strftime("%Y%m%d"))
    response = {
        "properties": {
            "columns": [
                {"name": "PreTaxCost"},
                {"name": "UsageDate"},
                {"name": "ServiceName"},
                {"name": "Currency"},
            ],
            "rows": [
                *[[1.25, usage_date, service, "USD"] for service in services],
                *(extra_rows or []),
            ],
        }
    }
    log_path = tmp_path / "az.log"
    env = os.environ | {
        "FAKE_AZ_LOG": str(log_path),
        "FAKE_COST_RESPONSE": json.dumps(response),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        [
            "scripts/check-azure-milestone.sh",
            *(["--portal-confirmed"] if portal_confirmed else []),
        ],
        check=True,
        capture_output=True,
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
    )
    return result.stdout, log_path.read_text(
        encoding="utf-8"
    ) if log_path.exists() else ""


class AzureMilestoneScriptTests(unittest.TestCase):
    def test_canary_alert_catches_failures_and_missing_results(self) -> None:
        bicep = Path("infra/main.bicep").read_text(encoding="utf-8")

        self.assertIn('Log_s has "daily_canary_failed"', bicep)
        self.assertIn('Log_s has "daily_canary_complete"', bicep)
        self.assertIn("TimeGenerated > ago(26h)", bicep)

    def test_tracks_only_the_five_intended_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, az_log = _run_milestone(
                Path(directory), SERVICES + ["Bandwidth"], portal_confirmed=True
            )

        self.assertIn("All five intended workloads crossed USD 1", output)
        self.assertNotIn("Bandwidth", output)
        self.assertIn("--set tags.mfs25kDayZero=", az_log)

    def test_unrelated_service_does_not_complete_the_milestone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, az_log = _run_milestone(
                Path(directory), SERVICES[:-1] + ["Bandwidth"]
            )

        self.assertIn("Qualified intended workloads: 4/5", output)
        self.assertEqual(az_log, "")

    def test_portal_confirmation_is_required_before_saving_day_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, az_log = _run_milestone(Path(directory), SERVICES)

        self.assertIn("rerun with --portal-confirmed", output)
        self.assertEqual(az_log, "")

    def test_refund_resets_the_crossing_date(self) -> None:
        today = datetime.now(UTC).date()
        rows = [
            [
                1.2,
                int((today - timedelta(days=2)).strftime("%Y%m%d")),
                "Service Bus",
                "USD",
            ],
            [
                -0.5,
                int((today - timedelta(days=1)).strftime("%Y%m%d")),
                "Service Bus",
                "USD",
            ],
            [0.5, int(today.strftime("%Y%m%d")), "Service Bus", "USD"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            output, _ = _run_milestone(Path(directory), SERVICES[:-1], extra_rows=rows)

        self.assertIn(f"Cost data suggests day zero {today.isoformat()}", output)
