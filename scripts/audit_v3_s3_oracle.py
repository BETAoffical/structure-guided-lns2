#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.v3_s3_oracle_audit import audit_v3_s3_oracle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit observed S3 Oracle headroom, model-selection regret, and "
            "local-versus-closed-loop objective mismatch without running the solver."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runtime-report",
        action="append",
        default=[],
        help=(
            "Optional v3-S3 runtime comparison report. Repeat this option to "
            "include independent timing repeats."
        ),
    )
    arguments = parser.parse_args()
    report = audit_v3_s3_oracle(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        runtime_reports=[
            resolve_cli_path(PROJECT_ROOT, value)
            for value in arguments.runtime_report
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
