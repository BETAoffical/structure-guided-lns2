#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_platform_action_robustness import (  # noqa: E402
    analyze_action_robustness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cross-seed platform robustness without running PP."
    )
    parser.add_argument(
        "--frontier-report",
        default=(
            "build/stride-platformentry-frontier-v1-r2/"
            "platformentry_frontier_report.json"
        ),
    )
    parser.add_argument(
        "--witnesses",
        default=(
            "build/stride-repairdependencypool-v1/"
            "platform_entry_witnesses.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "build/stride-platformentry-frontier-v1-r2/"
            "platform_action_robustness_report.json"
        ),
    )
    parser.add_argument("--summary-only", action="store_true")
    arguments = parser.parse_args()
    report = analyze_action_robustness(
        arguments.frontier_report, arguments.witnesses, arguments.output
    )
    printable = report
    if arguments.summary_only:
        printable = {
            key: value
            for key, value in report.items()
            if key
            not in {
                "cases",
            }
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
