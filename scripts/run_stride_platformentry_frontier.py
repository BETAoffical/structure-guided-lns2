#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir() and str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stride_platformentry_frontier import (  # noqa: E402
    analyze_frontier_collection,
    run_frontier_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered native-order frontier platform-entry test."
    )
    parser.add_argument(
        "command",
        choices=(
            "collect-initial",
            "analyze-initial",
            "collect-extension",
            "analyze-extended",
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/stride_platformentry_frontier_v1_registration.json",
    )
    parser.add_argument(
        "--output",
        default="build/stride-platformentry-frontier-v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-cases", type=int)
    arguments = parser.parse_args()
    if arguments.command == "collect-initial":
        report = run_frontier_collection(
            arguments.config,
            arguments.output,
            phase="initial",
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            limit_cases=arguments.limit_cases,
        )
    elif arguments.command == "collect-extension":
        report = run_frontier_collection(
            arguments.config,
            arguments.output,
            phase="extension",
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            limit_cases=arguments.limit_cases,
        )
    elif arguments.command == "analyze-initial":
        report = analyze_frontier_collection(
            arguments.config,
            arguments.output,
            phase="initial",
            expected_cases=arguments.limit_cases or 45,
        )
    else:
        report = analyze_frontier_collection(
            arguments.config,
            arguments.output,
            phase="extended",
            expected_cases=arguments.limit_cases or 45,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if arguments.dry_run:
        return 0
    if arguments.command.startswith("analyze"):
        return 0 if report.get("integrity_passed", False) else 1
    return 0 if not report.get("terminal_failure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
