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

from experiments.stride_repairclosurepool import (  # noqa: E402
    analyze_repairclosure_trials,
    collect_repairclosure_trials,
    materialize_repairclosure_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STRIDE RepairClosurePool v1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--design", required=True)
    materialize.add_argument("--output", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--execution", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--mode", choices=("preflight", "full"), required=True)
    collect.add_argument("--workers", type=int, default=16)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--preflight-output")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--execution", required=True)
    analyze.add_argument("--collection", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--mode", choices=("preflight", "full"), required=True)

    arguments = parser.parse_args()
    if arguments.command == "materialize":
        result = materialize_repairclosure_cohort(arguments.design, arguments.output)
        passed = result.get("integrity_passed") is True
    elif arguments.command == "collect":
        result = collect_repairclosure_trials(
            execution_path=arguments.execution,
            output=arguments.output,
            mode=arguments.mode,
            workers=arguments.workers,
            resume=arguments.resume,
            preflight_output=arguments.preflight_output,
        )
        passed = (
            result.get("integrity_passed") is True
            if "integrity_passed" in result
            else result.get("status") == "complete"
        )
    else:
        result = analyze_repairclosure_trials(
            execution_path=arguments.execution,
            collection=arguments.collection,
            output=arguments.output,
            mode=arguments.mode,
        )
        passed = result.get("integrity_passed") is True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
