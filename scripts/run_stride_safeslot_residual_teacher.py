from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_safeslot_residual_teacher import (  # noqa: E402
    analyze_safeslot_residual_teacher,
    collect_safeslot_residual_teacher,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SafeSlot residual teacher work")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument(
        "--config", default="configs/stride_safeslot_residual_teacher_v1.json"
    )
    collect.add_argument(
        "--output", default="build/stride-safeslot-residual-teacher-v1"
    )
    collect.add_argument("--workers", type=int, default=2)
    collect.add_argument("--resume", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--config", default="configs/stride_safeslot_residual_teacher_v1.json"
    )
    analyze.add_argument(
        "--collection", default="build/stride-safeslot-residual-teacher-v1"
    )
    analyze.add_argument(
        "--output", default="build/stride-safeslot-residual-teacher-analysis-v1"
    )
    args = parser.parse_args()
    if args.command == "collect":
        report = collect_safeslot_residual_teacher(
            args.config, args.output, workers=args.workers, resume=args.resume
        )
        exit_code = 0 if report["passed"] else 1
    else:
        report = analyze_safeslot_residual_teacher(
            args.config, args.collection, args.output
        )
        exit_code = 0 if report["integrity_passed"] else 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
