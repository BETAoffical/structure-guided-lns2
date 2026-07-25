#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_ROOT = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_ROOT.is_dir() and str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.v3_value_stability import (  # noqa: E402
    reanalyze_stability_followup,
    run_stability_followup,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add PP seeds only to unstable value-label states and extend only "
            "previously censored branches. No controller model is trained."
        )
    )
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-trials", type=int, default=4)
    parser.add_argument("--max-repairs", type=int, default=60)
    parser.add_argument("--wall-clock-seconds", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reanalyze-only", action="store_true")
    arguments = parser.parse_args()
    pilot = resolve_cli_path(PROJECT_ROOT, arguments.pilot)
    output = resolve_cli_path(PROJECT_ROOT, arguments.output)
    if arguments.reanalyze_only:
        report = reanalyze_stability_followup(
            pilot=pilot,
            output=output,
        )
    else:
        report = run_stability_followup(
            pilot=pilot,
            output=output,
            total_trials=arguments.total_trials,
            max_repairs=arguments.max_repairs,
            wall_clock_seconds=arguments.wall_clock_seconds,
            resume=arguments.resume,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
