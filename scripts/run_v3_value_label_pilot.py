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
from experiments.v3_value_pilot import run_value_label_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a small variable-horizon cost-to-go label pilot from "
            "existing paired S3 states. The first action varies; subsequent "
            "repairs use official Adaptive as the explicitly recorded teacher."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--oracle-state-comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=30)
    parser.add_argument("--wall-clock-seconds", type=float, default=60.0)
    parser.add_argument("--split", default="policy_train")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_value_label_pilot(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        oracle_state_comparison=resolve_cli_path(
            PROJECT_ROOT, arguments.oracle_state_comparison
        ),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        state_count=arguments.state_count,
        trials=arguments.trials,
        max_repairs=arguments.max_repairs,
        wall_clock_seconds=arguments.wall_clock_seconds,
        split=arguments.split,
        smoke_only=arguments.smoke_only,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
