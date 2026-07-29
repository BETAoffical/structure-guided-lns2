#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stall_trigger_counterfactual import (  # noqa: E402
    run_counterfactual_job,
    run_stall_trigger_counterfactual,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-v2 continuation with one cached rescue candidate "
            "from the same exact stall-trigger state."
        )
    )
    parser.add_argument("--source-v2", required=True)
    parser.add_argument("--trigger-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.job_index is None:
        report = run_stall_trigger_counterfactual(
            arguments.source_v2,
            arguments.trigger_csv,
            arguments.output,
            trials=arguments.trials,
            horizon=arguments.horizon,
            workers=arguments.workers,
            resume=arguments.resume,
        )
    else:
        report = run_counterfactual_job(
            arguments.source_v2,
            arguments.trigger_csv,
            arguments.output,
            job_index=arguments.job_index,
            trials=arguments.trials,
            horizon=arguments.horizon,
            resume=arguments.resume,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
