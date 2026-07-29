from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_oracle_batch import run_stall_oracle_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a resumable full-pool exact stall Oracle plan."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_stall_oracle_plan(
        arguments.plan,
        arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
