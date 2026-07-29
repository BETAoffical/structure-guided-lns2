from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_oracle_batch import (  # noqa: E402
    report_existing_stall_oracle_jobs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and report an existing exact stall Oracle batch."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--jobs-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = report_existing_stall_oracle_jobs(
        arguments.plan,
        arguments.jobs_source,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
