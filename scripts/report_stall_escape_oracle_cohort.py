from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_escape_oracle_cohort import (  # noqa: E402
    build_stall_escape_oracle_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine exact v2 stall Oracles and frozen-v3 risk audits."
    )
    parser.add_argument("--oracle", action="append", required=True)
    parser.add_argument("--risk", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = build_stall_escape_oracle_cohort(
        arguments.oracle,
        arguments.risk,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
