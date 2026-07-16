from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sequential_credit_audit import (  # noqa: E402
    CollectionLockError,
    run_sequential_credit_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Horizon-4 credit at frozen-v1 policy-visited InitLNS states."
    )
    parser.add_argument(
        "--config", default="configs/sequential_credit_audit.json"
    )
    parser.add_argument(
        "--output", default="build/initlns-sequential-credit-audit-v1"
    )
    parser.add_argument(
        "--phase",
        choices=("diagnose", "collect", "analyze", "all", "dry-run"),
        default="all",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-states", type=int)
    arguments = parser.parse_args()
    try:
        report = run_sequential_credit_audit(
            arguments.config,
            arguments.output,
            phase=arguments.phase,
            workers=arguments.workers,
            resume=arguments.resume,
            smoke_states=arguments.smoke_states,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
