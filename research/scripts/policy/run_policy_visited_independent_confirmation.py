from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.policy.policy_visited_independent_confirmation import (  # noqa: E402
    run_independent_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the independent natural-distribution policy confirmation."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--development-collection", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument(
        "--config", default="research/configs/policy/policy_visited_independent_confirmation.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase", choices=("qualify", "collect", "analyze", "all"), default="all"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = run_independent_confirmation(
        arguments.dataset,
        arguments.development_collection,
        arguments.training,
        arguments.offline,
        arguments.config,
        arguments.output,
        phase=arguments.phase,
        workers=arguments.workers,
        resume=arguments.resume,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
