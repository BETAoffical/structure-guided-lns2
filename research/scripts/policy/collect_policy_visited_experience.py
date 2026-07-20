from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.policy.policy_visited_aggregation import (  # noqa: E402
    CollectionLockError,
    run_policy_visited_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect counterfactual labels at frozen-policy visited InitLNS states."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default="research/configs/policy/policy_visited_collection.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("qualify", "source", "propose", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    arguments = parser.parse_args()
    try:
        report = run_policy_visited_collection(
            arguments.dataset,
            arguments.config,
            arguments.output,
            phase=arguments.phase,
            workers=arguments.workers,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            task_ids=arguments.task_ids,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
