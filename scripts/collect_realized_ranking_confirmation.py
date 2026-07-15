from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.realized_ranking_confirmation import (  # noqa: E402
    CollectionLockError,
    run_confirmation_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect proposal-only and explicit-repair confirmation data."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default="configs/realized_ranking_confirmation_collection.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase", choices=("qualify", "propose", "evaluate", "all"), default="all"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    arguments = parser.parse_args()
    try:
        report = run_confirmation_collection(
            arguments.dataset,
            arguments.config,
            arguments.output,
            phase=arguments.phase,
            workers=arguments.workers,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            max_tasks=arguments.max_tasks,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
