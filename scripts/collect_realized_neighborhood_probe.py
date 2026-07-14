from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.realized_neighborhood_probe import (  # noqa: E402
    CollectionLockError,
    run_realized_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay realized neighborhoods with independent PP-order seeds."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-collection", required=True)
    parser.add_argument(
        "--config", default="configs/realized_neighborhood_stability_probe.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        report = run_realized_collection(
            arguments.dataset,
            arguments.source_collection,
            arguments.config,
            arguments.output,
            workers=arguments.workers,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if int(report.get("error_count", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
