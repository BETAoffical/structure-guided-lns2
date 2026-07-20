from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.neighborhood.realized_neighborhood_ranking_audit import (  # noqa: E402
    run_realized_neighborhood_ranking_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit learned ranking of explicit InitLNS agent neighborhoods."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--config", default="research/configs/neighborhood/realized_neighborhood_ranking_audit.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when the registered realized-ranking gate fails.",
    )
    arguments = parser.parse_args()
    report = run_realized_neighborhood_ranking_audit(
        arguments.collection, arguments.config, arguments.output
    )
    print(
        json.dumps(
            {
                "decision": report["acceptance"]["decision"],
                "ranking_passed": report["acceptance"]["passed"],
                "static_context_passed": report["acceptance"][
                    "static_transfer_passed"
                ],
                "timings_seconds": report["timings_seconds"],
                "output": str(Path(arguments.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if arguments.strict and not report["acceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
