from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.policy.ranking_objective_audit import run_ranking_objective_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether the InitLNS ranking objective is aligned with top-1 selection."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument(
        "--config", default="research/configs/policy/ranking_objective_audit.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("diagnose", "cross_validate", "validate", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    result = run_ranking_objective_audit(
        arguments.collection,
        arguments.training,
        arguments.offline,
        arguments.config,
        arguments.output,
        phase=arguments.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
