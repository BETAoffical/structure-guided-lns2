from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_escape_qualification import (  # noqa: E402
    prepare_stall_escape_qualification,
)


def _thresholds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(map(int, value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("thresholds must not be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only, map-isolated v2 stall qualification pool and "
            "an exact missing-PP-branch plan without training a controller."
        )
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", type=_thresholds, default=(3, 4, 6))
    parser.add_argument("--future-decisions", type=int, default=3)
    parser.add_argument("--minimum-distinct-attempts", type=int, default=2)
    parser.add_argument("--maximum-per-class-per-split", type=int, default=48)
    parser.add_argument("--required-trials-per-candidate", type=int, default=4)
    arguments = parser.parse_args()
    report = prepare_stall_escape_qualification(
        arguments.source,
        arguments.output,
        oracle_root=arguments.oracle,
        thresholds=arguments.thresholds,
        future_observation_decisions=arguments.future_decisions,
        minimum_distinct_attempts=arguments.minimum_distinct_attempts,
        maximum_per_class_per_split=arguments.maximum_per_class_per_split,
        required_trials_per_candidate=arguments.required_trials_per_candidate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
