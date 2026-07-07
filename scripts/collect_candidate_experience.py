from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.experience import collect_experience  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Trace V4/V5 counterfactual neighborhood trials for "
            "Train or Validation."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation"), required=True
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--neighborhood", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--time-limit-ms", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--candidate-trial-limit-ms", type=int, default=2000
    )
    parser.add_argument(
        "--candidate-replan-order-seeds",
        default="0,1,2",
        help="Comma-separated deterministic order seeds for each candidate.",
    )
    arguments = parser.parse_args()
    seeds = [
        int(value.strip())
        for value in arguments.seeds.split(",")
        if value.strip()
    ]
    replan_order_seeds = [
        int(value.strip())
        for value in arguments.candidate_replan_order_seeds.split(",")
        if value.strip()
    ]
    summary = collect_experience(
        dataset=arguments.dataset,
        solver=arguments.solver,
        output=arguments.output,
        split=arguments.split,
        seeds=seeds,
        neighborhood=arguments.neighborhood,
        iterations=arguments.iterations,
        time_limit_ms=arguments.time_limit_ms,
        candidate_trials=True,
        candidate_count=arguments.candidate_count,
        candidate_trial_limit_ms=arguments.candidate_trial_limit_ms,
        candidate_replan_order_seeds=replan_order_seeds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
