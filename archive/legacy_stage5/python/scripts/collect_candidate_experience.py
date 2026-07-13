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
            "Collect Trace V4/V5/V6 counterfactual neighborhood trials for "
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
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument(
        "--candidate-generator-profile",
        choices=("core5", "full8"),
        default="core5",
    )
    parser.add_argument(
        "--candidate-trial-limit-ms", type=int, default=2000
    )
    parser.add_argument(
        "--candidate-replan-order-seeds",
        default="0,1",
        help="Comma-separated deterministic order seeds for each candidate.",
    )
    parser.add_argument(
        "--candidate-rollout-horizons",
        default="10,25",
        help="Optional comma-separated closed-loop rollout horizons.",
    )
    parser.add_argument(
        "--layout-modes",
        default="",
        help="Optional comma-separated layout modes to collect.",
    )
    parser.add_argument(
        "--task-variants",
        default="",
        help="Optional comma-separated task variants to collect.",
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--workers", type=int, default=1)
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
    rollout_horizons = [
        int(value.strip())
        for value in arguments.candidate_rollout_horizons.split(",")
        if value.strip()
    ]
    layout_modes = [
        value.strip()
        for value in arguments.layout_modes.split(",")
        if value.strip()
    ]
    task_variants = [
        value.strip()
        for value in arguments.task_variants.split(",")
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
        candidate_generator_profile=arguments.candidate_generator_profile,
        candidate_trial_limit_ms=arguments.candidate_trial_limit_ms,
        candidate_replan_order_seeds=replan_order_seeds,
        candidate_rollout_horizons=rollout_horizons,
        layout_modes=layout_modes or None,
        task_variants=task_variants or None,
        max_runs=arguments.max_runs,
        workers=arguments.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
