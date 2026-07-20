from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.policy.policy_visited_aggregation_analysis import (  # noqa: E402
    run_final_policy_visited_analysis,
    run_offline_policy_visited_analysis,
    run_policy_visited_training,
    run_v2_closed_loop_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the InitLNS policy-visited aggregate ranker."
    )
    parser.add_argument(
        "--phase", choices=("train", "offline", "validate", "final"), required=True
    )
    parser.add_argument("--dataset")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--training")
    parser.add_argument("--offline")
    parser.add_argument("--v2-validation")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="research/configs/policy/policy_visited_analysis.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "train":
        result = run_policy_visited_training(
            arguments.collection, arguments.config, arguments.output
        )
    elif arguments.phase == "offline":
        if not arguments.training:
            parser.error("--training is required for offline")
        result = run_offline_policy_visited_analysis(
            arguments.collection,
            arguments.training,
            arguments.config,
            arguments.output,
        )
    elif arguments.phase == "validate":
        if not arguments.dataset or not arguments.training:
            parser.error("--dataset and --training are required for validate")
        result = run_v2_closed_loop_validation(
            arguments.dataset,
            arguments.collection,
            arguments.training,
            arguments.output,
            workers=arguments.workers,
            resume=arguments.resume,
        )
    else:
        if (
            not arguments.training
            or not arguments.offline
            or not arguments.v2_validation
        ):
            parser.error(
                "--training, --offline and --v2-validation are required for final"
            )
        result = run_final_policy_visited_analysis(
            arguments.collection,
            arguments.training,
            arguments.offline,
            arguments.v2_validation,
            arguments.config,
            arguments.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
