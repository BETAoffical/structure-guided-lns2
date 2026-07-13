from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.stage5 import run_stage5_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy Stage 5 v1 repair-role experiments. "
            "Use run_stage5_v4_experiment.py for the current workflow."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), required=True
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--thresholds", default="0.5,0.67,0.8")
    parser.add_argument("--config")
    parser.add_argument("--neighborhood", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--time-limit-ms", type=int, default=5000)
    arguments = parser.parse_args()
    seeds = [
        int(value.strip())
        for value in arguments.seeds.split(",")
        if value.strip()
    ]
    thresholds = [
        float(value.strip())
        for value in arguments.thresholds.split(",")
        if value.strip()
    ]
    summary = run_stage5_experiment(
        dataset=arguments.dataset,
        solver=arguments.solver,
        index=arguments.index,
        evaluation=arguments.evaluation,
        output=arguments.output,
        split=arguments.split,
        seeds=seeds,
        thresholds=(
            thresholds if arguments.split == "validation" else None
        ),
        config=arguments.config,
        neighborhood=arguments.neighborhood,
        iterations=arguments.iterations,
        time_limit_ms=arguments.time_limit_ms,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
