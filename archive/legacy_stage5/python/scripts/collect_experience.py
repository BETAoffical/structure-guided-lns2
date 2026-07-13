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
        description="Collect raw LNS iteration traces for a dataset split."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--neighborhood", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--time-limit-ms", type=int, default=3000)
    arguments = parser.parse_args()

    summary = collect_experience(
        dataset=arguments.dataset,
        solver=arguments.solver,
        output=arguments.output,
        split=arguments.split,
        seed=arguments.seed if arguments.seed is not None else 1234,
        seeds=(
            None
            if arguments.seed is not None
            else [
                int(value.strip())
                for value in arguments.seeds.split(",")
                if value.strip()
            ]
        ),
        neighborhood=arguments.neighborhood,
        iterations=arguments.iterations,
        time_limit_ms=arguments.time_limit_ms,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
