from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.feasibility_benchmark import run_benchmark  # noqa: E402


def _seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LNS2 repair and GPBS with common MovingAI settings."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--solver", choices=("lns2_repair", "gpbs", "both"), default="both"
    )
    parser.add_argument("--lns2-binary")
    parser.add_argument("--gpbs-binary")
    parser.add_argument("--seeds", type=_seeds, default=[0, 1])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    requested = (
        ["lns2_repair", "gpbs"]
        if arguments.solver == "both"
        else [arguments.solver]
    )
    binaries = {
        "lns2_repair": arguments.lns2_binary,
        "gpbs": arguments.gpbs_binary,
    }
    missing = [solver for solver in requested if not binaries[solver]]
    if missing:
        parser.error(f"missing binary path for: {', '.join(missing)}")
    rows = run_benchmark(
        arguments.dataset,
        arguments.output,
        {solver: binaries[solver] for solver in requested},
        arguments.seeds,
        arguments.time_limit,
        arguments.limit,
        arguments.resume,
    )
    print(json.dumps({"runs": len(rows), "output": arguments.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
