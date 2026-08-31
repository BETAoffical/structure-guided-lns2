from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_dual16_hierarchical_admission_h1_collection import (  # noqa: E402
    analyze_collection,
    run_collection,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the independent Dual16 hierarchical-admission H1 label-support "
            "experiment. This command does not train or register a runtime model."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "collect", "analyze"),
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/stride_dual16_hierarchical_admission_h1_v1.json",
    )
    parser.add_argument(
        "--output",
        default="build/stride-dual16-hierarchical-admission-h1-v1",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.phase == "preflight":
        if arguments.workers is not None:
            parser.error("--workers applies only to the collect phase")
        report = run_preflight(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
        )
    elif arguments.phase == "collect":
        report = run_collection(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            workers=arguments.workers,
        )
    else:
        if arguments.workers is not None or arguments.resume:
            parser.error("--workers and --resume do not apply to analyze")
        report = analyze_collection(arguments.config, arguments.output)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("integrity_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
