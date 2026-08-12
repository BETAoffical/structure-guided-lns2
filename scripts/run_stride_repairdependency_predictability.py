from __future__ import annotations

import argparse
import json

from experiments.stride_repairdependency_predictability import (
    analyze_predictability,
    freeze_input_state_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered STRIDE RepairDependency predictability audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-input-manifest")
    freeze.add_argument("--collection", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--workers", type=int, default=16)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "freeze-input-manifest":
        result = freeze_input_state_manifest(
            arguments.collection, arguments.output, workers=arguments.workers
        )
    else:
        result = analyze_predictability(arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
