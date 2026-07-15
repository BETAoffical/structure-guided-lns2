from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.realized_ranking_confirmation_analysis import (  # noqa: E402
    freeze_confirmation_models,
    run_confirmation_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze development rankers or evaluate independent confirmation data."
    )
    parser.add_argument("--mode", choices=("freeze", "analyze"), required=True)
    parser.add_argument(
        "--config", default="configs/realized_ranking_confirmation_analysis.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection")
    parser.add_argument("--frozen-models")
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "freeze":
        report = freeze_confirmation_models(arguments.config, arguments.output)
        passed = True
    else:
        if not arguments.collection or not arguments.frozen_models:
            parser.error("--collection and --frozen-models are required for analyze")
        report = run_confirmation_analysis(
            arguments.collection,
            arguments.config,
            arguments.frozen_models,
            arguments.output,
        )
        passed = bool(report["acceptance"]["passed"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if arguments.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
