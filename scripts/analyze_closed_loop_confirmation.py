from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closed_loop_confirmation_analysis import (  # noqa: E402
    run_closed_loop_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze frozen InitLNS neighborhood-ranker closed-loop episodes."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--config", default="configs/closed_loop_confirmation_analysis.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    report = run_closed_loop_analysis(
        arguments.collection, arguments.config, arguments.output
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if arguments.strict and not report["acceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
