from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.context_confirmation import run_secondary_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered InitLNS context secondary audit."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument(
        "--reference-dataset",
        help="Pilot v2 root used to reject map-seed overlap in confirmation mode.",
    )
    arguments = parser.parse_args()
    report = run_secondary_audit(
        arguments.collection,
        arguments.output,
        arguments.dataset,
        arguments.mode,
        arguments.permutations,
        arguments.reference_dataset,
    )
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
