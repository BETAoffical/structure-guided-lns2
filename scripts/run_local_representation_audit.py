from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.local_representation_audit import (  # noqa: E402
    run_local_representation_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered InitLNS local representation audit."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-outcomes", type=int, default=7344)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when the pre-registered research gates fail.",
    )
    arguments = parser.parse_args()
    report = run_local_representation_audit(
        collection=arguments.collection,
        dataset=arguments.dataset,
        output=arguments.output,
        expected_outcomes=arguments.expected_outcomes,
        bootstrap_samples=arguments.bootstrap_samples,
        permutations=arguments.permutations,
    )
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 2 if arguments.strict and not report["acceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
