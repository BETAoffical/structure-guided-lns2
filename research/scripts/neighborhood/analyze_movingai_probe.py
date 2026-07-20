from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.neighborhood.movingai_mechanism_probe import analyze_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the pre-registered MovingAI InitLNS mechanism probe."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--config", default="research/configs/neighborhood/movingai_mechanism_probe_dataset.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze a smoke collection without enforcing registered volume.",
    )
    arguments = parser.parse_args()
    report = analyze_probe(
        arguments.collection,
        arguments.config,
        arguments.output,
        require_complete=not arguments.allow_partial,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gates": report["gates"],
                "output": str(Path(arguments.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["integrity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
