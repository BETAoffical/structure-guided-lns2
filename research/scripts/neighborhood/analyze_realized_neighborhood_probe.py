from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.neighborhood.realized_neighborhood_probe import (  # noqa: E402
    analyze_realized_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze explicit realized-neighborhood repair stability."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--config", default="research/configs/neighborhood/realized_neighborhood_stability_probe.json"
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze_realized_collection(
        arguments.collection, arguments.config, arguments.output
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "passed": report["passed"],
                "output": str(Path(arguments.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
