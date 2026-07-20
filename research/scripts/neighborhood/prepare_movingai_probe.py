from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.neighborhood.movingai_mechanism_probe import prepare_probe_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic MovingAI instances for the InitLNS mechanism probe."
    )
    parser.add_argument("--dataset", required=True, help="Fetched MovingAI devset root")
    parser.add_argument(
        "--config", default="research/configs/neighborhood/movingai_mechanism_probe_dataset.json"
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summary = prepare_probe_dataset(
        arguments.dataset, arguments.config, arguments.output
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
