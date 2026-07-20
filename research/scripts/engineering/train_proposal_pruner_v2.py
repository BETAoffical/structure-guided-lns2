from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.engineering.proposal_pruner.proposal_pruner_training import run_proposal_pruner_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train, calibrate, validate, and export the feature-v2 proposal pruner."
        )
    )
    parser.add_argument(
        "--config", default="research/configs/engineering/proposal_pruner_v2.json"
    )
    parser.add_argument(
        "--output", default="artifacts/initlns-closed-loop-controller-v2"
    )
    arguments = parser.parse_args()
    config = Path(arguments.config)
    output = Path(arguments.output)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    report = run_proposal_pruner_training(config, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
