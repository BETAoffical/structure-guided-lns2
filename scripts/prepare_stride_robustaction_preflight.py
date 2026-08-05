from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_expansion import (  # noqa: E402
    prepare_robustaction_preflight_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare STRIDE robust-action map/OD tasks without PP."
    )
    parser.add_argument(
        "--config", default="configs/stride_robustaction_expansion_design.json"
    )
    parser.add_argument(
        "--fetched", default="build/movingai-dao-compact-source-v1"
    )
    parser.add_argument(
        "--output", default="build/stride-robustaction-preflight-dataset-v1"
    )
    args = parser.parse_args()
    report = prepare_robustaction_preflight_dataset(
        config_path=args.config,
        fetched=args.fetched,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
