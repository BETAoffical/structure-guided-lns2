from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_recovery import (  # noqa: E402
    MODES,
    prepare_da2_recovery_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a preregistered DA2 recovery task ladder without PP."
    )
    parser.add_argument(
        "--config",
        default="configs/stride_robustaction_structpool_da2_recovery_design.json",
    )
    parser.add_argument("--mode", choices=MODES, default="primary_repair")
    parser.add_argument(
        "--fetched", default="build/movingai-da2-source-v1"
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = prepare_da2_recovery_dataset(
        config_path=arguments.config,
        mode=arguments.mode,
        fetched=arguments.fetched,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
