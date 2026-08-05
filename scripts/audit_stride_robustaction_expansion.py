from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_expansion import (  # noqa: E402
    audit_robustaction_expansion,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the outcome-blind STRIDE robust-action map expansion."
    )
    parser.add_argument(
        "--config", default="configs/stride_robustaction_expansion_design.json"
    )
    parser.add_argument(
        "--archive",
        default="build/movingai-dao-compact-source-v1/_archives/dao-map.zip",
    )
    parser.add_argument(
        "--maps", default="build/movingai-dao-compact-source-v1/maps-all"
    )
    parser.add_argument(
        "--output", default="build/stride-robustaction-expansion-design-v1"
    )
    args = parser.parse_args()
    report = audit_robustaction_expansion(
        config_path=args.config,
        archive=args.archive,
        map_root=args.maps,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
