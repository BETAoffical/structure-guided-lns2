from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_supplement import (  # noqa: E402
    audit_da2_supplement_design,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the outcome-blind DA2 StructPool supplement before initial PP."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stride_robustaction_structpool_da2_supplement_design.json",
    )
    parser.add_argument(
        "--archive",
        default="build/movingai-da2-source-v1/_archives/da2-map.zip",
    )
    parser.add_argument(
        "--map-root", default="build/movingai-da2-source-v1/maps-all"
    )
    parser.add_argument(
        "--output",
        default=(
            "build/stride-robustaction-structpool-da2-supplement-static-audit-v1"
        ),
    )
    args = parser.parse_args()
    report = audit_da2_supplement_design(
        config_path=args.config,
        archive=args.archive,
        map_root=args.map_root,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
