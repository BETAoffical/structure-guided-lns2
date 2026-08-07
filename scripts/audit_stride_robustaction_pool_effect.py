from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_pool_effect import (  # noqa: E402
    audit_robustaction_pool_effect,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen V2 over its exact base pool with the same V2 over "
            "the base-plus-StructPool candidates using paired current-step outcomes."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stride_robustaction_structpool_pool_effect.json",
    )
    parser.add_argument(
        "--output", default="build/stride-robustaction-pool-effect-v1"
    )
    args = parser.parse_args()
    report = audit_robustaction_pool_effect(
        config_path=args.config,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
