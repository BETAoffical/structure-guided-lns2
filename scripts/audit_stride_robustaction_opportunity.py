from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_opportunity import (  # noqa: E402
    audit_robustaction_opportunity,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen-V2 anchored RobustAction opportunity and paired-seed "
            "action uncertainty without reading runtime or TTF."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stride_robustaction_structpool_opportunity_audit.json",
    )
    parser.add_argument(
        "--output", default="build/stride-robustaction-opportunity-audit-v1"
    )
    args = parser.parse_args()
    report = audit_robustaction_opportunity(
        config_path=args.config,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # A completed scientific audit is operationally successful even when its
    # preregistered opportunity gate rejects the model line.
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
