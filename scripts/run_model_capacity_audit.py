from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.model_capacity_audit import run_model_capacity_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether flat realized-neighborhood features need more GBDT capacity."
    )
    parser.add_argument("--training", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument("--objective-audit", required=True)
    parser.add_argument("--config", default="configs/model_capacity_audit.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase", choices=("cross_validate", "validate", "all"), default="all"
    )
    arguments = parser.parse_args()
    result = run_model_capacity_audit(
        arguments.training,
        arguments.offline,
        arguments.objective_audit,
        arguments.config,
        arguments.output,
        phase=arguments.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
