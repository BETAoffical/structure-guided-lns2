from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_recovery_qualification import (  # noqa: E402
    analyze_da2_recovery_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the outcome-informed DA2 primary recovery resets."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_da2_recovery_primary_qualification_design.json"
        ),
    )
    parser.add_argument(
        "--recovery-dataset",
        default="build/stride-robustaction-da2-ca-caverns2-repair-dataset-v1",
    )
    parser.add_argument("--qualification", required=True)
    parser.add_argument(
        "--retained-dataset",
        default="build/stride-robustaction-structpool-da2-supplement-dataset-v1",
    )
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-da2-recovery-source-dataset-v1",
    )
    arguments = parser.parse_args()
    report = analyze_da2_recovery_qualification(
        config_path=arguments.config,
        recovery_dataset=arguments.recovery_dataset,
        qualification=arguments.qualification,
        retained_dataset=arguments.retained_dataset,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
