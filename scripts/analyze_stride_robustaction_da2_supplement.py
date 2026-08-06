from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_qualification import (  # noqa: E402
    analyze_da2_supplement_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze reset-only RobustAction DA2 supplement qualification."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_da2_supplement_qualification_design.json"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "build/stride-robustaction-structpool-da2-supplement-dataset-v1"
        ),
    )
    parser.add_argument("--qualification", required=True)
    parser.add_argument(
        "--output",
        default=(
            "build/stride-robustaction-structpool-da2-supplement-source-dataset-v1"
        ),
    )
    arguments = parser.parse_args()
    report = analyze_da2_supplement_qualification(
        config_path=arguments.config,
        dataset=arguments.dataset,
        qualification=arguments.qualification,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
