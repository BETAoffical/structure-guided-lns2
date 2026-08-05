from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_map_replacement import (  # noqa: E402
    analyze_map_replacement_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze reset-only RobustAction map-replacement qualification."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_map_replacement_qualification_design.json"
        ),
    )
    parser.add_argument(
        "--replacement-dataset",
        default="build/stride-robustaction-structpool-map-replacement-dataset-v3",
    )
    parser.add_argument("--replacement-qualification", required=True)
    parser.add_argument(
        "--v1-dataset",
        default="build/stride-robustaction-structpool-preflight-dataset-v2",
    )
    parser.add_argument(
        "--v2-extension-dataset",
        default="build/stride-robustaction-structpool-load-extension-dataset-v2",
    )
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-structpool-source-dataset-v3",
    )
    arguments = parser.parse_args()
    report = analyze_map_replacement_qualification(
        config_path=arguments.config,
        replacement_dataset=arguments.replacement_dataset,
        replacement_qualification=arguments.replacement_qualification,
        v1_dataset=arguments.v1_dataset,
        v2_extension_dataset=arguments.v2_extension_dataset,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
