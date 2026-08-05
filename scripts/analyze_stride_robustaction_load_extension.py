from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_load_extension import (  # noqa: E402
    analyze_load_extension_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze reset-only RobustAction load-extension qualification."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_load_extension_qualification_design.json"
        ),
    )
    parser.add_argument(
        "--extension-dataset",
        default="build/stride-robustaction-structpool-load-extension-dataset-v2",
    )
    parser.add_argument("--extension-qualification", required=True)
    parser.add_argument(
        "--v1-dataset",
        default="build/stride-robustaction-structpool-preflight-dataset-v2",
    )
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-structpool-source-dataset-v2",
    )
    arguments = parser.parse_args()
    report = analyze_load_extension_qualification(
        config_path=arguments.config,
        extension_dataset=arguments.extension_dataset,
        extension_qualification=arguments.extension_qualification,
        v1_dataset=arguments.v1_dataset,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
