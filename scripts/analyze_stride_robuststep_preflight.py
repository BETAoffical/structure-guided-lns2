from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robuststep_preflight import (  # noqa: E402
    analyze_robuststep_map_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the outcome-blind STRIDE robust-step map/load preflight."
    )
    parser.add_argument("--source-config", default="configs/stride_robuststep_map_preflight_source.json")
    parser.add_argument("--dataset", default="build/stride-robuststep-movingai-dataset-v1")
    parser.add_argument("--qualification", default="build/stride-robuststep-map-preflight-v1")
    parser.add_argument("--output", default="build/stride-robuststep-map-preflight-analysis-v1")
    arguments = parser.parse_args()
    report = analyze_robuststep_map_preflight(
        source_config=arguments.source_config,
        dataset=arguments.dataset,
        qualification=arguments.qualification,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
