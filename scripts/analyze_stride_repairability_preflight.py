#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability_preflight import (  # noqa: E402
    analyze_repairability_load_confirmation,
    analyze_repairability_reserve_extension,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze outcome-blind STRIDE repairability load confirmation."
    )
    parser.add_argument(
        "--analysis",
        choices=("load-confirmation", "reserve-extension"),
        default="load-confirmation",
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    analyzer = (
        analyze_repairability_load_confirmation
        if arguments.analysis == "load-confirmation"
        else analyze_repairability_reserve_extension
    )
    report = analyzer(
        source_config=arguments.source_config,
        dataset=arguments.dataset,
        qualification=arguments.qualification,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
