#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_slotpool import evaluate_slotpool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered map-grouped SlotPool v1 offline evaluation."
    )
    parser.add_argument(
        "--config", default="configs/stride_slotpool_v1_registration.json"
    )
    parser.add_argument(
        "--output", default="build/stride-slotpool-v1-offline-evaluation"
    )
    arguments = parser.parse_args()
    report = evaluate_slotpool(config_path=arguments.config, output=arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["runtime_integration_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
