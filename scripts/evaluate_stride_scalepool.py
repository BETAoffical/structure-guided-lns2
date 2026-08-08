#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_scalepool_evaluation import evaluate_scalepool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate preregistered ScalePool current-step quality gates."
    )
    parser.add_argument(
        "--config", default="configs/stride_scalepool_v1_registration.json"
    )
    parser.add_argument(
        "--output", default="build/stride-scalepool-v1-offline-evaluation"
    )
    arguments = parser.parse_args()
    report = evaluate_scalepool(
        config_path=arguments.config,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["runtime_integration_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
