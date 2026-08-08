#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_slotpool_confirmation import (
    analyze_slotpool_confirmation,
    collect_slotpool_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen SlotPool fresh-map confirmation.")
    parser.add_argument("phase", choices=("collect", "analyze"))
    parser.add_argument("--config", default="configs/stride_slotpool_fresh_confirmation_v1.json")
    parser.add_argument("--collection", default="build/stride-slotpool-fresh-confirmation-collection-v1")
    parser.add_argument("--output", default="build/stride-slotpool-fresh-confirmation-analysis-v1")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "collect":
        report = collect_slotpool_confirmation(
            config_path=arguments.config,
            output=arguments.collection,
            resume=not arguments.no_resume,
        )
        success = bool(report["complete"])
    else:
        report = analyze_slotpool_confirmation(
            config_path=arguments.config,
            collection=arguments.collection,
            output=arguments.output,
        )
        success = bool(report["runtime_integration_allowed"])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
