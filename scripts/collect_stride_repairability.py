#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability_collection import (  # noqa: E402
    collect_repairability_trials,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect topology-augmented paired repairability trials."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = collect_repairability_trials(
        config_path=arguments.config,
        selection_path=arguments.selection,
        output=arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
