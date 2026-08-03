#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability import build_repairability_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build STRIDE repairability labels from paired 16-seed trials."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", action="append", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = build_repairability_labels(
        config_path=arguments.config,
        trial_paths=[Path(value) for value in arguments.trials],
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
