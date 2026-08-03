#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_maprank import run_maprank_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the map-diverse V2-anchored STRIDE-MapRank selector."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run_maprank_training(
        config_path=arguments.config,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["fresh_development_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
