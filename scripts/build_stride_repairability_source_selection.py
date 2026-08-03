#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability_selection import (  # noqa: E402
    build_repairability_source_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the result-blind 240-state repairability cohort."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = build_repairability_source_selection(
        config_path=arguments.config,
        source_root=arguments.source,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
