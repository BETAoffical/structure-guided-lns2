#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_topology_anchor_quality import (  # noqa: E402
    analyze_topology_anchor_quality_pilot,
    collect_topology_anchor_quality_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired immediate-quality STRIDE topology-anchor Pilot."
    )
    parser.add_argument("--mode", choices=("collect", "analyze"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--collection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "collect":
        if arguments.collection:
            parser.error("--collection is only valid in analyze mode")
        report = collect_topology_anchor_quality_pilot(
            arguments.config, arguments.output, resume=not arguments.no_resume
        )
    else:
        if not arguments.collection:
            parser.error("analyze mode requires --collection")
        report = analyze_topology_anchor_quality_pilot(
            arguments.config, arguments.collection, arguments.output
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
