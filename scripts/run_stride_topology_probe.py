#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_topology_probe import (  # noqa: E402
    analyze_topology_probe,
    extract_topology_probe_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the consumed STRIDE topology-interaction diagnostic."
    )
    parser.add_argument("--mode", choices=("extract", "analyze"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extraction")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.mode == "extract":
        if arguments.extraction:
            parser.error("--extraction is only valid in analyze mode")
        report = extract_topology_probe_features(arguments.config, arguments.output)
    else:
        if not arguments.extraction:
            parser.error("analyze mode requires --extraction")
        report = analyze_topology_probe(
            arguments.config, arguments.extraction, arguments.output
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
