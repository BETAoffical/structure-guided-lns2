#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_topology_anchor_coverage import (  # noqa: E402
    collect_topology_anchor_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the proposal-only STRIDE topology-anchor coverage diagnostic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = collect_topology_anchor_coverage(arguments.config, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
