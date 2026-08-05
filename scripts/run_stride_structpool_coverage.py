#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_structpool_coverage import collect_structpool_coverage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the proposal-only STRIDE StructPool v1 coverage audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = collect_structpool_coverage(arguments.config, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
