#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_paretopool_audit import run_paretopool_gap_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ParetoPool candidate gaps.")
    parser.add_argument(
        "--config",
        default="configs/stride_paretopool_gap_audit_v1_registration.json",
    )
    parser.add_argument(
        "--output", default="build/stride-paretopool-gap-audit-v1"
    )
    arguments = parser.parse_args()
    report = run_paretopool_gap_audit(arguments.config, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
