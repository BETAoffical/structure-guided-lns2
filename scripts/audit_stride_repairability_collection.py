#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability_audit import (  # noqa: E402
    audit_repairability_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the complete STRIDE 16-seed repairability collection."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = audit_repairability_collection(
        collection=arguments.collection,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
