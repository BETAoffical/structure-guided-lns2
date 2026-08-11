#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir() and str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stride_multivalue_collection import (  # noqa: E402
    prepare_multivalue_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STRIDE MultiValue stages.")
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument(
        "--config", default="configs/stride_multivalue_pilot_v1_registration.json"
    )
    parser.add_argument(
        "--output", default="build/stride-multivalue-pilot-v1"
    )
    arguments = parser.parse_args()
    report = prepare_multivalue_pilot(arguments.config, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
