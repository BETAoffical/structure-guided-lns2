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
    analyze_multivalue_collection,
    prepare_multivalue_pilot,
    run_multivalue_collection,
    run_worker_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STRIDE MultiValue stages.")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "preflight",
            "collect-initial",
            "analyze-initial",
            "collect-extension",
            "analyze-extended",
        ),
    )
    parser.add_argument(
        "--config", default="configs/stride_multivalue_pilot_v1_registration.json"
    )
    parser.add_argument(
        "--output", default="build/stride-multivalue-pilot-v1"
    )
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        report = prepare_multivalue_pilot(arguments.config, arguments.output)
    elif arguments.command == "preflight":
        report = run_worker_preflight(arguments.config, arguments.output)
    elif arguments.command == "collect-initial":
        report = run_multivalue_collection(
            arguments.config, arguments.output, phase="initial"
        )
    elif arguments.command == "analyze-initial":
        report = analyze_multivalue_collection(
            arguments.config, arguments.output, include_extension=False
        )
    elif arguments.command == "collect-extension":
        report = run_multivalue_collection(
            arguments.config, arguments.output, phase="extension"
        )
    else:
        report = analyze_multivalue_collection(
            arguments.config, arguments.output, include_extension=True
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report.get("status") in {"complete", "not_required"}:
        return 0
    return 0 if report.get("integrity_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
