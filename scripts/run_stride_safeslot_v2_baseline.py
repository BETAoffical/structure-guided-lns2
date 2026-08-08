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

from experiments.stride_safeslot_v2_baseline import (  # noqa: E402
    analyze_safeslot_v2_baseline,
    run_safeslot_v2_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect the exact-runtime V2 trajectory baseline for SafeSlot."
    )
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "analyze":
        report = analyze_safeslot_v2_baseline(arguments.config, arguments.output)
    else:
        report = run_safeslot_v2_baseline(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if arguments.dry_run or report.get("integrity_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
