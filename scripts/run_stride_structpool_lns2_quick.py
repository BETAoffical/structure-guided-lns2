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

from experiments.stride_structpool_lns2_quick import (  # noqa: E402
    analyze_structpool_lns2_quick,
    run_structpool_lns2_quick,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired official-LNS2/V2/StructPool raw-TTF Quick."
    )
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "analyze":
        report = analyze_structpool_lns2_quick(arguments.config, arguments.output)
    else:
        report = run_structpool_lns2_quick(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if arguments.dry_run or report.get("integrity_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
