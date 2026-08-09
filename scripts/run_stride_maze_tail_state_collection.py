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

from experiments.stride_maze_tail_state_collection import (  # noqa: E402
    analyze_maze_tail_state_collection,
    run_maze_tail_state_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered fused Maze tail state collection."
    )
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument(
        "--config",
        default="configs/stride_maze_tail_state_collection_v2.json",
    )
    parser.add_argument(
        "--output",
        default="build/stride-maze-tail-state-collection-v2",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "analyze":
        report = analyze_maze_tail_state_collection(arguments.config, arguments.output)
    else:
        report = run_maze_tail_state_collection(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if arguments.dry_run or report.get("integrity_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
