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

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.stride_maze_tail_evidence import (  # noqa: E402
    analyze_maze_tail_evidence_qualification,
    prepare_maze_tail_evidence_dataset,
    run_maze_tail_evidence_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run the preregistered reset-only Maze long-tail "
            "evidence preflight."
        )
    )
    parser.add_argument("command", choices=("prepare", "qualify", "analyze"))
    parser.add_argument(
        "--config",
        default="configs/stride_maze_tail_evidence_preflight_v1.json",
    )
    parser.add_argument(
        "--fetched",
        default="build/stride-maze-tail-evidence-raw-v1",
    )
    parser.add_argument(
        "--dataset",
        default="build/stride-maze-tail-evidence-candidate-dataset-v1",
    )
    parser.add_argument(
        "--qualification",
        default="build/stride-maze-tail-evidence-qualification-v1",
    )
    parser.add_argument(
        "--output",
        default="build/stride-maze-tail-evidence-preflight-analysis-v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_maze_tail_evidence_dataset(
            resolve_cli_path(PROJECT_ROOT, arguments.fetched),
            resolve_cli_path(PROJECT_ROOT, arguments.config),
            resolve_cli_path(PROJECT_ROOT, arguments.dataset),
        )
    elif arguments.command == "qualify":
        result = run_maze_tail_evidence_qualification(
            resolve_cli_path(PROJECT_ROOT, arguments.config),
            resolve_cli_path(PROJECT_ROOT, arguments.qualification),
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    else:
        result = analyze_maze_tail_evidence_qualification(
            resolve_cli_path(PROJECT_ROOT, arguments.config),
            resolve_cli_path(PROJECT_ROOT, arguments.dataset),
            resolve_cli_path(PROJECT_ROOT, arguments.qualification),
            resolve_cli_path(PROJECT_ROOT, arguments.output),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if arguments.dry_run or result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
