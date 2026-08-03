#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_boundary_gate_ablation import (  # noqa: E402
    analyze_boundary_gate_ablation,
    run_boundary_gate_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the non-promoting, run-to-completion boundary gate ablation."
        )
    )
    parser.add_argument("--mode", choices=("run", "analyze"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "analyze":
        if arguments.resume or arguments.dry_run:
            parser.error("--resume/--dry-run are only valid in run mode")
        report = analyze_boundary_gate_ablation(arguments.config, arguments.output)
    else:
        report = run_boundary_gate_ablation(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
