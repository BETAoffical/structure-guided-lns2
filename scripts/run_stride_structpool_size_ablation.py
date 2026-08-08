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

from experiments.stride_structpool_size_ablation import (  # noqa: E402
    analyze_size_ablation,
    audit_size_labels,
    collect_size_labels,
    extract_size_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered StructPool family-by-size ablation."
    )
    parser.add_argument(
        "command", choices=("extract-grid", "collect-labels", "audit-labels", "analyze")
    )
    parser.add_argument(
        "--config", default="configs/stride_structpool_size_ablation_v1.json"
    )
    parser.add_argument("--grid", default="build/stride-structpool-size-grid-v1")
    parser.add_argument("--labels", default="build/stride-structpool-size-labels-v1")
    parser.add_argument("--output")
    parser.add_argument(
        "--audit-output",
        default="build/stride-structpool-size-labels-current-audit-v1",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "extract-grid":
        report = extract_size_grid(
            config_path=arguments.config,
            output=arguments.output or arguments.grid,
            workers=arguments.workers,
            resume=arguments.resume,
        )
    elif arguments.command == "collect-labels":
        report = collect_size_labels(
            config_path=arguments.config,
            grid=arguments.grid,
            output=arguments.output or arguments.labels,
            workers=arguments.workers,
            resume=arguments.resume,
        )
    elif arguments.command == "audit-labels":
        report = audit_size_labels(
            config_path=arguments.config,
            grid=arguments.grid,
            labels=arguments.labels,
            output=arguments.output or arguments.audit_output,
        )
    else:
        report = analyze_size_ablation(
            config_path=arguments.config,
            labels=arguments.labels,
            output=arguments.output or "build/stride-structpool-size-ablation-v1",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
