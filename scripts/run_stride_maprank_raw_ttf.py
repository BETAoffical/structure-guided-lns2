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

from experiments.stride_maprank_raw_ttf import (  # noqa: E402
    analyze_maprank_raw_ttf_layer,
    analyze_maprank_runtime_equivalence,
    prepare_maprank_fresh_dataset,
    run_maprank_raw_ttf_layer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run paired run-to-completion STRIDE-MapRank raw-TTF layers."
    )
    parser.add_argument(
        "command",
        choices=(
            "high-load",
            "analyze-high-load",
            "analyze-runtime-equivalence",
            "prepare-fresh",
            "fresh",
            "analyze-fresh",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--reference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "analyze-runtime-equivalence":
        if not arguments.reference or not arguments.output:
            parser.error(
                "analyze-runtime-equivalence requires --reference and --output"
            )
        report = analyze_maprank_runtime_equivalence(
            arguments.reference, arguments.output
        )
        code = 0 if report["passed"] else 1
    elif arguments.command == "prepare-fresh":
        report = prepare_maprank_fresh_dataset(arguments.config)
        code = 0
    else:
        if not arguments.output:
            parser.error(f"{arguments.command} requires --output")
        layer = (
            "high_load_development"
            if "high-load" in arguments.command
            else "fresh_map_raw_ttf"
        )
        if arguments.command.startswith("analyze-"):
            report = analyze_maprank_raw_ttf_layer(
                arguments.config, arguments.output, layer_name=layer
            )
        else:
            report = run_maprank_raw_ttf_layer(
                arguments.config,
                arguments.output,
                layer_name=layer,
                resume=arguments.resume,
                dry_run=arguments.dry_run,
            )
        code = 0 if arguments.dry_run or report.get("integrity_passed") else 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
