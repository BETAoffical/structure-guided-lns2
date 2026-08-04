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

from experiments.stride_maprank_counterfactual import (  # noqa: E402
    analyze_override_counterfactual,
    collect_override_counterfactual,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired-seed STRIDE-MapRank override counterfactual."
    )
    parser.add_argument("command", choices=("collect", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "collect":
        report = collect_override_counterfactual(
            arguments.config, arguments.output, resume=arguments.resume
        )
    else:
        report = analyze_override_counterfactual(
            arguments.config, arguments.output, arguments.output
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("passed", report.get("complete", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
