#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_augcontrol_evaluation import (  # noqa: E402
    analyze_augcontrol_layer,
    analyze_augcontrol_shadow,
    run_augcontrol_layer,
    run_augcontrol_shadow,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run staged STRIDE augcontrol Shadow and raw-TTF evaluation."
    )
    parser.add_argument(
        "mode",
        choices=(
            "shadow",
            "analyze-shadow",
            "development",
            "analyze-development",
            "formal",
            "analyze-formal",
        ),
    )
    parser.add_argument(
        "--config", default="configs/stride_augcontrol_evaluation.json"
    )
    parser.add_argument(
        "--output", default="build/stride-augcontrol-evaluation-v1"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if arguments.mode in {"shadow", "analyze-shadow"}:
        target = output / "shadow"
        report = (
            run_augcontrol_shadow(arguments.config, target, resume=arguments.resume)
            if arguments.mode == "shadow"
            else analyze_augcontrol_shadow(arguments.config, target)
        )
    else:
        formal = arguments.mode.endswith("formal")
        layer = "formal_ood" if formal else "development"
        target = output / ("formal_ood" if formal else "development")
        report = (
            analyze_augcontrol_layer(arguments.config, target, layer_name=layer)
            if arguments.mode.startswith("analyze-")
            else run_augcontrol_layer(
                arguments.config,
                target,
                layer_name=layer,
                resume=arguments.resume,
                dry_run=arguments.dry_run,
            )
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
