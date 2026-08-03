from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stride_stage4r_seed_diagnostic import (  # noqa: E402
    analyze_seed_diagnostic,
    prepare_seed_diagnostic_dataset,
    run_seed_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered outcome-informed, diagnostic-only STRIDE "
            "Stage 4R multi-seed TTF check."
        )
    )
    parser.add_argument("mode", choices=("prepare", "dry-run", "run", "analyze"))
    parser.add_argument(
        "--config", default="configs/stride_stage4r_seed_diagnostic.json"
    )
    parser.add_argument(
        "--dataset", default="build/stride-stage4r-seed-diagnostic-dataset-v1"
    )
    parser.add_argument(
        "--output", default="build/stride-stage4r-seed-diagnostic-v1"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "prepare":
        report = prepare_seed_diagnostic_dataset(arguments.config, arguments.dataset)
    elif arguments.mode == "dry-run":
        report = run_seed_diagnostic(
            arguments.config,
            arguments.dataset,
            arguments.output,
            dry_run=True,
        )
    elif arguments.mode == "run":
        report = run_seed_diagnostic(
            arguments.config,
            arguments.dataset,
            arguments.output,
            resume=arguments.resume,
        )
    else:
        report = analyze_seed_diagnostic(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
