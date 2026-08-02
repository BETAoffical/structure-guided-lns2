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

from experiments.stride_stage4r_quick import (  # noqa: E402
    analyze_stage4r_quick,
    prepare_stage4r_quick_dataset,
    run_stage4r_quick,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the registered diagnostic-only STRIDE Stage 4R TTF Quick."
    )
    parser.add_argument(
        "mode", choices=("prepare", "dry-run", "run", "analyze")
    )
    parser.add_argument(
        "--config", default="configs/stride_stage4r_quick.json"
    )
    parser.add_argument(
        "--dataset", default="build/stride-stage4r-quick-dataset-v1"
    )
    parser.add_argument(
        "--output", default="build/stride-stage4r-quick-v1"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "prepare":
        report = prepare_stage4r_quick_dataset(arguments.config, arguments.dataset)
    elif arguments.mode == "dry-run":
        report = run_stage4r_quick(
            arguments.config,
            arguments.dataset,
            arguments.output,
            dry_run=True,
        )
    elif arguments.mode == "run":
        report = run_stage4r_quick(
            arguments.config,
            arguments.dataset,
            arguments.output,
            resume=arguments.resume,
        )
    else:
        report = analyze_stage4r_quick(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
