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

from experiments.stride_stage4r_high_load import (  # noqa: E402
    analyze_high_load_diagnostic,
    run_high_load_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired STRIDE Stage 4R high-load diagnostic."
    )
    parser.add_argument("mode", choices=("dry-run", "run", "analyze"))
    parser.add_argument(
        "--config", default="configs/stride_stage4r_high_load_diagnostic.json"
    )
    parser.add_argument(
        "--output", default="build/stride-stage4r-high-load-v1"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "analyze":
        report = analyze_high_load_diagnostic(arguments.config, arguments.output)
    else:
        report = run_high_load_diagnostic(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.mode == "dry-run",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
