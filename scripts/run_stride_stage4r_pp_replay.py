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

from experiments.stride_stage4r_pp_replay import (  # noqa: E402
    analyze_pp_replay,
    prepare_pp_replay_selection,
    run_pp_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only Stage 4R same-state, same-action-set "
            "paired PP-seed replay."
        )
    )
    parser.add_argument("mode", choices=("prepare", "dry-run", "run", "analyze"))
    parser.add_argument(
        "--config", default="configs/stride_stage4r_pp_replay.json"
    )
    parser.add_argument(
        "--selection",
        default="build/stride-stage4r-pp-replay-selection-v1/selection.jsonl",
    )
    parser.add_argument(
        "--output", default="build/stride-stage4r-pp-replay-v1"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "prepare":
        report = prepare_pp_replay_selection(arguments.config, arguments.selection)
    elif arguments.mode == "dry-run":
        report = run_pp_replay(
            arguments.config,
            arguments.selection,
            arguments.output,
            dry_run=True,
        )
    elif arguments.mode == "run":
        report = run_pp_replay(
            arguments.config,
            arguments.selection,
            arguments.output,
            resume=arguments.resume,
        )
    else:
        report = analyze_pp_replay(
            arguments.config,
            arguments.selection,
            arguments.output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
