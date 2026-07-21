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

from experiments.high_load_rescue import collect_high_load_rescue_data  # noqa: E402
from experiments._common import resolve_cli_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect paired high-load failure-state rescue outcomes."
    )
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--train-states", type=int, default=48)
    parser.add_argument("--validation-states", type=int, default=12)
    parser.add_argument("--initial-trials", type=int, default=2)
    parser.add_argument("--maximum-trials", type=int, default=2)
    parser.add_argument(
        "--neighborhood-sizes",
        default="4,8,12,16",
        help="sorted comma-separated rescue candidate sizes",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = collect_high_load_rescue_data(
        source_roots={
            "policy_train": resolve_cli_path(PROJECT_ROOT, arguments.train_source),
            "policy_validation": resolve_cli_path(PROJECT_ROOT, arguments.validation_source),
        },
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        controller_bundle=resolve_cli_path(PROJECT_ROOT, arguments.controller_bundle),
        maximum_states={
            "policy_train": arguments.train_states,
            "policy_validation": arguments.validation_states,
        },
        initial_trials=arguments.initial_trials,
        maximum_trials=arguments.maximum_trials,
        neighborhood_sizes=tuple(
            int(value.strip())
            for value in str(arguments.neighborhood_sizes).split(",")
            if value.strip()
        ),
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
