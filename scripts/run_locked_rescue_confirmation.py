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

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.rescue_lite_locked_confirmation import (  # noqa: E402
    run_locked_rescue_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot locked rescue-lite confirmation using recipes "
            "frozen by the independent qualification stage."
        )
    )
    parser.add_argument(
        "--output", default="build/initlns-rescue-lite-locked-confirmation-v1"
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/rescue_lite_locked_confirmation_dataset.json",
    )
    parser.add_argument(
        "--qualification-report",
        default=(
            "build/initlns-rescue-confirmation-qualification-v2/"
            "qualification_report.json"
        ),
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--repair-aware-bundle",
        default="build/initlns-high-load-rescue-pilot-dense-v2/controller",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quota-per-cell", type=int, default=5)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--wall-time-seconds", type=float, default=180.0)
    parser.add_argument(
        "--reference-dataset",
        action="append",
        default=[
            "build/initlns-high-load-rescue-dataset-dense-v1",
            "build/initlns-high-load-rescue-dataset-hard-v9",
            "build/initlns-rescue-lite-confirmation-v1/dataset",
            "build/initlns-rescue-confirmation-qualification-v2/dataset",
        ],
    )
    arguments = parser.parse_args()
    report = run_locked_rescue_confirmation(
        project_root=PROJECT_ROOT,
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        dataset_config=resolve_cli_path(PROJECT_ROOT, arguments.dataset_config),
        qualification_report=resolve_cli_path(
            PROJECT_ROOT, arguments.qualification_report
        ),
        controller_bundle=resolve_cli_path(
            PROJECT_ROOT, arguments.controller_bundle
        ),
        repair_aware_bundle=resolve_cli_path(
            PROJECT_ROOT, arguments.repair_aware_bundle
        ),
        reference_datasets=[
            resolve_cli_path(PROJECT_ROOT, value)
            for value in arguments.reference_dataset
        ],
        workers=arguments.workers,
        resume=arguments.resume,
        quota_per_cell=arguments.quota_per_cell,
        trial_count=arguments.trials,
        max_decisions=arguments.max_decisions,
        wall_time_seconds=arguments.wall_time_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["decision"] == "locked_confirmation_insufficient_states" else 0


if __name__ == "__main__":
    raise SystemExit(main())
