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
from experiments.rescue_confirmation_qualification import (  # noqa: E402
    run_rescue_confirmation_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify fresh high-pressure recipes before creating another "
            "locked rescue-lite confirmation set."
        )
    )
    parser.add_argument(
        "--output", default="build/initlns-rescue-confirmation-qualification-v1"
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/rescue_lite_qualification_dataset.json",
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--wall-time-seconds", type=float, default=180.0)
    parser.add_argument(
        "--reference-dataset",
        action="append",
        default=[
            "build/initlns-high-load-rescue-dataset-dense-v1",
            "build/initlns-high-load-rescue-dataset-hard-v9",
            "build/initlns-rescue-lite-confirmation-v1/dataset",
        ],
    )
    arguments = parser.parse_args()
    report = run_rescue_confirmation_qualification(
        project_root=PROJECT_ROOT,
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        dataset_config=resolve_cli_path(PROJECT_ROOT, arguments.dataset_config),
        controller_bundle=resolve_cli_path(
            PROJECT_ROOT, arguments.controller_bundle
        ),
        reference_datasets=[
            resolve_cli_path(PROJECT_ROOT, value)
            for value in arguments.reference_dataset
        ],
        workers=arguments.workers,
        resume=arguments.resume,
        max_decisions=arguments.max_decisions,
        wall_time_seconds=arguments.wall_time_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualification_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
