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
from experiments.rescue_lite_confirmation import (  # noqa: E402
    run_rescue_lite_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fresh 30-state exact-fingerprint confirmation for the frozen "
            "4>8>Adaptive rescue policy."
        )
    )
    parser.add_argument(
        "--output", default="build/initlns-rescue-lite-confirmation-v1"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dataset-config",
        default="configs/rescue_lite_confirmation_dataset.json",
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--repair-aware-bundle",
        default="build/initlns-high-load-rescue-pilot-dense-v2/controller",
    )
    parser.add_argument(
        "--reference-dataset",
        action="append",
        default=["build/initlns-high-load-rescue-dataset-dense-v1"],
        help="previous synthetic dataset whose map hashes must not overlap",
    )
    parser.add_argument("--quota-per-cell", type=int, default=5)
    parser.add_argument("--trials", type=int, default=4)
    arguments = parser.parse_args()
    report = run_rescue_lite_confirmation(
        project_root=PROJECT_ROOT,
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        dataset_config=resolve_cli_path(PROJECT_ROOT, arguments.dataset_config),
        controller_bundle=resolve_cli_path(PROJECT_ROOT, arguments.controller_bundle),
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("decision") != "insufficient_confirmation_states" else 2


if __name__ == "__main__":
    raise SystemExit(main())
