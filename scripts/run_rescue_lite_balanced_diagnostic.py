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
from experiments.rescue_lite_balanced_diagnostic import (  # noqa: E402
    run_balanced_rescue_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a balanced same-state rescue diagnostic that is explicitly "
            "not eligible for controller promotion."
        )
    )
    parser.add_argument(
        "--source",
        default="build/initlns-rescue-lite-locked-confirmation-v1",
    )
    parser.add_argument(
        "--output",
        default="build/initlns-rescue-lite-balanced-diagnostic-v1",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quota-per-cell", type=int, default=4)
    parser.add_argument("--trials", type=int, default=4)
    arguments = parser.parse_args()
    report = run_balanced_rescue_diagnostic(
        project_root=PROJECT_ROOT,
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        workers=arguments.workers,
        resume=arguments.resume,
        quota_per_cell=arguments.quota_per_cell,
        trial_count=arguments.trials,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
