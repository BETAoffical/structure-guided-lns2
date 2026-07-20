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

from experiments.closed_loop_confirmation import (  # noqa: E402
    CONTROLLER_MODES,
    CONTROLLER_RUNTIMES,
    VERIFICATION_PROFILES,
    CollectionLockError,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import (  # noqa: E402
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMATS,
)
from experiments.online_feature_engine import FEATURE_BACKENDS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect frozen InitLNS neighborhood-ranker closed-loop episodes."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default="configs/closed_loop_confirmation_collection.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "qualify",
            "official_adaptive",
            "fixed_target",
            "fixed_collision",
            "fixed_random",
            "proposal_dynamic",
            "realized_dynamic",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument(
        "--trace-format",
        choices=TRACE_FORMATS,
        default=TRACE_FORMAT_DELTA_GZIP_V2,
    )
    parser.add_argument("--controller", choices=CONTROLLER_MODES)
    parser.add_argument(
        "--feature-backend",
        choices=tuple(value for value in FEATURE_BACKENDS if value != "reference"),
        default="auto",
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--controller-runtime",
        choices=CONTROLLER_RUNTIMES,
        default="reference",
    )
    parser.add_argument(
        "--verification-profile",
        choices=VERIFICATION_PROFILES,
        default="audit",
    )
    parser.add_argument("--stall-guard-config")
    arguments = parser.parse_args()
    try:
        report = run_closed_loop_collection(
            arguments.dataset,
            arguments.config,
            arguments.output,
            phase=arguments.phase,
            workers=arguments.workers,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            task_ids=arguments.task_ids,
            trace_format=arguments.trace_format,
            controller=arguments.controller,
            feature_backend=arguments.feature_backend,
            controller_bundle=arguments.controller_bundle,
            controller_runtime=arguments.controller_runtime,
            verification_profile=arguments.verification_profile,
            stall_guard_config=arguments.stall_guard_config,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
