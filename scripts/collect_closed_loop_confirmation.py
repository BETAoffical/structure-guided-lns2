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
    STOPPING_RULES,
    VERIFICATION_PROFILES,
    CollectionLockError,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import (  # noqa: E402
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMATS,
)
from experiments.online_feature_engine import FEATURE_BACKENDS  # noqa: E402


def _selected_job_keys(
    task_ids: list[str] | None,
    solver_seeds: list[int] | None,
) -> set[tuple[str, int]] | None:
    if solver_seeds is None:
        return None
    if not task_ids:
        raise ValueError("--solver-seed requires at least one --task-id")
    if any(seed < 0 for seed in solver_seeds):
        raise ValueError("--solver-seed must be non-negative")
    return {
        (str(task_id), int(seed))
        for task_id in task_ids
        for seed in solver_seeds
    }


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
        "--solver-seed",
        action="append",
        type=int,
        dest="solver_seeds",
        help=(
            "Restrict collection to one or more solver seeds. This requires "
            "at least one --task-id and registers the filtered cohort."
        ),
    )
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
    parser.add_argument(
        "--v3-s3-bundle",
        help="Sequence-aware bundle; required only with --controller v3-s3.",
    )
    parser.add_argument(
        "--qualification-source",
        help="Reuse a compatible qualification collection instead of resetting again.",
    )
    parser.add_argument("--wall-time-budget-seconds", type=float)
    parser.add_argument("--episode-process-timeout-seconds", type=float)
    parser.add_argument("--environment-time-limit-seconds", type=float)
    parser.add_argument(
        "--stopping-rule",
        choices=STOPPING_RULES,
        default="wall-clock",
    )
    arguments = parser.parse_args()
    try:
        job_keys = _selected_job_keys(arguments.task_ids, arguments.solver_seeds)
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
            v3_s3_bundle=arguments.v3_s3_bundle,
            job_keys=job_keys,
            cohort_job_keys=job_keys,
            wall_time_budget_seconds=arguments.wall_time_budget_seconds,
            episode_process_timeout_seconds=(
                arguments.episode_process_timeout_seconds
            ),
            environment_time_limit_seconds=(
                arguments.environment_time_limit_seconds
            ),
            stopping_rule=arguments.stopping_rule,
            qualification_source=arguments.qualification_source,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
