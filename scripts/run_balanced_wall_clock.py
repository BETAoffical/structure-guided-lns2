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

from experiments.balanced_wall_clock import (  # noqa: E402
    analyze_scheduled,
    audit_balanced_cohort_difficulty,
    build_replacement_dataset,
    collect_scheduled,
    merge_datasets,
    prepare_movingai_dataset,
    select_balanced_cohort,
    select_compute_load_balanced_cohort,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, run, and audit the conflict-balanced V2/Mixed Full wall-clock study."
        )
    )
    parser.add_argument(
        "phase",
        choices=(
            "prepare-movingai",
            "merge",
            "replace",
            "select",
            "select-load-balanced",
            "dry-run",
            "collect",
            "analyze",
            "audit-difficulty",
        ),
    )
    parser.add_argument("--fetched-movingai", default="build/initlns-v2-mixed-movingai-raw-v1")
    parser.add_argument(
        "--movingai-config", default="configs/balanced_wall_clock_movingai_source.json"
    )
    parser.add_argument("--movingai-dataset", default="build/initlns-v2-mixed-movingai-v1")
    parser.add_argument("--generated-dataset", default="build/initlns-v2-mixed-balanced-generated-v1")
    parser.add_argument("--dataset", default="build/initlns-v2-mixed-balanced-dataset-v1")
    parser.add_argument("--original-dataset", default="build/initlns-v2-mixed-balanced-dataset-v1")
    parser.add_argument(
        "--movingai-candidates", default="build/initlns-v2-mixed-movingai-candidates-v2"
    )
    parser.add_argument(
        "--generated-candidates", default="build/initlns-v2-mixed-generated-candidates-v2b"
    )
    parser.add_argument(
        "--additional-generated-candidates",
        default="build/initlns-v2-mixed-generated-high-pool-v2",
    )
    parser.add_argument(
        "--replacement-config", default="configs/balanced_wall_clock_replacement_v2.json"
    )
    parser.add_argument(
        "--replacement-output",
        default="build/initlns-v2-mixed-balanced-dataset-v2",
    )
    parser.add_argument("--config", default="configs/balanced_wall_clock_collection.json")
    parser.add_argument("--qualification", default="build/initlns-v2-mixed-balanced-qualification-v1")
    parser.add_argument("--cohort", default="build/initlns-v2-mixed-balanced-cohort-v1")
    parser.add_argument("--collection", default="build/initlns-v2-mixed-balanced-collection-v1")
    parser.add_argument("--report", default="build/initlns-v2-mixed-balanced-report-v1")
    parser.add_argument(
        "--difficulty-config",
        default="configs/balanced_wall_clock_difficulty_audit.json",
    )
    parser.add_argument(
        "--original-bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument(
        "--mixed-bundle", default="artifacts/initlns-mixed-full-controller-v2"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "prepare-movingai":
        result = prepare_movingai_dataset(
            _resolve(arguments.fetched_movingai),
            _resolve(arguments.movingai_config),
            _resolve(arguments.movingai_dataset),
        )
    elif arguments.phase == "merge":
        result = merge_datasets(
            _resolve(arguments.generated_dataset),
            _resolve(arguments.movingai_dataset),
            _resolve(arguments.dataset),
        )
    elif arguments.phase == "replace":
        result = build_replacement_dataset(
            original=_resolve(arguments.original_dataset),
            movingai_candidates=_resolve(arguments.movingai_candidates),
            generated_candidates=_resolve(arguments.generated_candidates),
            additional_generated_candidates=_resolve(
                arguments.additional_generated_candidates
            ),
            selection_config=_resolve(arguments.replacement_config),
            output=_resolve(arguments.replacement_output),
        )
    elif arguments.phase == "select":
        result = select_balanced_cohort(
            _resolve(arguments.dataset),
            _resolve(arguments.qualification),
            _resolve(arguments.cohort),
        )
    elif arguments.phase == "select-load-balanced":
        result = select_compute_load_balanced_cohort(
            _resolve(arguments.dataset),
            _resolve(arguments.qualification),
            _resolve(arguments.cohort),
            _resolve(arguments.difficulty_config),
        )
    elif arguments.phase in {"dry-run", "collect"}:
        result = collect_scheduled(
            dataset=_resolve(arguments.dataset),
            config=_resolve(arguments.config),
            qualification=_resolve(arguments.qualification),
            schedule_root=_resolve(arguments.cohort),
            output=_resolve(arguments.collection),
            original_bundle=_resolve(arguments.original_bundle),
            mixed_bundle=_resolve(arguments.mixed_bundle),
            resume=arguments.resume,
            dry_run=arguments.phase == "dry-run",
        )
    elif arguments.phase == "analyze":
        result = analyze_scheduled(
            _resolve(arguments.collection),
            _resolve(arguments.cohort),
            _resolve(arguments.report),
        )
    else:
        result = audit_balanced_cohort_difficulty(
            _resolve(arguments.collection),
            _resolve(arguments.cohort),
            _resolve(arguments.report),
            _resolve(arguments.difficulty_config),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
