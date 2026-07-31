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
    analyze_success_only_ttf,
    audit_balanced_cohort_difficulty,
    build_replacement_dataset,
    collect_scheduled,
    materialize_compute_load_candidate_pool,
    materialize_qualified_compute_load_pool,
    materialize_registered_compute_load_cohort,
    merge_datasets,
    prepare_corrected_native_formal_config,
    prepare_movingai_map_derived_dataset,
    prepare_movingai_dataset,
    qualify_corrected_native_schedule,
    qualify_corrected_native_seed_pool,
    rebind_corrected_native_schedule,
    recover_scheduled_partial_traces,
    select_balanced_cohort,
    select_compute_load_balanced_cohort,
    select_corrected_native_seed_schedule,
    verify_compute_load_cohort_registration,
)
from experiments.closed_loop_confirmation import (  # noqa: E402
    run_closed_loop_collection,
)
from experiments.corrected_native_full_pool import (  # noqa: E402
    materialize_corrected_native_selected_dataset,
    select_corrected_native_full_pool_schedule,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _extension_specs(values: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    expected = {"id", "dataset", "qualification", "config"}
    for raw in values:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "--corrected-extension must be a JSON object"
            ) from error
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or not all(
                isinstance(value.get(field), str) and value[field]
                for field in expected
            )
        ):
            raise ValueError(
                "--corrected-extension requires exactly the string fields "
                "id,dataset,qualification,config"
            )
        specs.append(
            {
                "id": value["id"],
                "dataset": str(_resolve(value["dataset"])),
                "qualification": str(_resolve(value["qualification"])),
                "config": str(_resolve(value["config"])),
            }
        )
    if len({value["id"] for value in specs}) != len(specs):
        raise ValueError("--corrected-extension IDs must be unique")
    return specs


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
            "prepare-map-derived-movingai",
            "merge",
            "replace",
            "materialize-load-pool",
            "materialize-qualified-load-pool",
            "select",
            "select-load-balanced",
            "verify-load-cohort",
            "materialize-registered-cohort",
            "qualify-corrected-schedule",
            "qualify-corrected-seed-pool",
            "qualify-corrected-pool",
            "select-corrected-full-pool",
            "materialize-corrected-full-pool",
            "prepare-corrected-full-config",
            "collect-corrected-full-pool",
            "rebind-corrected-schedule",
            "select-corrected-seed-schedule",
            "recover-partials",
            "dry-run",
            "collect",
            "collect-corrected",
            "analyze",
            "analyze-success-ttf",
            "audit-difficulty",
        ),
    )
    parser.add_argument("--fetched-movingai", default="build/initlns-v2-mixed-movingai-raw-v1")
    parser.add_argument(
        "--movingai-config", default="configs/balanced_wall_clock_movingai_source.json"
    )
    parser.add_argument(
        "--movingai-map-config",
        default="configs/balanced_wall_clock_movingai_map_derived_candidates_v4.json",
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
    parser.add_argument(
        "--load-pool-config",
        default="configs/balanced_wall_clock_compute_load_pool_v3.json",
    )
    parser.add_argument(
        "--load-pool-dataset",
        default="build/initlns-v2-mixed-compute-load-pool-v3",
    )
    parser.add_argument(
        "--qualified-load-pool-config",
        default="configs/balanced_wall_clock_qualified_pool_v6.json",
    )
    parser.add_argument(
        "--qualified-load-pool-dataset",
        default="build/initlns-v2-mixed-qualified-compute-load-pool-v6",
    )
    parser.add_argument(
        "--qualified-load-pool-qualification",
        default="build/initlns-v2-mixed-qualified-compute-load-qualification-v6",
    )
    parser.add_argument("--config", default="configs/balanced_wall_clock_collection.json")
    parser.add_argument(
        "--corrected-config",
        default="configs/balanced_wall_clock_corrected_native_v1.json",
    )
    parser.add_argument(
        "--corrected-pool-dataset",
        default="build/initlns-v2-mixed-qualified-compute-load-pool-v6",
    )
    parser.add_argument(
        "--corrected-pool-config",
        default="configs/balanced_wall_clock_corrected_native_pool_v1.json",
    )
    parser.add_argument(
        "--corrected-pool-qualification",
        default="build/initlns-v2-corrected-native-full-pool-qualification-v3",
    )
    parser.add_argument(
        "--corrected-source-registration",
        default="configs/balanced_wall_clock_compute_load_cohort_v6.json",
    )
    parser.add_argument(
        "--corrected-full-schedule-output",
        default="build/initlns-v2-corrected-native-expanded-pool-cohort-v3",
    )
    parser.add_argument(
        "--corrected-formal-output",
        default="build/initlns-v2-corrected-native-selected-formal-v3",
    )
    parser.add_argument(
        "--corrected-formal-config-output",
        default=(
            "build/initlns-v2-corrected-native-selected-formal-v3/"
            "formal_config.json"
        ),
    )
    parser.add_argument(
        "--corrected-full-collection",
        default="build/initlns-v2-corrected-native-selected-collection-v3",
    )
    parser.add_argument(
        "--corrected-extension",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "Repeatable JSON object with id,dataset,qualification,config "
            "for an outcome-blind corrected-native extension pool."
        ),
    )
    parser.add_argument("--qualification", default="build/initlns-v2-mixed-balanced-qualification-v1")
    parser.add_argument("--cohort", default="build/initlns-v2-mixed-balanced-cohort-v1")
    parser.add_argument(
        "--source-schedule-root",
        default="build/initlns-v2-mixed-qualified-compute-load-cohort-v6",
    )
    parser.add_argument(
        "--corrected-qualification",
        default="build/initlns-v2-corrected-native-qualification-v1",
    )
    parser.add_argument(
        "--corrected-schedule-output",
        default="build/initlns-v2-corrected-native-cohort-v1",
    )
    parser.add_argument(
        "--corrected-seed-pool-qualification",
        default="build/initlns-v2-corrected-native-seed-pool-qualification-v1",
    )
    parser.add_argument(
        "--corrected-seed-schedule-output",
        default="build/initlns-v2-corrected-native-seed-reselected-cohort-v1",
    )
    parser.add_argument(
        "--corrected-observed-qualification",
        default="build/initlns-v2-corrected-native-qualification-v1",
    )
    parser.add_argument(
        "--corrected-seed-probe-qualification",
        default="build/initlns-v2-corrected-native-seed-probe-v1",
    )
    parser.add_argument(
        "--corrected-collection",
        default="build/initlns-v2-corrected-native-collection-v1",
    )
    parser.add_argument("--collection", default="build/initlns-v2-mixed-balanced-collection-v1")
    parser.add_argument("--report", default="build/initlns-v2-mixed-balanced-report-v1")
    parser.add_argument(
        "--difficulty-config",
        default="configs/balanced_wall_clock_difficulty_audit.json",
    )
    parser.add_argument(
        "--cohort-registration",
        default=None,
        help="Optional checksum-pinned cohort registration required by formal runs.",
    )
    parser.add_argument(
        "--formal-cohort-dataset",
        default="build/initlns-v2-mixed-qualified-compute-load-formal-dataset-v6",
    )
    parser.add_argument(
        "--original-bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument(
        "--mixed-bundle", default="artifacts/initlns-mixed-full-controller-v2"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stopping-rule",
        choices=("wall-clock-fixed-metric",),
        default="wall-clock-fixed-metric",
        help=(
            "Formal reruns have no 100-repair execution cap while retaining "
            "the frozen 100-step AUC metric."
        ),
    )
    arguments = parser.parse_args()
    if arguments.phase == "prepare-movingai":
        result = prepare_movingai_dataset(
            _resolve(arguments.fetched_movingai),
            _resolve(arguments.movingai_config),
            _resolve(arguments.movingai_dataset),
        )
    elif arguments.phase == "prepare-map-derived-movingai":
        result = prepare_movingai_map_derived_dataset(
            _resolve(arguments.fetched_movingai),
            _resolve(arguments.movingai_map_config),
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
    elif arguments.phase == "materialize-load-pool":
        result = materialize_compute_load_candidate_pool(
            _resolve(arguments.load_pool_config),
            _resolve(arguments.load_pool_dataset),
        )
    elif arguments.phase == "materialize-qualified-load-pool":
        result = materialize_qualified_compute_load_pool(
            _resolve(arguments.qualified_load_pool_config),
            _resolve(arguments.qualified_load_pool_dataset),
            _resolve(arguments.qualified_load_pool_qualification),
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
    elif arguments.phase == "verify-load-cohort":
        if arguments.cohort_registration is None:
            parser.error("verify-load-cohort requires --cohort-registration")
        result = verify_compute_load_cohort_registration(
            _resolve(arguments.cohort_registration)
        )
    elif arguments.phase == "materialize-registered-cohort":
        if arguments.cohort_registration is None:
            parser.error("materialize-registered-cohort requires --cohort-registration")
        result = materialize_registered_compute_load_cohort(
            _resolve(arguments.cohort_registration),
            _resolve(arguments.formal_cohort_dataset),
        )
    elif arguments.phase == "qualify-corrected-schedule":
        result = qualify_corrected_native_schedule(
            dataset=_resolve(arguments.formal_cohort_dataset),
            config=_resolve(arguments.corrected_config),
            source_schedule_root=_resolve(arguments.source_schedule_root),
            output=_resolve(arguments.corrected_qualification),
            original_bundle=_resolve(arguments.original_bundle),
            workers=arguments.workers,
            dry_run=False,
        )
    elif arguments.phase == "qualify-corrected-seed-pool":
        result = qualify_corrected_native_seed_pool(
            dataset=_resolve(arguments.formal_cohort_dataset),
            config=_resolve(arguments.corrected_config),
            source_schedule_root=_resolve(arguments.source_schedule_root),
            output=_resolve(arguments.corrected_seed_pool_qualification),
            original_bundle=_resolve(arguments.original_bundle),
            workers=arguments.workers,
            dry_run=False,
        )
    elif arguments.phase == "qualify-corrected-pool":
        result = run_closed_loop_collection(
            dataset=_resolve(arguments.corrected_pool_dataset),
            config_path=_resolve(arguments.corrected_pool_config),
            output=_resolve(arguments.corrected_pool_qualification),
            phase="qualify",
            workers=arguments.workers,
            resume=arguments.resume,
            dry_run=False,
            controller="v2-full",
            feature_backend="auto",
            controller_bundle=_resolve(arguments.original_bundle),
            controller_runtime="optimized",
            verification_profile="deployment",
            stopping_rule=arguments.stopping_rule,
            use_global_collection_lock=False,
        )
    elif arguments.phase == "select-corrected-full-pool":
        result = select_corrected_native_full_pool_schedule(
            dataset=_resolve(arguments.corrected_pool_dataset),
            qualification=_resolve(
                arguments.corrected_pool_qualification
            ),
            source_registration=_resolve(
                arguments.corrected_source_registration
            ),
            source_schedule_root=_resolve(arguments.source_schedule_root),
            config=_resolve(arguments.corrected_pool_config),
            output=_resolve(arguments.corrected_full_schedule_output),
            extensions=_extension_specs(arguments.corrected_extension),
        )
    elif arguments.phase == "materialize-corrected-full-pool":
        result = materialize_corrected_native_selected_dataset(
            schedule_root=_resolve(
                arguments.corrected_full_schedule_output
            ),
            base_dataset=_resolve(arguments.corrected_pool_dataset),
            output=_resolve(arguments.corrected_formal_output),
            extensions=_extension_specs(arguments.corrected_extension),
        )
    elif arguments.phase == "prepare-corrected-full-config":
        formal_root = _resolve(arguments.corrected_formal_output)
        result = prepare_corrected_native_formal_config(
            base_config=_resolve(arguments.corrected_pool_config),
            schedule_root=formal_root,
            dataset=formal_root / "dataset",
            output=_resolve(arguments.corrected_formal_config_output),
        )
    elif arguments.phase == "collect-corrected-full-pool":
        formal_root = _resolve(arguments.corrected_formal_output)
        result = collect_scheduled(
            dataset=formal_root / "dataset",
            config=_resolve(arguments.corrected_formal_config_output),
            qualification=None,
            schedule_root=formal_root,
            output=_resolve(arguments.corrected_full_collection),
            original_bundle=_resolve(arguments.original_bundle),
            mixed_bundle=_resolve(arguments.mixed_bundle),
            resume=arguments.resume,
            dry_run=False,
            registration=None,
            stopping_rule=arguments.stopping_rule,
        )
    elif arguments.phase == "rebind-corrected-schedule":
        result = rebind_corrected_native_schedule(
            source_schedule_root=_resolve(arguments.source_schedule_root),
            qualification=_resolve(arguments.corrected_qualification),
            output=_resolve(arguments.corrected_schedule_output),
        )
    elif arguments.phase == "select-corrected-seed-schedule":
        result = select_corrected_native_seed_schedule(
            dataset=_resolve(arguments.formal_cohort_dataset),
            source_schedule_root=_resolve(arguments.source_schedule_root),
            qualification=_resolve(
                arguments.corrected_seed_pool_qualification
            ),
            observed_qualification=_resolve(
                arguments.corrected_observed_qualification
            ),
            seed_probe_qualification=_resolve(
                arguments.corrected_seed_probe_qualification
            ),
            output=_resolve(arguments.corrected_seed_schedule_output),
        )
    elif arguments.phase == "collect-corrected":
        result = collect_scheduled(
            dataset=_resolve(arguments.formal_cohort_dataset),
            config=_resolve(arguments.corrected_config),
            qualification=_resolve(
                arguments.corrected_seed_pool_qualification
            ),
            schedule_root=_resolve(
                arguments.corrected_seed_schedule_output
            ),
            output=_resolve(arguments.corrected_collection),
            original_bundle=_resolve(arguments.original_bundle),
            mixed_bundle=_resolve(arguments.mixed_bundle),
            resume=arguments.resume,
            dry_run=False,
            registration=None,
            stopping_rule=arguments.stopping_rule,
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
            registration=(
                _resolve(arguments.cohort_registration)
                if arguments.cohort_registration is not None
                else None
            ),
            stopping_rule=arguments.stopping_rule,
        )
    elif arguments.phase == "recover-partials":
        result = recover_scheduled_partial_traces(
            _resolve(arguments.collection),
            _resolve(arguments.cohort),
        )
    elif arguments.phase == "analyze":
        result = analyze_scheduled(
            _resolve(arguments.collection),
            _resolve(arguments.cohort),
            _resolve(arguments.report),
        )
    elif arguments.phase == "analyze-success-ttf":
        result = analyze_success_only_ttf(
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
