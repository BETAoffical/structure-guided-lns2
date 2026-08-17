from __future__ import annotations

import collections
import itertools
import json
import math
import os
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    CLOSED_LOOP_IMPLEMENTATION_FILES,
    NATIVE_SEMANTICS_SCHEMA,
    episode_id as _episode_id,
    select_rows_by_task_id as _selected_rows,
    sha256_file as _sha256,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V1,
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    TRACE_FORMATS,
    encode_finish_event,
    encode_initial_event,
    encode_transition_event,
    open_trace_text,
    partial_trace_path,
    storage_fingerprint,
    trace_file_metadata,
    trace_suffix,
)
from experiments.compact_controller_model import (
    compact_runtime_model,
    load_controller_bundle,
)
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.state_analysis import (
    StateAnalysis,
    analyze_static_grid,
)
from experiments.trace_replay import target_state_from_trace
from experiments.neighborhood_candidates import (
    _seed_isolation,
    conflict_density,
    conflict_severity,
    no_pruning_metrics,
)
from experiments.online_feature_engine import (
    FEATURE_BACKENDS,
    OnlineFeatureEngine,
    TopologyAnalysisCache,
    _native_vector_function,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _make_environment,
    _plain,
    POLICY_DESTROY_STRATEGIES,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.v3_s3 import (
    V3_S3_BUNDLE_SCHEMA,
    V3_S3_FEATURE_SCHEMA_ID,
    V3_S3_FEATURE_SCHEMA_SHA256,
    V3S3Bundle,
    load_v3_s3_bundle,
    s3_temporal_context,
)
from lns2_selector.compatibility.metrics import fixed_budget_conflict_auc
from lns2_selector.controllers import load_selector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.evaluation.trace_validation import (
    ClosedLoopTraceError,
    REPAIR_TIMING_SCHEMA,
    REPAIR_TIMING_SCHEMA_V1,
    REPAIR_TIMING_SCHEMA_V2,
    REPAIR_TIMING_SCHEMAS,
    _valid_episode_trace,
    native_repair_timing_schema as _native_repair_timing_schema,
    validate_closed_loop_trace,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.bounded_native_retry import (
    BoundedNativeRetryTracker,
    merged_retry_metrics,
)
from lns2_selector.runtime.failure_informed_rescue import (
    FailureInformedRescueTracker,
)
from lns2_selector.runtime.hybridstructpool import (
    HybridStructPoolResult,
    generate_hybridstructpool_runtime_candidates,
    hybridstructpool_high_stress_gate,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    OVERALL_ROLLBACK_ROUTED_HYBRIDSTRUCTPOOL_ID,
    ROLLBACK_AWARE_ROUTED_HYBRIDSTRUCTPOOL_ID,
    ROUTED_HYBRIDSTRUCTPOOL_ID,
    generate_routed_hybridstructpool_runtime_candidates,
    routed_hybridstructpool_high_stress_gate,
    validate_any_hybridstructpool_augmentation,
)
from lns2_selector.runtime.rollback_aware_selection import (
    ROLLBACK_AWARE_SELECTION_ID,
    ExactRollbackCandidateGuard,
)
from lns2_selector.runtime.overall_rollback_selection import (
    OVERALL_ROLLBACK_SELECTION_ID,
    ExactRollbackStateGuard,
)
from lns2_selector.runtime.signature_scoped_rescue import (
    SignatureScopedRescueTracker,
)
from lns2_selector.runtime.structshell_single_family import (
    STRUCTSHELL_SINGLE_FAMILY_POOL_ID,
    generate_structshell_single_family_runtime_candidates,
    structshell_single_family_ablation_gate,
)
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    generate_structshell_dual16_runtime_candidates,
    structshell_dual16_ablation_gate,
)
from lns2_selector.runtime.metrics import wall_clock_conflict_auc
from lns2_selector.runtime.contracts import (
    CONTROLLER_IDS,
    DIAGNOSTIC_CONTROLLER_IDS,
    SelectionRequest,
    Selector,
)
from lns2_selector.runtime.online_selection import (
    ClosedLoopExecutionError,
    EpisodeRepairSeedStream,
    feature_range_diagnostic,
    generate_online_candidates,
    online_candidate_rows,
    pp_replay_random_seed,
    proposal_random_seed,
    proposal_random_seeds,
    repair_random_seed,
    score_online_candidates,
    structpool_high_stress_gate,
    validate_repair_seed_policy,
    validate_structpool_augmentation,
    validate_topology_boundary_augmentation,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.runtime.slotpool_selection import (
    load_slotpool_model,
    reduce_slotpool_candidates,
)
from lns2_selector.solver.native import load_native_module
from lns2_selector.training.policy_bundle import (
    PortablePairwiseModel,
    export_portable_policy_bundle,
    load_frozen_policy_bundle,
    load_portable_pairwise_model_payload,
    verify_portable_policy_bundle,
)


def _proposal_uses_static_grid_cache(proposal_config: Mapping[str, Any]) -> bool:
    """Return whether any enabled topology augmentation requests map caching."""

    return any(
        dict(proposal_config.get(name) or {}).get("static_grid_cache") is True
        for name in ("topology_boundary", "structpool", "hybridstructpool")
    )


def _generate_fixed_structshell_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult | None:
    """Dispatch separately registered fixed StructShell runtime contracts."""

    pool_id = str(config.get("pool_id") or "")
    if pool_id == STRUCTSHELL_SINGLE_FAMILY_POOL_ID:
        return generate_structshell_single_family_runtime_candidates(
            state,
            analysis,
            v2_candidates=v2_candidates,
            v2_anchors=v2_anchors,
            config=config,
        )
    if pool_id == STRUCTSHELL_DUAL16_POOL_ID:
        return generate_structshell_dual16_runtime_candidates(
            state,
            analysis,
            v2_candidates=v2_candidates,
            v2_anchors=v2_anchors,
            config=config,
        )
    return None


CLOSED_LOOP_SCHEMA = "lns2.closed_loop_confirmation.v1"
EPISODE_SCHEMA = EPISODE_SCHEMA_V1
FIXED_POLICIES = ("fixed_target", "fixed_collision", "fixed_random")
POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")
SUPPORTED_POLICIES = ("official_adaptive", *FIXED_POLICIES, "proposal_dynamic", "realized_dynamic")
LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")
CONTROLLER_MODES = CONTROLLER_IDS
EXECUTABLE_CONTROLLER_MODES = (*CONTROLLER_IDS, *DIAGNOSTIC_CONTROLLER_IDS)
PAIRWISE_CONTROLLER_MODES = {
    "v2-full",
    "mixed-full-v2",
    "stride-control-v1",
    "stride-quality-v1",
    "stride-augcontrol-v1",
}
CONTROLLER_RUNTIMES = ("reference", "optimized", "auto")
VERIFICATION_PROFILES = ("audit", "deployment")
STOPPING_RULES = (
    "historical",
    "wall-clock",
    "wall-clock-fixed-metric",
    "run-to-completion",
)
WALL_CLOCK_SAFETY_MAX_DECISIONS = 100_000


def _ranking_order(
    rows: list[dict[str, Any]], values: list[float]
) -> list[str]:
    return [
        str(rows[index]["candidate_key"])
        for index in sorted(
            range(len(rows)),
            key=lambda index: (
                -round(float(values[index]), 12),
                str(rows[index]["candidate_key"]),
            ),
        )
    ]


def _selector_required_model_features(
    selector: Selector, profile: str
) -> set[str]:
    """Return every dense feature consumed by a selector's model layers."""

    main_models = getattr(selector, "models", None)
    if not isinstance(main_models, Mapping) or profile not in main_models:
        raise ValueError(f"selector lacks model profile: {profile}")
    model_groups = [main_models]
    anchor_models = getattr(selector, "anchor_models", None)
    if anchor_models is not None:
        if not isinstance(anchor_models, Mapping) or profile not in anchor_models:
            raise ValueError(f"selector lacks anchor profile: {profile}")
        model_groups.append(anchor_models)
    required = set()
    for models in model_groups:
        required.update(
            map(str, getattr(models[profile], "base_feature_names", ()))
        )
    if not required:
        raise ValueError(f"selector has no model features for profile: {profile}")
    return required


def _score_equivalence_diagnostic(
    *,
    left_rows: list[dict[str, Any]],
    left_index: int,
    left_scores: list[float],
    left_margin: float,
    right_rows: list[dict[str, Any]],
    right_index: int,
    right_scores: list[float],
    right_margin: float,
) -> dict[str, Any]:
    maximum_score_delta = max(
        (
            abs(float(left) - float(right))
            for left, right in zip(left_scores, right_scores)
        ),
        default=0.0,
    )
    left_order = _ranking_order(left_rows, left_scores)
    right_order = _ranking_order(right_rows, right_scores)
    return {
        "candidate_count": len(left_rows),
        "maximum_score_delta": maximum_score_delta,
        "selected_candidate_matches": left_index == right_index,
        "left_selected_candidate_id": str(left_rows[left_index]["candidate_key"]),
        "right_selected_candidate_id": str(right_rows[right_index]["candidate_key"]),
        "ranking_matches": left_order == right_order,
        "left_top_candidate_ids": left_order[:3],
        "right_top_candidate_ids": right_order[:3],
        "margin_delta": abs(float(left_margin) - float(right_margin)),
    }


def _matching_source_model(
    *,
    controller_path: Path,
    controller_bundle: Any,
    frozen_bundle: Any,
    model_registration: dict[str, Any],
    profile: str,
) -> tuple[Any, dict[str, Any]]:
    ranker_row = dict(controller_bundle.manifest["main_rankers"][profile])
    compact_path = controller_path / str(ranker_row["file"])
    compact_payload = _read_json(compact_path)
    expected_source_sha = str(compact_payload.get("source_model_sha256", "")).lower()
    if not expected_source_sha:
        raise ValueError(f"controller source model SHA256 is missing: {profile}")

    source_row_value = dict(
        controller_bundle.manifest.get("source_rankers") or {}
    ).get(profile)
    if source_row_value is not None:
        source_row = dict(source_row_value)
        source_path = controller_path / str(source_row["file"])
        actual_portable_sha = _sha256(source_path)
        if actual_portable_sha != str(source_row["sha256"]).lower():
            raise ValueError(
                f"controller portable source model SHA256 mismatch: {profile}"
            )
        source_payload = _read_json(source_path)
        model = load_portable_pairwise_model_payload(
            source_payload,
            expected_profile=profile,
            expected_source_model_sha256=expected_source_sha,
        )
        return model, {
            "kind": "controller_local_portable_source",
            "profile": profile,
            "source_model_sha256": expected_source_sha,
            "portable_sha256": actual_portable_sha,
            "file": source_path.name,
        }

    registered_frozen_sha = str(
        dict(model_registration.get("model_sha256") or {}).get(profile, "")
    ).lower()
    if expected_source_sha == registered_frozen_sha:
        return frozen_bundle.models[profile], {
            "kind": "registered_frozen_model",
            "profile": profile,
            "source_model_sha256": expected_source_sha,
        }

    raise ValueError(
        f"controller audit portable source model is missing for {profile}"
    )


DEFAULT_CONTROLLER_BUNDLE = "artifacts/initlns-closed-loop-controller-v2"
DEFAULT_V3_S3_BUNDLE = (
    "build/initlns-v3-s3-mixed-load-pilot-v5-adaptive/controller"
)
CONTROLLER_IMPLEMENTATION_FILES = CLOSED_LOOP_IMPLEMENTATION_FILES


def controller_implementation_fingerprint(project_root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256(project_root / relative)
        for relative in CONTROLLER_IMPLEMENTATION_FILES
    }
    native_module = None
    try:
        lns2_env = load_native_module()

        native_path = Path(str(lns2_env.__file__)).resolve()
        semantics_schema = str(
            getattr(lns2_env, "native_semantics_schema", "")
        )
        if semantics_schema != NATIVE_SEMANTICS_SCHEMA:
            raise RuntimeError(
                "loaded lns2_env has unsupported native semantics schema: "
                f"{semantics_schema or 'missing'}"
            )
        native_module = {
            "path": native_path.name,
            "sha256": _sha256(native_path),
            "repair_timing_schema": str(
                getattr(lns2_env, "repair_timing_schema", "")
            ),
            "native_semantics_schema": semantics_schema,
        }
    except ImportError:
        pass
    return {
        "sha256": _fingerprint({"files": files, "native_module": native_module}),
        "files": files,
        "native_module": native_module,
    }


def _controller_bundle_path(
    project_root: Path, controller_bundle: str | Path | None
) -> Path:
    value = Path(str(controller_bundle or DEFAULT_CONTROLLER_BUNDLE))
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def resolve_controller_mode(
    project_root: Path,
    controller: str | None,
    controller_bundle: str | Path | None = None,
) -> tuple[str, Path, dict[str, Any] | None]:
    bundle_path = _controller_bundle_path(project_root, controller_bundle)
    loaded = None
    if (bundle_path / "controller_manifest.json").is_file():
        loaded = load_controller_bundle(bundle_path)
    if controller is None:
        mode = (
            str(loaded.manifest.get("default_controller", "official_adaptive"))
            if loaded is not None
            else "official_adaptive"
        )
    else:
        mode = str(controller)
    if mode not in EXECUTABLE_CONTROLLER_MODES:
        raise ValueError(f"unsupported controller mode: {mode}")
    return mode, bundle_path, loaded.manifest if loaded is not None else None


def _number_summary(values: Iterable[float | int]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": numbers[-1],
    }


def closed_loop_dataset_design(
    rows: list[dict[str, Any]],
    split: str,
    registered: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = dict(registered or {})
    mode = str(settings.get("mode", "structured"))
    if mode == "movingai_ood":
        return movingai_ood_dataset_design(rows, split, settings)
    if mode == "balanced_wall_clock":
        errors: list[str] = []
        if any(str(row.get("split")) != split for row in rows):
            errors.append("dataset contains an unexpected split")
        task_ids = [str(row.get("task_id", "")) for row in rows]
        if not all(task_ids) or len(task_ids) != len(set(task_ids)):
            errors.append("dataset task IDs are empty or repeated")
        by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            by_map[str(row["map_id"])].append(row)
        source_counts = collections.Counter(
            str(row.get("source_group", "generated")) for row in rows
        )
        expected_sources = {
            str(name): int(count)
            for name, count in dict(settings.get("source_counts", {})).items()
        }
        if dict(sorted(source_counts.items())) != dict(sorted(expected_sources.items())):
            errors.append("dataset source counts differ from registration")
        if len(rows) != int(settings.get("instance_count", -1)) or len(by_map) != int(
            settings.get("map_count", -1)
        ):
            errors.append("dataset dimensions differ from registration")
        layout_counts = collections.Counter(str(row["layout_mode"]) for row in rows)
        expected_layouts = {
            str(name): int(count)
            for name, count in dict(settings.get("layout_counts", {})).items()
        }
        if expected_layouts and dict(sorted(layout_counts.items())) != dict(
            sorted(expected_layouts.items())
        ):
            errors.append("dataset layout counts differ from registration")
        return {
            "passed": not errors,
            "errors": errors,
            "mode": mode,
            "map_count": len(by_map),
            "task_count": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "layout_counts": dict(sorted(layout_counts.items())),
        }
    expected_tasks = set(
        map(
            str,
            settings.get(
                "task_variants",
                ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100"),
            ),
        )
    )
    tasks_per_map = int(settings.get("tasks_per_map", len(expected_tasks)))
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-closed-loop split")
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(row)
    layout_counts: collections.Counter[str] = collections.Counter()
    for map_id, tasks in sorted(by_map.items()):
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if len(layouts) != 1:
            errors.append(f"{map_id}: inconsistent layout")
            continue
        layout_counts[next(iter(layouts))] += 1
        if (
            {str(row.get("task_variant")) for row in tasks} != expected_tasks
            or len(tasks) != tasks_per_map
        ):
            errors.append(f"{map_id}: incomplete registered task pairing")
        if len({int(row["map_seed"]) for row in tasks}) != 1:
            errors.append(f"{map_id}: inconsistent map seed")
        if len({int(row["task_seed"]) for row in tasks}) != tasks_per_map:
            errors.append(f"{map_id}: repeated task seed")
    expected_layouts = dict(
        settings.get(
            "layout_counts",
            {
                "regular_beltway": 2,
                "compartmentalized": 2,
                "dead_end_aisles": 2,
            },
        )
    )
    expected_layouts = {str(name): int(count) for name, count in expected_layouts.items()}
    if dict(sorted(layout_counts.items())) != expected_layouts:
        errors.append("layout replication does not match the registered design")
    expected_maps = int(settings.get("map_count", sum(expected_layouts.values())))
    expected_rows = expected_maps * tasks_per_map
    if len(rows) != expected_rows or len(by_map) != expected_maps:
        errors.append("dataset dimensions do not match the registered design")
    return {
        "passed": not errors,
        "errors": errors,
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(layout_counts.items())),
    }


def movingai_ood_dataset_design(
    rows: list[dict[str, Any]], split: str, settings: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-OOD split")
    registered_maps = {
        str(row["map_id"]): {
            "layout_family": str(row["layout_family"]),
            "agent_counts": set(map(int, row["agent_counts"])),
        }
        for row in settings.get("maps", [])
    }
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(row)
    if set(by_map) != set(registered_maps):
        errors.append("MovingAI map IDs differ from the registration")
    family_counts: collections.Counter[str] = collections.Counter()
    scenario_indices = set(map(int, settings.get("scenario_indices", [4, 5])))
    for map_id, tasks in sorted(by_map.items()):
        registered = registered_maps.get(map_id)
        if registered is None:
            continue
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if layouts != {registered["layout_family"]}:
            errors.append(f"{map_id}: layout family differs from registration")
        family_counts[registered["layout_family"]] += 1
        observed = {
            (int(str(row.get("scenario_type", "")).rsplit("_", 1)[-1]), int(row["agent_count"]))
            for row in tasks
        }
        expected = {
            (scenario, agents)
            for scenario in scenario_indices
            for agents in registered["agent_counts"]
        }
        if observed != expected or len(tasks) != len(expected):
            errors.append(f"{map_id}: scenario/agent pairing differs from registration")
    expected_families = {
        str(name): int(count)
        for name, count in dict(settings.get("layout_family_counts", {})).items()
    }
    if dict(sorted(family_counts.items())) != dict(sorted(expected_families.items())):
        errors.append("MovingAI layout-family replication differs from registration")
    expected_tasks = int(settings.get("task_count", 0))
    if len(rows) != expected_tasks or len(by_map) != int(settings.get("map_count", 0)):
        errors.append("MovingAI dataset dimensions differ from registration")
    return {
        "passed": not errors,
        "errors": errors,
        "mode": "movingai_ood",
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(family_counts.items())),
    }


def configured_solver_seeds(config: dict[str, Any]) -> tuple[int, ...]:
    values = config.get("solver_seeds")
    seeds = (
        tuple(map(int, values))
        if values is not None
        else (int(config.get("solver_seed", 0)),)
    )
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("solver seeds must be unique non-negative integers")
    return seeds


def _qualification_reuse_fingerprint(run_config: dict[str, Any]) -> str:
    """Fingerprint the state-reset inputs that qualification depends on."""

    configuration = dict(run_config.get("configuration") or {})
    implementation = dict(run_config.get("controller_implementation") or {})
    return _fingerprint(
        {
            "dataset_fingerprint": str(run_config.get("dataset_fingerprint", "")),
            "split": str(configuration.get("split", "")),
            "solver_seeds": list(configured_solver_seeds(configuration)),
            "environment": dict(configuration.get("environment") or {}),
            "seed_isolation": dict(run_config.get("seed_isolation") or {}),
            # Qualification performs environment reset only.  Controller and
            # candidate-generator Python hashes must not invalidate otherwise
            # identical reset evidence; the loaded native producer still must
            # match exactly.
            "native_module": dict(implementation.get("native_module") or {}),
        }
    )


def configured_policies(config: dict[str, Any]) -> tuple[str, ...]:
    policies = tuple(map(str, config.get("policies", POLICIES)))
    if (
        not policies
        or len(policies) != len(set(policies))
        or any(policy not in SUPPORTED_POLICIES for policy in policies)
    ):
        raise ValueError("closed-loop policies are invalid or repeated")
    if "official_adaptive" not in policies or "realized_dynamic" not in policies:
        raise ValueError("closed-loop confirmation requires Adaptive and realized_dynamic")
    return policies


def closed_loop_qualification_report(
    rows: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    isolation: dict[str, Any],
    *,
    formal: bool,
    expected_job_keys: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    solver_seeds = configured_solver_seeds(config)
    available_job_keys = {
        (str(row["task_id"]), int(seed)) for row in rows for seed in solver_seeds
    }
    expected_keys = (
        {(str(task_id), int(seed)) for task_id, seed in expected_job_keys}
        if expected_job_keys is not None
        else available_job_keys
    )
    if not expected_keys or not expected_keys <= available_job_keys:
        raise ValueError("qualification expected-job cohort is empty or invalid")
    expected_solver_seeds = sorted({seed for _task_id, seed in expected_keys})
    indexed = {
        (str(row["task_id"]), int(row.get("solver_seed", solver_seeds[0]))): row
        for row in qualification
    }
    errors = [
        str(row.get("error"))
        for row in qualification
        if str(row.get("status")) != "ok"
    ]
    cohort = []
    thresholds = dict(config["severity_thresholds"])
    for source in rows:
        for solver_seed in solver_seeds:
            if (str(source["task_id"]), solver_seed) not in expected_keys:
                continue
            result = indexed.get((str(source["task_id"]), solver_seed))
            if result is None or str(result.get("status")) != "ok":
                continue
            conflicts = int(result["initial_conflicts"])
            agents = int(source["agent_count"])
            density = conflict_density(conflicts, agents)
            initial_complete = bool(result.get("initial_complete", False))
            reported_initial_feasible = bool(
                result.get("initial_feasible", False)
            )
            expected_initial_feasible = initial_complete and conflicts == 0
            initial_state_consistent = (
                reported_initial_feasible == expected_initial_feasible
            )
            initial_feasible = expected_initial_feasible
            cohort.append(
                {
                    "map_id": str(source["map_id"]),
                    "task_id": str(source["task_id"]),
                    "solver_seed": solver_seed,
                    "layout_mode": str(source["layout_mode"]),
                    "task_variant": str(source["task_variant"]),
                    "agent_count": agents,
                    "initial_conflicts": conflicts,
                    "initial_feasible": initial_feasible,
                    "reported_initial_feasible": reported_initial_feasible,
                    "initial_state_consistent": initial_state_consistent,
                    "initial_complete": initial_complete,
                    "conflict_density": density,
                    "conflict_severity": conflict_severity(density, thresholds),
                    "state_fingerprint": str(result["state_fingerprint"]),
                }
            )
    nonzero = [row for row in cohort if int(row["initial_conflicts"]) > 0]
    by_layout = collections.Counter(str(row["layout_mode"]) for row in nonzero)
    by_agent_band = collections.Counter(
        "low_mid" if int(row["agent_count"]) <= 200 else "high"
        for row in nonzero
    )
    active_maps = sorted({str(row["map_id"]) for row in nonzero})
    by_solver_seed = collections.Counter(int(row["solver_seed"]) for row in nonzero)
    registered_layouts = tuple(
        sorted(
            map(
                str,
                dict(design.get("layout_counts") or {}).keys()
                or {str(row["layout_mode"]) for row in rows},
            )
        )
    )
    fingerprints_by_seed = {
        seed: tuple(
            str(row["state_fingerprint"])
            for row in sorted(
                (item for item in cohort if int(item["solver_seed"]) == seed),
                key=lambda item: str(item["task_id"]),
            )
        )
        for seed in expected_solver_seeds
    }
    duplicate_seed_streams = [
        [left, right]
        for left, right in itertools.combinations(expected_solver_seeds, 2)
        if fingerprints_by_seed[left] == fingerprints_by_seed[right]
    ]
    incomplete_resets = [
        [str(row["task_id"]), int(row["solver_seed"])]
        for row in cohort
        if not bool(row["initial_complete"])
    ]
    inconsistent_initial_states = [
        [str(row["task_id"]), int(row["solver_seed"])]
        for row in cohort
        if not bool(row["initial_state_consistent"])
    ]
    settings = dict(config["qualification"])
    qualification_mode = str(settings.get("mode", "structured"))
    enforce_registered_thresholds = formal or bool(
        settings.get("enforce_registered_thresholds", False)
    )
    minimum_by_agent_band = {
        str(name): int(value)
        for name, value in dict(
            settings.get("minimum_nonzero_states_per_agent_band", {})
        ).items()
    }
    if (
        not set(minimum_by_agent_band) <= {"low_mid", "high"}
        or any(value < 0 for value in minimum_by_agent_band.values())
    ):
        raise ValueError(
            "qualification agent-band thresholds must be non-negative "
            "low_mid/high counts"
        )
    if formal and qualification_mode == "movingai_ood":
        required_families = set(map(str, settings["required_layout_families"]))
        active_families = {
            str(row["layout_mode"]) for row in nonzero
        }
        sample_gates = {
            "minimum_nonzero_states": len(nonzero)
            >= int(settings["minimum_nonzero_states"]),
            "minimum_active_maps": len(active_maps)
            >= int(settings["minimum_active_maps"]),
            "required_layout_families_active": required_families.issubset(active_families),
        }
    else:
        sample_gates = (
        {
            "minimum_nonzero_states": len(nonzero) >= int(settings["minimum_nonzero_states"]),
            "minimum_nonzero_per_layout": all(
                by_layout.get(layout, 0) >= int(settings["minimum_nonzero_states_per_layout"])
                for layout in registered_layouts
            ),
            "minimum_active_maps": len(active_maps) >= int(settings["minimum_active_maps"]),
            "minimum_nonzero_per_solver_seed": all(
                by_solver_seed.get(seed, 0)
                >= int(settings.get("minimum_nonzero_states_per_solver_seed", 0))
                for seed in solver_seeds
            ),
            "minimum_nonzero_per_agent_band": all(
                by_agent_band.get(name, 0) >= minimum
                for name, minimum in minimum_by_agent_band.items()
            ),
        }
        if enforce_registered_thresholds
        else {
            "minimum_nonzero_states": bool(nonzero),
            "minimum_nonzero_per_layout": True,
            "minimum_active_maps": True,
            "minimum_nonzero_per_solver_seed": True,
            "minimum_nonzero_per_agent_band": True,
        }
        )
    gates = {
        "dataset_design": (
            bool(design["passed"]) if enforce_registered_thresholds else True
        ),
        "seed_isolation": bool(isolation["passed"]),
        "all_resets_valid": (
            len(cohort) == len(expected_keys)
            and not errors
            and not incomplete_resets
            and not inconsistent_initial_states
        ),
        "distinct_solver_seed_trajectories": not duplicate_seed_streams,
        **sample_gates,
    }
    grouped = {}
    for field in (
        "layout_mode",
        "task_variant",
        "agent_count",
        "conflict_severity",
        "solver_seed",
    ):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in cohort:
            groups[str(row[field])].append(row)
        grouped[field] = {
            name: {
                "task_count": len(group),
                "initial_feasible_count": sum(
                    bool(item["initial_feasible"]) for item in group
                ),
                "conflicts": _number_summary(item["initial_conflicts"] for item in group),
                "conflict_density": _number_summary(item["conflict_density"] for item in group),
            }
            for name, group in sorted(groups.items())
        }
    passed = all(gates.values())
    return {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "passed": passed,
        "decision": "eligible_for_closed_loop" if passed else "inconclusive_do_not_resample",
        "gates": gates,
        "valid_count": len(cohort),
        "expected_reset_count": len(expected_keys),
        "solver_seeds": expected_solver_seeds,
        "registered_solver_seeds": list(solver_seeds),
        "initial_feasible_count": sum(
            bool(item["initial_feasible"]) for item in cohort
        ),
        "incomplete_reset_count": len(incomplete_resets),
        "incomplete_reset_job_keys": incomplete_resets,
        "inconsistent_initial_state_count": len(inconsistent_initial_states),
        "inconsistent_initial_state_job_keys": inconsistent_initial_states,
        "nonzero_state_count": len(nonzero),
        "registered_layout_modes": list(registered_layouts),
        "nonzero_by_layout": dict(sorted(by_layout.items())),
        "nonzero_by_solver_seed": {
            str(seed): by_solver_seed.get(seed, 0) for seed in solver_seeds
        },
        "nonzero_by_agent_band": {
            name: by_agent_band.get(name, 0) for name in ("low_mid", "high")
        },
        "duplicate_solver_seed_trajectories": duplicate_seed_streams,
        "active_map_count": len(active_maps),
        "active_maps": active_maps,
        "repairable_task_ids": sorted({str(row["task_id"]) for row in nonzero}),
        "repairable_episode_keys": sorted(
            [str(row["task_id"]), int(row["solver_seed"])] for row in nonzero
        ),
        "zero_conflict_task_ids": sorted(
            {str(row["task_id"]) for row in cohort if int(row["initial_conflicts"]) == 0}
        ),
        "severity_thresholds": thresholds,
        "natural_distribution": {
            "conflicts": _number_summary(row["initial_conflicts"] for row in cohort),
            "conflict_density": _number_summary(row["conflict_density"] for row in cohort),
            "severity_counts": dict(
                sorted(collections.Counter(str(row["conflict_severity"]) for row in cohort).items())
            ),
            "grouped": grouped,
            "tasks": sorted(
                cohort, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))
            ),
        },
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": isolation,
    }


def _emit(stream: Any, row: dict[str, Any]) -> float:
    started = time.perf_counter()
    stream.write(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()
    return time.perf_counter() - started


def _pool_runtime_modes(value: dict[str, Any] | None) -> tuple[bool, bool]:
    config = dict(value or {})
    slotpool = config.get("pool_id") == "stride-slotpool-v1"
    guardpool = bool(slotpool and config.get("stall_guard"))
    return slotpool, guardpool


def _closed_loop_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    policy = str(job["policy"])
    solver_seed = int(job["solver_seed"])
    episode_id = _episode_id(row, solver_seed, policy)
    output_root = Path(job["output_root"])
    trace_format = str(job.get("trace_format", TRACE_FORMAT_DELTA_GZIP_V2))
    if trace_format not in TRACE_FORMATS:
        raise ValueError(f"unsupported trace format: {trace_format}")
    storage_fp = str(job.get("storage_fingerprint", storage_fingerprint(trace_format)))
    if storage_fp != storage_fingerprint(trace_format):
        raise ValueError("trace storage fingerprint does not match the selected format")
    trace_path = (
        output_root
        / "episodes"
        / str(row["split"])
        / policy
        / f"{episode_id}{trace_suffix(trace_format)}"
    )
    partial_path = partial_trace_path(trace_path)
    relative_trace = trace_path.relative_to(output_root).as_posix()
    if job["resume"]:
        validated = _valid_episode_trace(
            trace_path,
            job["run_fingerprint"],
            expected_episode_id=episode_id,
            expected_policy=policy,
            expected_solver_seed=solver_seed,
            metric_iteration_budget=(
                int(job["metric_iteration_budget"])
                if job.get("metric_iteration_budget") is not None
                else None
            ),
        )
        if validated is not None:
            metadata = trace_file_metadata(trace_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "schema": CLOSED_LOOP_SCHEMA,
                "episode_id": episode_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "layout_mode": row["layout_mode"],
                "task_variant": row.get("task_variant"),
                "agent_count": int(row["agent_count"]),
                "solver_seed": solver_seed,
                "policy": policy,
                "trace_file": relative_trace,
                "trace_format": trace_format,
                "storage_fingerprint": storage_fp,
                **metadata,
                "trace_event_count": int(validated["event_count"]),
                "initial_state_ref": validated.get("initial_state_ref"),
                "status": "resumed",
                "summary": validated["summary"],
                "error": None,
            }
            previous = job.get("existing_manifest_row")
            if (
                isinstance(previous, dict)
                and str(previous.get("trace_sha256"))
                == str(result.get("trace_sha256"))
                and isinstance(previous.get("episode_finalization_timings"), dict)
            ):
                result["episode_finalization_timings"] = dict(
                    previous["episode_finalization_timings"]
                )
                return result
            if not bool(job.get("require_finalization_timings", False)):
                return result
    episode_override = dict(job.get("episode_override") or {})
    initial_restore = dict(episode_override.get("initial_restore") or {})
    forced_first_action = dict(episode_override.get("forced_first_action") or {})
    bounded_native_retry = dict(
        episode_override.get("bounded_native_retry") or {}
    )
    failure_informed_rescue = dict(
        episode_override.get("failure_informed_rescue") or {}
    )
    signature_scoped_rescue = dict(
        episode_override.get("signature_scoped_rescue") or {}
    )
    pp_replay_seed_salt = episode_override.get("pp_replay_seed_salt")
    if pp_replay_seed_salt is not None and (
        not isinstance(pp_replay_seed_salt, str) or not pp_replay_seed_salt
    ):
        raise ValueError("pp_replay_seed_salt must be a non-empty string")
    repair_seed_policy = validate_repair_seed_policy(
        job.get("repair_seed_policy")
    )
    source_state: dict[str, Any] | None = None
    source_trace_path: Path | None = None
    if initial_restore:
        source_state, source_trace_path = target_state_from_trace(
            Path(str(initial_restore["collection_root"])),
            dict(initial_restore["manifest"]),
            decision_index=int(initial_restore["decision_index"]),
            expected_fingerprint=str(initial_restore["expected_fingerprint"]),
        )
        expected_repair = str(initial_restore["repair_structure_fingerprint"])
        if repair_structure_fingerprint(source_state) != expected_repair:
            raise ValueError("episode override source repair fingerprint changed")
        if int(source_state["num_of_colliding_pairs"]) != int(
            initial_restore["expected_conflicts"]
        ):
            raise ValueError("episode override source conflict count changed")
    if forced_first_action and not initial_restore:
        raise ValueError("forced first action requires an initial restored state")
    if bounded_native_retry and not initial_restore:
        raise ValueError("bounded native retry requires an initial restored state")
    if failure_informed_rescue and not initial_restore:
        raise ValueError(
            "failure-informed rescue requires an initial restored state"
        )
    if signature_scoped_rescue and not initial_restore:
        raise ValueError(
            "signature-scoped rescue requires an initial restored state"
        )
    enabled_repair_overrides = sum(
        bool(value)
        for value in (
            bounded_native_retry,
            failure_informed_rescue,
            signature_scoped_rescue,
        )
    )
    if enabled_repair_overrides > 1:
        raise ValueError(
            "bounded retry, failure-informed rescue, and signature-scoped rescue "
            "are mutually exclusive"
        )
    if forced_first_action:
        if (
            str(forced_first_action.get("mode")) != "explicit_neighborhood"
            or not list(forced_first_action.get("agents") or ())
            or int(forced_first_action.get("pp_random_seed", -1)) < 0
        ):
            raise ValueError("invalid forced first action override")
    bundle = None
    learned_selector: Selector | None = None
    controller_mode = str(job.get("controller", "official_adaptive"))
    if controller_mode not in EXECUTABLE_CONTROLLER_MODES:
        raise ValueError(f"unsupported controller mode: {controller_mode}")
    feature_backend = str(job.get("feature_backend", "auto"))
    requested_controller_runtime = str(job.get("controller_runtime", "reference"))
    if requested_controller_runtime not in CONTROLLER_RUNTIMES:
        raise ValueError(
            f"unsupported controller runtime: {requested_controller_runtime}"
        )
    verification_profile = str(job.get("verification_profile", "audit"))
    if verification_profile not in VERIFICATION_PROFILES:
        raise ValueError(f"unsupported verification profile: {verification_profile}")
    v3_s3_bundle: V3S3Bundle | None = None
    if controller_mode == "v3-s3":
        raw_v3_s3_bundle = job.get("v3_s3_bundle")
        if raw_v3_s3_bundle is None:
            raise ValueError("v3-s3 requires a frozen v3-S3 bundle")
        v3_s3_bundle = load_v3_s3_bundle(raw_v3_s3_bundle)
        if str(v3_s3_bundle.manifest.get("schema")) != V3_S3_BUNDLE_SCHEMA:
            raise ValueError("v3-s3 requires a sequence-aware controller bundle")
    runtime_models: dict[str, Any] = {}
    runtime_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    source_models: dict[str, Any] = {}
    source_model_provenance: dict[str, dict[str, Any]] = {}
    diagnostic_shadow_selectors: dict[str, Selector] = {}
    diagnostic_shadow_ranges: dict[
        str, dict[str, dict[str, tuple[float, float]]]
    ] = {}
    diagnostic_shadow_fallback_thresholds: dict[str, float] = {}
    structpool_runtime_config = dict(
        dict(job.get("proposal") or {}).get("structpool") or {}
    )
    hybridstructpool_runtime_config = dict(
        dict(job.get("proposal") or {}).get("hybridstructpool") or {}
    )
    validate_any_hybridstructpool_augmentation(
        hybridstructpool_runtime_config or None
    )
    if hybridstructpool_runtime_config and (
        policy != "realized_dynamic" or controller_mode != "v2-full"
    ):
        raise ValueError(
            "HybridStructPool requires a realized_dynamic v2-full episode"
        )
    slotpool_runtime_enabled, guardpool_runtime_enabled = _pool_runtime_modes(
        structpool_runtime_config
    )
    slotpool_model_payload: dict[str, Any] | None = None
    if slotpool_runtime_enabled:
        if policy != "realized_dynamic" or controller_mode != "v2-full":
            raise ValueError(
                "SlotPool/GuardPool requires a realized_dynamic v2-full episode"
            )
        model_registration = dict(structpool_runtime_config["slotpool_model"])
        model_path = Path(str(model_registration["path"]))
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        slotpool_model_payload = load_slotpool_model(
            model_path,
            expected_sha256=str(model_registration["sha256"]),
        )
    if policy in LEARNED_POLICIES:
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
        if controller_mode == "official_adaptive":
            runtime_models = bundle.models
            runtime_ranges = bundle.ranges
        else:
            controller_path = Path(str(job["controller_bundle"]))
            if (controller_path / "controller_manifest.json").is_file():
                compact_bundle = load_controller_bundle(controller_path)
                runtime_models = compact_bundle.main_models
                runtime_ranges = compact_bundle.main_ranges
                if bool(job.get("feature_shadow_validation", False)):
                    source_model, provenance = _matching_source_model(
                        controller_path=controller_path,
                        controller_bundle=compact_bundle,
                        frozen_bundle=bundle,
                        model_registration=dict(job["model_registration"]),
                        profile=policy,
                    )
                    source_models[policy] = source_model
                    source_model_provenance[policy] = provenance
            else:
                runtime_models = {
                    name: compact_runtime_model(model)
                    for name, model in bundle.models.items()
                }
                runtime_ranges = {
                    name: {
                        feature: bundle.ranges[name][feature]
                        for feature in model.base_feature_names
                    }
                    for name, model in runtime_models.items()
                }
                if bool(job.get("feature_shadow_validation", False)):
                    source_models = dict(bundle.models)
                    source_model_provenance[policy] = {
                        "kind": "registered_frozen_model",
                        "profile": policy,
                        "source_model_sha256": str(
                            dict(job["model_registration"]["model_sha256"])[policy]
                        ).lower(),
                    }
        if controller_mode in PAIRWISE_CONTROLLER_MODES:
            learned_selector = PairwiseV2Selector(controller_mode, runtime_models)
        elif controller_mode in {"stride-guardrank-v1", "stride-maprank-v1"}:
            learned_selector = load_selector(controller_mode, controller_path)
        raw_diagnostic_shadows = dict(job.get("diagnostic_shadow_bundles") or {})
        if raw_diagnostic_shadows:
            if controller_mode != "v2-full" or policy != "realized_dynamic":
                raise ValueError(
                    "diagnostic shadows require a realized_dynamic v2-full episode"
                )
            for shadow_id, shadow_path in sorted(raw_diagnostic_shadows.items()):
                shadow_bundle = load_controller_bundle(str(shadow_path))
                if (
                    str(shadow_bundle.manifest.get("controller_id")) != shadow_id
                    or shadow_bundle.manifest.get("scientific_status")
                    != "diagnostic_only"
                    or shadow_bundle.manifest.get("default_replacement_allowed")
                    is not False
                ):
                    raise ValueError(
                        f"invalid diagnostic shadow bundle: {shadow_id}"
                    )
                diagnostic_shadow_selectors[shadow_id] = load_selector(
                    shadow_id, shadow_path
                )
                diagnostic_shadow_ranges[shadow_id] = shadow_bundle.main_ranges
                diagnostic_shadow_fallback_thresholds[shadow_id] = float(
                    shadow_bundle.manifest["fallback_rules"][
                        "maximum_outside_feature_fraction"
                    ]
                )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    started_wall = time.perf_counter()
    try:
        destroy_strategy = POLICY_DESTROY_STRATEGIES.get(policy, "Adaptive")
        environment_started = time.perf_counter()
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], destroy_strategy
        )
        optimized_runtime_available = bool(
            callable(getattr(environment, "propose_batch_compact", None))
            and feature_backend in {"auto", "native"}
            and _native_vector_function() is not None
        )
        if (
            requested_controller_runtime == "optimized"
            and policy in LEARNED_POLICIES
            and controller_mode != "official_adaptive"
            and not optimized_runtime_available
        ):
            raise RuntimeError(
                "optimized controller runtime requires compact proposals and dense native features"
            )
        controller_runtime = (
            "optimized"
            if policy in LEARNED_POLICIES
            and controller_mode != "official_adaptive"
            and (
                requested_controller_runtime == "optimized"
                or (
                    requested_controller_runtime == "auto"
                    and optimized_runtime_available
                )
            )
            else "reference"
        )
        environment_construct_seconds = time.perf_counter() - environment_started
        initial_state_ref: str | None = None
        with open_trace_text(partial_path, "w") as stream:
            # TTF and its live wall budget deliberately start immediately before
            # reset. Environment construction remains separately reported and
            # must not consume a controller's registered solve budget.
            ttf_started_wall = time.perf_counter()
            reset_started = time.perf_counter()
            if source_state is None:
                state = _plain(environment.reset(seed=solver_seed))
            else:
                source_agents = sorted(
                    source_state.get("agents", []), key=lambda value: int(value["id"])
                )
                if [int(value["id"]) for value in source_agents] != list(
                    range(len(source_agents))
                ):
                    raise ValueError("episode override source has non-contiguous agents")
                source_paths = [
                    list(map(int, value.get("path", []))) for value in source_agents
                ]
                if not source_paths or any(not path for path in source_paths):
                    raise ValueError("episode override source contains an empty path")
                state = _plain(
                    environment.reset_paths(
                        source_paths,
                        seed=int(initial_restore["restore_seed"]),
                    )
                )
                if repair_structure_fingerprint(state) != str(
                    initial_restore["repair_structure_fingerprint"]
                ):
                    raise RuntimeError(
                        "episode override restored repair structure differs from source"
                    )
                if int(state["num_of_colliding_pairs"]) != int(
                    initial_restore["expected_conflicts"]
                ):
                    raise RuntimeError(
                        "episode override restored conflict count differs from source"
                    )
            reset_completed_wall = time.perf_counter()
            reset_wall_seconds = reset_completed_wall - reset_started
            initial_state_elapsed_seconds = reset_completed_wall - ttf_started_wall
            reset_timing_getter = getattr(environment, "get_last_reset_timings", None)
            reset_timings = (
                _plain(reset_timing_getter())
                if callable(reset_timing_getter)
                else {"reset_total_seconds": reset_wall_seconds}
            )
            initial_fingerprint_started = time.perf_counter()
            initial_fingerprint = state_fingerprint(state)
            # The post-step fingerprint is also the next iteration's before
            # fingerprint.  Keep the exact value instead of rescanning every
            # agent path at the top of the next loop.  This changes neither the
            # fingerprint definition nor any controller/random-seed semantics.
            current_state_fingerprint = initial_fingerprint
            episode_repair_seed_stream = (
                EpisodeRepairSeedStream.from_episode(
                    task_id=str(row["task_id"]),
                    solver_seed=solver_seed,
                    episode_id=episode_id,
                )
                if repair_seed_policy == "episode_stream"
                else None
            )
            initial_fingerprint_seconds = (
                time.perf_counter() - initial_fingerprint_started
            )
            conflicts = [int(state["num_of_colliding_pairs"])]
            transition_elapsed_seconds: list[float] = []
            transition_trace_write_seconds: list[float] = []
            budget_final_conflicts = conflicts[0]
            budget_final_sum_of_costs = int(state["sum_of_costs"])
            budget_final_low_level = dict(state["low_level"])
            repair_iterations_within_budget = 0
            native_retry_tracker = (
                BoundedNativeRetryTracker.from_spec(state, bounded_native_retry)
                if bounded_native_retry
                else None
            )
            failure_rescue_tracker = (
                FailureInformedRescueTracker.from_spec(failure_informed_rescue)
                if failure_informed_rescue
                else None
            )
            signature_rescue_tracker = (
                SignatureScopedRescueTracker.from_spec(
                    state, signature_scoped_rescue
                )
                if signature_scoped_rescue
                else None
            )
            episode_hybrid_runtime = dict(
                dict(job.get("proposal") or {}).get("hybridstructpool") or {}
            )
            exact_rollback_guard_config = dict(
                episode_hybrid_runtime.get("exact_rollback_guard") or {}
            )
            exact_rollback_guard_id = str(
                exact_rollback_guard_config.get("guard_id") or ""
            )
            if exact_rollback_guard_id == ROLLBACK_AWARE_SELECTION_ID:
                exact_rollback_guard = ExactRollbackCandidateGuard(
                    exact_rollback_limit=int(
                        exact_rollback_guard_config["exact_rollback_limit"]
                    )
                )
            elif exact_rollback_guard_id == OVERALL_ROLLBACK_SELECTION_ID:
                exact_rollback_guard = ExactRollbackStateGuard(
                    exact_rollback_limit=int(
                        exact_rollback_guard_config["exact_rollback_limit"]
                    )
                )
            elif exact_rollback_guard_config:
                raise ValueError("unsupported exact rollback guard identity")
            else:
                exact_rollback_guard = None
            state_bounded_rollback_guard = isinstance(
                exact_rollback_guard, ExactRollbackStateGuard
            )
            rollback_guard_trace_key = (
                "exact_rollback_state_guard"
                if state_bounded_rollback_guard
                else "exact_rollback_candidate_guard"
            )
            repair_state_cache_mode = exact_rollback_guard_config.get(
                "repair_state_cache"
            )
            repair_state_cache_enabled = bool(
                repair_state_cache_mode is True
                or repair_state_cache_mode == "pre_budget_only"
            )
            if exact_rollback_guard is not None and (
                native_retry_tracker is not None
                or failure_rescue_tracker is not None
                or signature_rescue_tracker is not None
                or forced_first_action
                or controller_mode != "v2-full"
                or policy != "realized_dynamic"
            ):
                raise ValueError(
                    "rollback-aware selection requires an unmodified realized V2 "
                    "decision stream without retry, rescue, or forced actions"
                )
            rollback_selection_cache: dict[str, Any] | None = None
            initial_event = {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "initial",
                "episode_id": episode_id,
                "policy": policy,
                "solver_seed": solver_seed,
                "repair_seed_policy": repair_seed_policy,
                "repair_seed_stream_root": (
                    episode_repair_seed_stream.root_seed
                    if episode_repair_seed_stream is not None
                    else None
                ),
                "state_fingerprint": initial_fingerprint,
                "state": state,
                "episode_override": (
                    {
                        "schema": str(episode_override.get("schema", "")),
                        "state_id": str(episode_override.get("state_id", "")),
                        "source_trace_file": (
                            str(source_trace_path) if source_trace_path is not None else None
                        ),
                        "source_decision_index": (
                            int(initial_restore["decision_index"])
                            if initial_restore
                            else None
                        ),
                        "source_full_fingerprint": (
                            str(initial_restore["expected_fingerprint"])
                            if initial_restore
                            else None
                        ),
                        "source_repair_fingerprint": (
                            str(initial_restore["repair_structure_fingerprint"])
                            if initial_restore
                            else None
                        ),
                        "forced_first_action": bool(forced_first_action),
                        "bounded_native_retry": (
                            native_retry_tracker.summary()
                            if native_retry_tracker is not None
                            else None
                        ),
                        "failure_informed_rescue": (
                            failure_rescue_tracker.summary()
                            if failure_rescue_tracker is not None
                            else None
                        ),
                        "signature_scoped_rescue": (
                            signature_rescue_tracker.summary()
                            if signature_rescue_tracker is not None
                            else None
                        ),
                    }
                    if episode_override
                    else None
                ),
            }
            if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                initial_event, initial_state_ref = encode_initial_event(
                    initial_event, state, output_root
                )
            initial_trace_write_seconds = _emit(stream, initial_event)
            controller_totals = collections.Counter()
            selected_sizes: collections.Counter[int] = collections.Counter()
            selected_families: collections.Counter[str] = collections.Counter()
            invalid_actions = 0
            fingerprint_mismatches = 0
            external_timeout = False
            total_repair_wall_seconds = 0.0
            max_decisions = int(job.get("max_decisions") or 0)
            stopping_rule = str(job.get("stopping_rule", "historical"))
            if stopping_rule not in STOPPING_RULES:
                raise ValueError(f"unsupported stopping rule: {stopping_rule}")
            raw_safety_max_decisions = job.get(
                "safety_max_decisions", WALL_CLOCK_SAFETY_MAX_DECISIONS
            )
            safety_max_decisions = (
                None
                if raw_safety_max_decisions is None
                else int(raw_safety_max_decisions)
            )
            raw_wall_budget = job.get("wall_time_budget_seconds")
            wall_budget = (
                None if raw_wall_budget is None else float(raw_wall_budget)
            )
            static_grid = (
                analyze_static_grid(state)
                if policy in LEARNED_POLICIES
                and controller_mode == "official_adaptive"
                else None
            )
            required_model_features = (
                set(runtime_models[policy].base_feature_names)
                if policy in LEARNED_POLICIES and v3_s3_bundle is None
                else set()
            )
            if learned_selector is not None:
                required_model_features.update(
                    _selector_required_model_features(learned_selector, policy)
                )
            if policy in LEARNED_POLICIES:
                for shadow_selector in diagnostic_shadow_selectors.values():
                    required_model_features.update(
                        _selector_required_model_features(shadow_selector, policy)
                    )
            if v3_s3_bundle is not None and policy == "realized_dynamic":
                required_model_features.update(
                    set(v3_s3_bundle.required_feature_names)
                    & set(PROFILE_FEATURE_NAMES["realized_dynamic"])
                )
            if slotpool_runtime_enabled:
                required_model_features.update(
                    PROFILE_FEATURE_NAMES["realized_dynamic"]
                )

            def make_feature_engine(current_state: dict[str, Any]) -> OnlineFeatureEngine:
                return OnlineFeatureEngine(
                    current_state,
                    backend=feature_backend,
                    shadow_validation=bool(job.get("feature_shadow_validation", False)),
                    required_features={
                        policy: required_model_features,
                    },
                    dense_output=controller_runtime == "optimized",
                )

            feature_engine = (
                make_feature_engine(state)
                if policy in LEARNED_POLICIES
                and controller_mode != "official_adaptive"
                else None
            )
            topology_analysis_cache: TopologyAnalysisCache | None = None
            topology_pending_changed_agents: set[int] = set()
            pending_changed_agents: set[int] = set()
            no_progress_streak = 0
            guard_was_active = False
            hybrid_stall_guard_was_active = False
            previous_route: str | None = None
            v3_s3_selector = (
                V3S3Selector(v3_s3_bundle)
                if v3_s3_bundle is not None
                and controller_mode == "v3-s3"
                and policy == "realized_dynamic"
                else None
            )
            v3_s3_state = (
                v3_s3_selector.state if v3_s3_selector is not None else None
            )
            v3_s3_history: list[dict[str, Any]] = []
            stateful_cache: dict[str, Any] | None = None
            controller_stalled = False
            while not bool(state["done"]) and (
                max_decisions <= 0 or len(conflicts) - 1 < max_decisions
            ):
                if (
                    safety_max_decisions is not None
                    and len(conflicts) - 1 >= safety_max_decisions
                ):
                    raise ClosedLoopExecutionError(
                        "safety_iteration_limit",
                        "wall-clock execution reached its diagnostic safety limit",
                    )
                if (
                    wall_budget is not None
                    and time.perf_counter() - ttf_started_wall >= wall_budget
                ):
                    external_timeout = True
                    break
                iteration_started = time.perf_counter()
                before = state
                before_fingerprint_started = time.perf_counter()
                before_hash = current_state_fingerprint
                if v3_s3_state is not None or exact_rollback_guard is not None:
                    before_repair_hash = repair_structure_fingerprint(before)
                else:
                    before_repair_hash = before_hash
                before_fingerprint_seconds = (
                    time.perf_counter() - before_fingerprint_started
                )
                decision_index = len(conflicts) - 1
                controller: dict[str, Any] = {}
                rollback_guard_selected_candidate_id: str | None = None
                rollback_guard_bannable_candidate_ids: set[str] = set()
                rescue_override = (
                    failure_rescue_tracker.action_for_decision(
                        decision_index, state
                    )
                    if failure_rescue_tracker is not None
                    else None
                )
                signature_rescue_override = (
                    signature_rescue_tracker.action_for_decision(
                        decision_index, state
                    )
                    if signature_rescue_tracker is not None
                    else None
                )
                if rescue_override is not None and signature_rescue_override is not None:
                    raise ValueError("multiple deferred rescue actions were scheduled")
                if signature_rescue_override is not None:
                    rescue_override = signature_rescue_override
                signature_scoped_action = signature_rescue_override is not None
                force_this_action = bool(
                    forced_first_action and decision_index == 0
                )
                route = "model" if policy in LEARNED_POLICIES else "official_adaptive"
                route_started = time.perf_counter()
                pre_step_orchestration_seconds = route_started - iteration_started
                if rescue_override is not None:
                    route = "model"
                    rescue_agents = list(map(int, rescue_override["agents"]))
                    rescue_seed = int(rescue_override["pp_random_seed"])
                    action = {
                        "mode": "explicit_neighborhood",
                        "agents": rescue_agents,
                        "random_seed": rescue_seed,
                        "pp_random_seed": rescue_seed,
                        "collect_pp_diagnostics": True,
                    }
                    controller_seconds_before_repair = (
                        time.perf_counter() - route_started
                    )
                    rescue_mode = str(rescue_override["mode"])
                    rescue_prefix = (
                        "signature-scoped-rescue"
                        if signature_scoped_action
                        else "failure-informed-rescue"
                    )
                    rescue_family = f"{rescue_prefix}:{rescue_mode}"
                    rescue_candidate_id = _fingerprint(
                        {
                            "state": before_hash,
                            "decision_index": decision_index,
                            "mode": rescue_mode,
                            "agents": rescue_agents,
                        }
                    )
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "forced_first_action": False,
                            "failure_informed_rescue_action": (
                                not signature_scoped_action
                            ),
                            "signature_scoped_rescue_action": signature_scoped_action,
                            "forced_candidate_role": rescue_mode,
                            "selected_candidate_id": rescue_candidate_id,
                            "candidate_pool": [
                                {
                                    "candidate_id": rescue_candidate_id,
                                    "agents": rescue_agents,
                                    "actual_size": len(rescue_agents),
                                    "selection_families": [rescue_family],
                                    "failure_informed_rescue_action": (
                                        not signature_scoped_action
                                    ),
                                    "signature_scoped_rescue_action": (
                                        signature_scoped_action
                                    ),
                                    "selected_blockers": list(
                                        map(
                                            int,
                                            rescue_override.get(
                                                "selected_blockers"
                                            )
                                            or (),
                                        )
                                    ),
                                    "removed_agents": list(
                                        map(
                                            int,
                                            rescue_override.get("removed_agents")
                                            or (),
                                        )
                                    ),
                                    "compact_plan": dict(
                                        rescue_override.get("compact_plan") or {}
                                    ),
                                }
                            ],
                            "controller_seconds_before_repair": (
                                controller_seconds_before_repair
                            ),
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "state_analysis_seconds": 0.0,
                            "proposal_feature_seconds": 0.0,
                            "realized_feature_seconds": 0.0,
                            "ranking_inference_seconds": 0.0,
                            "selection_residual_seconds": (
                                controller_seconds_before_repair
                            ),
                        }
                    )
                    selected_sizes[len(rescue_agents)] += 1
                    selected_families[rescue_family] += 1
                    controller_totals[
                        (
                            "signature_scoped_rescue_action_count"
                            if signature_scoped_action
                            else "failure_informed_rescue_action_count"
                        )
                    ] += 1
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_seconds_before_repair
                    )
                elif force_this_action:
                    route = "model"
                    action = dict(forced_first_action)
                    controller_seconds_before_repair = (
                        time.perf_counter() - route_started
                    )
                    forced_agents = list(map(int, action["agents"]))
                    forced_candidate_id = str(
                        episode_override.get("forced_candidate_id", "")
                    )
                    forced_families = list(
                        map(str, episode_override.get("forced_selection_families") or ())
                    )
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "forced_first_action": True,
                            "forced_candidate_role": str(
                                episode_override.get("forced_candidate_role", "")
                            ),
                            "selected_candidate_id": forced_candidate_id,
                            "candidate_pool": [
                                {
                                    "candidate_id": forced_candidate_id,
                                    "agents": forced_agents,
                                    "actual_size": len(forced_agents),
                                    "selection_families": forced_families,
                                    "forced_first_action": True,
                                }
                            ],
                            "controller_seconds_before_repair": (
                                controller_seconds_before_repair
                            ),
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "state_analysis_seconds": 0.0,
                            "proposal_feature_seconds": 0.0,
                            "realized_feature_seconds": 0.0,
                            "ranking_inference_seconds": 0.0,
                            "selection_residual_seconds": (
                                controller_seconds_before_repair
                            ),
                        }
                    )
                    selected_sizes[len(forced_agents)] += 1
                    for family in forced_families:
                        selected_families[family] += 1
                    controller_totals["forced_first_action_count"] += 1
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_seconds_before_repair
                    )
                elif route == "official_adaptive":
                    action = {"mode": "official"}
                    controller_seconds_before_repair = time.perf_counter() - route_started
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "controller_seconds_before_repair": controller_seconds_before_repair,
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "state_analysis_seconds": 0.0,
                            "proposal_feature_seconds": 0.0,
                            "realized_feature_seconds": 0.0,
                            "ranking_inference_seconds": 0.0,
                            "selection_residual_seconds": controller_seconds_before_repair,
                        }
                    )
                else:
                    proposal_started = time.perf_counter()
                    effective_proposal = dict(job["proposal"])
                    if v3_s3_state is not None:
                        generation_request = (
                            v3_s3_state.candidate_generation_request()
                        )
                        if generation_request["mode"] == "restricted":
                            effective_proposal.update(
                                {
                                    "heuristics": list(
                                        generation_request["heuristics"]
                                    ),
                                    "neighborhood_sizes": list(
                                        generation_request["neighborhood_sizes"]
                                    ),
                                    "candidates_per_family": int(
                                        generation_request[
                                            "candidates_per_family"
                                        ]
                                    ),
                                }
                            )
                    cache_key = (
                        before_repair_hash,
                        _fingerprint(effective_proposal),
                        int(solver_seed),
                    )
                    stateful_controller = v3_s3_state
                    state_bounded_v2_fallback = bool(
                        state_bounded_rollback_guard
                        and exact_rollback_guard is not None
                        and exact_rollback_guard.requires_fresh_v2_fallback(
                            before_repair_hash
                        )
                    )
                    stateful_cache_hit = bool(
                        stateful_controller is not None
                        and stateful_cache is not None
                        and stateful_cache.get("key") == cache_key
                    )
                    rollback_cache_hit = bool(
                        exact_rollback_guard is not None
                        and repair_state_cache_enabled
                        and exact_rollback_guard.cache_reuse_allowed
                        and rollback_selection_cache is not None
                        and rollback_selection_cache.get("key") == cache_key
                    )
                    cache_hit = stateful_cache_hit or rollback_cache_hit
                    active_selection_cache = (
                        stateful_cache if stateful_cache_hit else rollback_selection_cache
                    )
                    state_feature_metrics: dict[str, Any] = {}
                    proposal_feature_metrics = {"proposal_feature_seconds": 0.0}
                    realized_feature_metrics = {"realized_feature_seconds": 0.0}
                    proposal_rows: list[dict[str, Any]] | None = None
                    state_analysis_seconds = 0.0
                    slotpool_raw_candidates: list[dict[str, Any]] | None = None
                    slotpool_raw_candidate_rows: list[dict[str, Any]] | None = None
                    slotpool_retained_raw_indices: list[int] | None = None
                    if cache_hit:
                        assert active_selection_cache is not None
                        candidates = active_selection_cache["candidates"]
                        proposal_metrics = {
                            **dict(active_selection_cache["proposal_metrics"]),
                            "proposal_seconds": 0.0,
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "hybridstructpool_seconds": 0.0,
                            "hybridstructpool_generation_seconds": 0.0,
                            "hybridstructpool_structural_generation_seconds": 0.0,
                            "hybridstructpool_causal_generation_seconds": 0.0,
                            "hybridstructpool_gate_seconds": 0.0,
                            "hybridstructpool_gate_evaluated": False,
                            "hybridstructpool_gate_result_reused": True,
                            "backend": (
                                "v3-s3-cache"
                                if stateful_cache_hit
                                else "repair-state-cache"
                            ),
                            "state_check_backend": "cached-state-fingerprint",
                            "full_state_verified": False,
                            "v3_s3_cache_hit": stateful_cache_hit,
                            "repair_state_cache_hit": rollback_cache_hit,
                            "repair_state_cache_generation_decision_index": int(
                                active_selection_cache["generation_decision_index"]
                            ),
                        }
                        if rollback_cache_hit:
                            # A repair fingerprint deliberately excludes attempt
                            # counters.  The frozen V2 feature schema does not:
                            # state.iteration and cumulative low-level work are
                            # model inputs.  Reuse only the expensive candidate
                            # structure, then recompute features, the V2 anchor,
                            # and Copeland scores for the current decision.
                            feature_engine_created = False
                            if feature_engine is None:
                                feature_engine = make_feature_engine(state)
                                feature_engine_created = True
                            if feature_engine_created:
                                state_feature_metrics = dict(
                                    feature_engine.last_prepare_metrics
                                )
                            else:
                                state_feature_metrics = feature_engine.prepare(
                                    state,
                                    changed_agents=sorted(pending_changed_agents),
                                )
                            pending_changed_agents.clear()
                            (
                                candidate_rows,
                                realized_feature_metrics,
                            ) = feature_engine.realized_rows(
                                candidates, state_hash=before_hash
                            )
                            state_analysis_seconds = float(
                                state_feature_metrics.get(
                                    "state_analysis_seconds", 0.0
                                )
                            ) + float(
                                realized_feature_metrics.get(
                                    "state_analysis_seconds", 0.0
                                )
                            )
                            feature_seconds = state_analysis_seconds + float(
                                realized_feature_metrics.get(
                                    "realized_feature_seconds", 0.0
                                )
                            )
                            v2_base_indices = [
                                index
                                for index, candidate in enumerate(candidates)
                                if "v2_base"
                                in set(
                                    map(
                                        str,
                                        candidate.get(
                                            "hybridstructpool_provenance"
                                        )
                                        or (),
                                    )
                                )
                            ]
                            if not v2_base_indices:
                                raise ClosedLoopExecutionError(
                                    "repair_state_cache_missing_v2_anchor_pool",
                                    "cached HybridStructPool contains no V2 base candidate",
                                )
                            inference_started = time.perf_counter()
                            (
                                v2_anchor_base_index,
                                _v2_anchor_scores,
                                _v2_anchor_margin,
                            ) = score_online_candidates(
                                [candidate_rows[index] for index in v2_base_indices],
                                runtime_models[policy],
                            )
                            proposal_metrics[
                                "hybridstructpool_v2_anchor_candidate_id"
                            ] = str(
                                candidates[
                                    v2_base_indices[v2_anchor_base_index]
                                ]["candidate_id"]
                            )
                            (
                                selected_local_index,
                                scores,
                                margin,
                            ) = score_online_candidates(
                                candidate_rows, runtime_models[policy]
                            )
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )
                            proposal_metrics.update(
                                {
                                    "repair_state_cache_candidate_pool_reused": True,
                                    "repair_state_cache_feature_rows_recomputed": True,
                                    "repair_state_cache_scores_recomputed": True,
                                    "repair_state_cache_v2_anchor_refreshed": True,
                                }
                            )
                        else:
                            candidate_rows = active_selection_cache["candidate_rows"]
                            scores = active_selection_cache["scores"]
                            margin = float(active_selection_cache["margin"])
                            selected_local_index = int(
                                active_selection_cache[
                                    "base_selected_local_index"
                                ]
                            )
                            feature_seconds = 0.0
                            inference_seconds = 0.0
                        if stateful_controller is not None:
                            stateful_controller.note_cache_hit()
                    else:
                        verification_mode = str(
                            job.get("proposal_state_verification", "always")
                        )
                        verify_full_state = verification_mode == "always" or (
                            verification_mode == "sampled"
                            and decision_index % 20 == 0
                        )
                        topology_state_analysis = None
                        topology_state_analysis_seconds = 0.0
                        topology_prepared_native_analysis = None
                        topology_runtime = dict(
                            effective_proposal.get("topology_boundary") or {}
                        )
                        structpool_runtime = dict(
                            effective_proposal.get("structpool") or {}
                        )
                        hybridstructpool_runtime = dict(
                            effective_proposal.get("hybridstructpool") or {}
                        )
                        hybrid_stall_guard_config = dict(
                            hybridstructpool_runtime.get("stall_guard") or {}
                        )
                        hybrid_stall_guard_active_for_decision = bool(
                            hybrid_stall_guard_config
                            and no_progress_streak
                            >= int(
                                hybrid_stall_guard_config[
                                    "no_progress_limit"
                                ]
                            )
                        )
                        hybrid_stall_guard_triggered_now = bool(
                            hybrid_stall_guard_active_for_decision
                            and not hybrid_stall_guard_was_active
                        )
                        hybrid_stall_guard_released_now = bool(
                            hybrid_stall_guard_was_active
                            and not hybrid_stall_guard_active_for_decision
                        )
                        hybrid_stall_guard_was_active = (
                            hybrid_stall_guard_active_for_decision
                        )
                        guard_config = dict(
                            structpool_runtime.get("stall_guard") or {}
                        )
                        guard_active_for_decision = bool(
                            guard_config
                            and no_progress_streak
                            >= int(guard_config["no_progress_limit"])
                        )
                        guard_triggered_now = bool(
                            guard_active_for_decision
                            and not guard_was_active
                        )
                        guard_released_now = bool(
                            guard_was_active and not guard_active_for_decision
                        )
                        conflict_signature = (
                            _fingerprint(
                                {
                                    "conflict_pairs": int(
                                        state["num_of_colliding_pairs"]
                                    ),
                                    "conflict_edges": sorted(
                                        tuple(sorted(map(int, edge)))
                                        for edge in state.get("conflict_edges", [])
                                    ),
                                }
                            )
                            if guardpool_runtime_enabled
                            else None
                        )
                        guard_was_active = guard_active_for_decision
                        structpool_gate_result = (
                            structpool_high_stress_gate(state, structpool_runtime)
                            if structpool_runtime
                            else None
                        )
                        if (
                            structpool_gate_result is not None
                            and guard_active_for_decision
                        ):
                            structpool_gate_result = {
                                **structpool_gate_result,
                                "passed": False,
                                "reason": "stall_guard_active",
                            }
                        structpool_gate_passed = bool(
                            structpool_gate_result
                            and structpool_gate_result["passed"]
                        )
                        hybridstructpool_gate_result = None
                        if hybridstructpool_runtime:
                            if hybrid_stall_guard_active_for_decision:
                                hybridstructpool_gate_result = {
                                    "passed": False,
                                    "reason": "plateau_guard_active",
                                    "seconds": 0.0,
                                    "evaluated": False,
                                    "gate_id": str(
                                        hybridstructpool_runtime[
                                            "activation_gate"
                                        ]["gate_id"]
                                    ),
                                    "agent_count": len(
                                        state.get("agents", [])
                                    ),
                                    "conflict_pair_count": int(
                                        state["num_of_colliding_pairs"]
                                    ),
                                    "active_conflict_agent_count": 0,
                                    "largest_conflict_component_size": 0,
                                    "pre_guard_passed": None,
                                }
                            elif state_bounded_v2_fallback:
                                hybridstructpool_gate_result = {
                                    "passed": False,
                                    "reason": "state_exact_rollback_budget_exhausted",
                                    "seconds": 0.0,
                                    "evaluated": False,
                                    "gate_id": str(
                                        hybridstructpool_runtime["activation_gate"][
                                            "gate_id"
                                        ]
                                    ),
                                    "agent_count": len(state.get("agents", [])),
                                    "conflict_pair_count": int(
                                        state["num_of_colliding_pairs"]
                                    ),
                                    "active_conflict_agent_count": 0,
                                    "largest_conflict_component_size": 0,
                                    "pre_guard_passed": None,
                                }
                            else:
                                hybrid_pool_id = str(
                                    hybridstructpool_runtime.get("pool_id")
                                )
                                if (
                                    hybrid_pool_id
                                    == STRUCTSHELL_SINGLE_FAMILY_POOL_ID
                                ):
                                    hybridstructpool_gate_result = (
                                        structshell_single_family_ablation_gate(
                                            state, hybridstructpool_runtime
                                        )
                                    )
                                elif hybrid_pool_id == STRUCTSHELL_DUAL16_POOL_ID:
                                    hybridstructpool_gate_result = (
                                        structshell_dual16_ablation_gate(
                                            state, hybridstructpool_runtime
                                        )
                                    )
                                elif hybrid_pool_id in {
                                    ROUTED_HYBRIDSTRUCTPOOL_ID,
                                    ROLLBACK_AWARE_ROUTED_HYBRIDSTRUCTPOOL_ID,
                                    OVERALL_ROLLBACK_ROUTED_HYBRIDSTRUCTPOOL_ID,
                                }:
                                    hybridstructpool_gate_result = (
                                        routed_hybridstructpool_high_stress_gate(
                                            state, hybridstructpool_runtime
                                        )
                                    )
                                else:
                                    hybridstructpool_gate_result = (
                                        hybridstructpool_high_stress_gate(
                                            state, hybridstructpool_runtime
                                        )
                                    )
                        hybridstructpool_gate_passed = bool(
                            hybridstructpool_gate_result
                            and hybridstructpool_gate_result["passed"]
                        )
                        if (
                            topology_runtime
                            and not topology_runtime.get("activation_gate")
                            and not topology_runtime.get("phase_guard")
                        ) or structpool_gate_passed or hybridstructpool_gate_passed:
                            if topology_analysis_cache is None:
                                topology_analysis_cache = TopologyAnalysisCache(
                                    state,
                                    static_grid=(
                                        feature_engine.static_grid
                                        if feature_engine is not None
                                        else None
                                    ),
                                )
                            else:
                                topology_analysis_cache.prepare(
                                    state,
                                    changed_agents=sorted(
                                        topology_pending_changed_agents
                                    ),
                                )
                            topology_pending_changed_agents.clear()
                            topology_state_analysis = topology_analysis_cache.analysis
                            topology_prepared_native_analysis = (
                                topology_analysis_cache.last_native_prepared
                            )
                            topology_state_analysis_seconds = (
                                topology_analysis_cache.last_prepare_seconds
                            )
                        candidates, proposal_metrics = generate_online_candidates(
                            environment,
                            state,
                            task_id=str(row["task_id"]),
                            solver_seed=solver_seed,
                            decision_index=decision_index,
                            proposal_config=effective_proposal,
                            state_hash=before_hash,
                            verify_full_state=verify_full_state,
                            proposal_backend=controller_runtime,
                            shadow_validation=bool(
                                job.get("proposal_shadow_validation", False)
                                and optimized_runtime_available
                            ),
                            topology_static_grid=(
                                feature_engine.static_grid
                                if feature_engine is not None
                                and _proposal_uses_static_grid_cache(
                                    effective_proposal
                                )
                                else None
                            ),
                            topology_state_analysis=topology_state_analysis,
                            topology_state_analysis_seconds=(
                                topology_state_analysis_seconds
                            ),
                            topology_no_progress_streak=no_progress_streak,
                            topology_remaining_wall_seconds=(
                                max(
                                    0.0,
                                    wall_budget
                                    - (time.perf_counter() - ttf_started_wall),
                                )
                                if wall_budget is not None
                                else None
                            ),
                            structpool_gate_result=structpool_gate_result,
                        )
                        proposal_metrics.update(
                            {
                                "guardpool_enabled": guardpool_runtime_enabled,
                                "guardpool_no_progress_streak": no_progress_streak,
                                "guardpool_active": guard_active_for_decision,
                                "guardpool_triggered": guard_triggered_now,
                                "guardpool_released": guard_released_now,
                                "guardpool_conflict_signature": conflict_signature,
                            }
                        )
                        if hybrid_stall_guard_config:
                            proposal_metrics.update(
                                {
                                    "hybridstructpool_stall_guard_enabled": True,
                                    "hybridstructpool_stall_guard_id": str(
                                        hybrid_stall_guard_config["guard_id"]
                                    ),
                                    "hybridstructpool_stall_guard_limit": int(
                                        hybrid_stall_guard_config[
                                            "no_progress_limit"
                                        ]
                                    ),
                                    "hybridstructpool_stall_guard_no_progress_streak": (
                                        no_progress_streak
                                    ),
                                    "hybridstructpool_stall_guard_active": (
                                        hybrid_stall_guard_active_for_decision
                                    ),
                                    "hybridstructpool_stall_guard_triggered": (
                                        hybrid_stall_guard_triggered_now
                                    ),
                                    "hybridstructpool_stall_guard_released": (
                                        hybrid_stall_guard_released_now
                                    ),
                                }
                            )
                        proposal_metrics["v3_s3_cache_hit"] = False
                        proposal_metrics["repair_state_cache_hit"] = False
                        if controller_mode == "official_adaptive":
                            feature_started = time.perf_counter()
                            candidate_rows = online_candidate_rows(
                                state, candidates, static_grid=static_grid
                            )
                            feature_seconds = time.perf_counter() - feature_started
                        else:
                            feature_engine_created = False
                            if feature_engine is None:
                                feature_engine = make_feature_engine(state)
                                feature_engine_created = True
                            if topology_prepared_native_analysis is not None:
                                state_feature_metrics = feature_engine.prepare(
                                    state,
                                    changed_agents=sorted(pending_changed_agents),
                                    prepared_native_analysis=(
                                        topology_prepared_native_analysis
                                    ),
                                )
                            elif feature_engine_created or decision_index == 0:
                                state_feature_metrics = dict(
                                    feature_engine.last_prepare_metrics
                                )
                            else:
                                state_feature_metrics = feature_engine.prepare(
                                    state,
                                    changed_agents=sorted(pending_changed_agents),
                                )
                            pending_changed_agents.clear()
                            if policy == "realized_dynamic":
                                proposal_rows = None
                            else:
                                proposal_rows, proposal_feature_metrics = (
                                    feature_engine.proposal_rows(
                                        candidates, state_hash=before_hash
                                    )
                                )
                            if policy == "realized_dynamic":
                                candidate_rows, realized_feature_metrics = (
                                    feature_engine.realized_rows(
                                        candidates, state_hash=before_hash
                                    )
                                )
                            else:
                                assert proposal_rows is not None
                                candidate_rows = list(proposal_rows)
                            state_analysis_seconds = (
                                float(
                                    state_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                                + float(
                                    proposal_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                                + float(
                                    realized_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                            )
                            feature_seconds = (
                                state_analysis_seconds
                                + float(
                                    proposal_feature_metrics.get(
                                        "proposal_feature_seconds", 0.0
                                    )
                                )
                                + float(
                                    realized_feature_metrics.get(
                                        "realized_feature_seconds", 0.0
                                    )
                                )
                            )
                        if hybridstructpool_runtime:
                            proposal_metrics.update(
                                {
                                    "hybridstructpool_enabled": True,
                                    "hybridstructpool_gate_evaluated": bool(
                                        hybridstructpool_gate_result.get(
                                            "evaluated", True
                                        )
                                    ),
                                    "hybridstructpool_gate_passed": (
                                        hybridstructpool_gate_passed
                                    ),
                                    "hybridstructpool_gate_reason": str(
                                        hybridstructpool_gate_result["reason"]
                                    ),
                                    "hybridstructpool_gate_seconds": float(
                                        hybridstructpool_gate_result["seconds"]
                                    ),
                                    "hybridstructpool_gate_id": str(
                                        hybridstructpool_gate_result.get(
                                            "gate_id",
                                            hybridstructpool_runtime[
                                                "activation_gate"
                                            ]["gate_id"],
                                        )
                                    ),
                                    "hybridstructpool_gate_conflict_pair_count": int(
                                        hybridstructpool_gate_result.get(
                                            "conflict_pair_count",
                                            state["num_of_colliding_pairs"],
                                        )
                                    ),
                                    "hybridstructpool_gate_active_conflict_agent_count": int(
                                        hybridstructpool_gate_result.get(
                                            "active_conflict_agent_count", 0
                                        )
                                    ),
                                    "hybridstructpool_gate_largest_conflict_component_size": int(
                                        hybridstructpool_gate_result.get(
                                            "largest_conflict_component_size", 0
                                        )
                                    ),
                                    "hybridstructpool_source_mode": str(
                                        hybridstructpool_runtime.get(
                                            "source_mode", "full_v8"
                                        )
                                    ),
                                }
                            )
                            if state_bounded_rollback_guard:
                                proposal_metrics[
                                    "hybridstructpool_state_bounded_v2_fallback"
                                ] = state_bounded_v2_fallback
                            if hybridstructpool_gate_passed:
                                if topology_state_analysis is None:
                                    raise ClosedLoopExecutionError(
                                        "hybridstructpool_analysis_missing",
                                        "HybridStructPool gate passed without state analysis",
                                    )
                                if feature_engine is None:
                                    raise ClosedLoopExecutionError(
                                        "hybridstructpool_feature_engine_missing",
                                        "HybridStructPool requires realized V2 features",
                                    )
                                hybrid_started = time.perf_counter()
                                base_candidates = list(candidates)
                                base_candidate_rows = list(candidate_rows)
                                (
                                    v2_anchor_index,
                                    _v2_anchor_scores,
                                    _v2_anchor_margin,
                                ) = score_online_candidates(
                                    base_candidate_rows, runtime_models[policy]
                                )
                                hybrid_pool_id = str(
                                    hybridstructpool_runtime.get("pool_id")
                                )
                                fixed_structshell_result = (
                                    _generate_fixed_structshell_runtime_candidates(
                                        state,
                                        topology_state_analysis,
                                        v2_candidates=base_candidates,
                                        v2_anchors=[
                                            base_candidates[v2_anchor_index]
                                        ],
                                        config=hybridstructpool_runtime,
                                    )
                                )
                                if fixed_structshell_result is not None:
                                    hybrid_result = fixed_structshell_result
                                elif hybrid_pool_id in {
                                    ROUTED_HYBRIDSTRUCTPOOL_ID,
                                    ROLLBACK_AWARE_ROUTED_HYBRIDSTRUCTPOOL_ID,
                                    OVERALL_ROLLBACK_ROUTED_HYBRIDSTRUCTPOOL_ID,
                                }:
                                    hybrid_result = (
                                        generate_routed_hybridstructpool_runtime_candidates(
                                            state,
                                            topology_state_analysis,
                                            v2_candidates=base_candidates,
                                            v2_anchors=[
                                                base_candidates[v2_anchor_index]
                                            ],
                                            config=hybridstructpool_runtime,
                                        )
                                    )
                                else:
                                    hybrid_result = generate_hybridstructpool_runtime_candidates(
                                        state,
                                        topology_state_analysis,
                                        v2_candidates=base_candidates,
                                        v2_anchors=[base_candidates[v2_anchor_index]],
                                        structural_family_sizes=hybridstructpool_runtime[
                                            "runtime_structural_family_sizes"
                                        ],
                                        maximum_causal_candidates=int(
                                            hybridstructpool_runtime[
                                                "maximum_causal_candidates"
                                            ]
                                        ),
                                        maximum_causal_neighborhood_size=int(
                                            hybridstructpool_runtime[
                                                "maximum_causal_neighborhood_size"
                                            ]
                                        ),
                                        causal_temporal_window=int(
                                            hybridstructpool_runtime[
                                                "causal_temporal_window"
                                            ]
                                        ),
                                        maximum_causal_jaccard=float(
                                            hybridstructpool_runtime[
                                                "maximum_causal_jaccard_similarity"
                                            ]
                                        ),
                                    )
                                candidates = list(hybrid_result.candidates)
                                if len(candidates) > int(
                                    hybridstructpool_runtime[
                                        "maximum_total_candidates"
                                    ]
                                ):
                                    raise ClosedLoopExecutionError(
                                        "hybridstructpool_candidate_cap_exceeded",
                                        "HybridStructPool runtime exceeded its registered cap",
                                        details={"candidate_count": len(candidates)},
                                    )
                                for candidate in candidates:
                                    candidate["hybridstructpool_provenance"] = list(
                                        hybrid_result.provenance_by_candidate_id[
                                            str(candidate["candidate_id"])
                                        ]
                                    )
                                hybrid_generation_seconds = (
                                    time.perf_counter() - hybrid_started
                                )
                                base_rows_by_id = {
                                    str(candidate["candidate_id"]): row
                                    for candidate, row in zip(
                                        base_candidates, base_candidate_rows
                                    )
                                }
                                challenger_candidates = [
                                    candidate
                                    for candidate in candidates
                                    if str(candidate["candidate_id"])
                                    not in base_rows_by_id
                                ]
                                if challenger_candidates:
                                    (
                                        challenger_rows,
                                        hybrid_feature_metrics,
                                    ) = feature_engine.realized_rows(
                                        challenger_candidates,
                                        state_hash=before_hash,
                                    )
                                else:
                                    challenger_rows = []
                                    hybrid_feature_metrics = {}
                                challenger_rows_by_id = {
                                    str(candidate["candidate_id"]): row
                                    for candidate, row in zip(
                                        challenger_candidates, challenger_rows
                                    )
                                }
                                candidate_rows = [
                                    base_rows_by_id.get(str(candidate["candidate_id"]))
                                    or challenger_rows_by_id[
                                        str(candidate["candidate_id"])
                                    ]
                                    for candidate in candidates
                                ]
                                hybrid_seconds = time.perf_counter() - hybrid_started
                                feature_seconds += sum(
                                    float(value)
                                    for key, value in hybrid_feature_metrics.items()
                                    if key.endswith("_seconds")
                                )
                                proposal_metrics.update(
                                    {
                                        "hybridstructpool_full_union_required": bool(
                                            hybridstructpool_runtime[
                                                "full_union_required"
                                            ]
                                        ),
                                        "hybridstructpool_full_union_audit_preserved": bool(
                                            hybridstructpool_runtime[
                                                "full_union_audit_preserved"
                                            ]
                                        ),
                                        "hybridstructpool_runtime_filter_id": str(
                                            hybridstructpool_runtime[
                                                "runtime_filter_id"
                                            ]
                                        ),
                                        "hybridstructpool_reused_base_feature_count": len(
                                            base_rows_by_id
                                        ),
                                        "hybridstructpool_computed_challenger_feature_count": len(
                                            challenger_candidates
                                        ),
                                        "hybridstructpool_v2_anchor_candidate_id": str(
                                            base_candidates[v2_anchor_index][
                                                "candidate_id"
                                            ]
                                        ),
                                        "hybridstructpool_base_candidate_count": (
                                            hybrid_result.base_candidate_count
                                        ),
                                        "hybridstructpool_structural_candidate_count": (
                                            hybrid_result.structural_candidate_count
                                        ),
                                        "hybridstructpool_causal_candidate_count": (
                                            hybrid_result.causal_candidate_count
                                        ),
                                        "hybridstructpool_challenger_count": len(
                                            hybrid_result.challengers
                                        ),
                                        "hybridstructpool_exact_duplicate_count": (
                                            hybrid_result.exact_duplicate_count
                                        ),
                                        "hybridstructpool_causal_attempt_count": len(
                                            hybrid_result.causal_attempts
                                        ),
                                        "hybridstructpool_candidate_count": len(
                                            candidates
                                        ),
                                        "hybridstructpool_seconds": hybrid_seconds,
                                        "hybridstructpool_generation_seconds": (
                                            hybrid_generation_seconds
                                        ),
                                        "hybridstructpool_structural_generation_seconds": float(
                                            hybrid_result.structural_generation_seconds
                                        ),
                                        "hybridstructpool_causal_generation_seconds": float(
                                            hybrid_result.causal_generation_seconds
                                        ),
                                        "candidate_count": len(candidates),
                                        "candidate_generation_seconds": float(
                                            proposal_metrics.get(
                                                "candidate_generation_seconds", 0.0
                                            )
                                        )
                                        + hybrid_generation_seconds,
                                    }
                                )
                        else:
                            proposal_metrics.update(
                                {
                                    "hybridstructpool_enabled": False,
                                    "hybridstructpool_gate_evaluated": False,
                                    "hybridstructpool_gate_passed": False,
                                    "hybridstructpool_gate_reason": "not_enabled",
                                    "hybridstructpool_gate_seconds": 0.0,
                                }
                            )
                        if (
                            slotpool_model_payload is not None
                            and bool(
                                proposal_metrics.get(
                                    "slotpool_reduction_pending", False
                                )
                            )
                        ):
                            slotpool_raw_candidates = list(candidates)
                            slotpool_raw_candidate_rows = list(candidate_rows)
                            base_raw_indices = [
                                index
                                for index, candidate in enumerate(candidates)
                                if not bool(
                                    candidate.get("structpool_family_groups")
                                )
                            ]
                            if not base_raw_indices:
                                raise ClosedLoopExecutionError(
                                    "slotpool_missing_v2_base_pool",
                                    "SlotPool runtime produced no V2 base candidates",
                                )
                            slotpool_started = time.perf_counter()
                            (
                                v2_anchor_base_index,
                                v2_base_scores,
                                _,
                            ) = score_online_candidates(
                                [candidate_rows[index] for index in base_raw_indices],
                                runtime_models[policy],
                            )
                            v2_anchor_raw_index = base_raw_indices[
                                v2_anchor_base_index
                            ]
                            v2_anchor_seconds = time.perf_counter() - slotpool_started
                            slotpool_rank_started = time.perf_counter()
                            slotpool_result = reduce_slotpool_candidates(
                                candidates=candidates,
                                candidate_rows=candidate_rows,
                                model_payload=slotpool_model_payload,
                                v2_anchor_index=v2_anchor_raw_index,
                                maximum_challengers=int(
                                    structpool_runtime_config[
                                        "maximum_slotpool_candidates"
                                    ]
                                ),
                            )
                            slotpool_rank_seconds = (
                                time.perf_counter() - slotpool_rank_started
                            )
                            slotpool_retained_raw_indices = list(
                                map(int, slotpool_result["retained_indices"])
                            )
                            retained_raw_set = set(slotpool_retained_raw_indices)
                            dropped_candidate_ids = [
                                str(candidate["candidate_id"])
                                for index, candidate in enumerate(candidates)
                                if index not in retained_raw_set
                            ]
                            candidates = [
                                candidates[index]
                                for index in slotpool_retained_raw_indices
                            ]
                            candidate_rows = [
                                candidate_rows[index]
                                for index in slotpool_retained_raw_indices
                            ]
                            slotpool_total_seconds = (
                                v2_anchor_seconds + slotpool_rank_seconds
                            )
                            proposal_metrics.update(
                                {
                                    "raw_candidate_count": len(
                                        slotpool_raw_candidates
                                    ),
                                    "candidate_count": len(candidates),
                                    "slotpool_reduction_applied": True,
                                    "slotpool_raw_structural_candidate_count": int(
                                        slotpool_result[
                                            "raw_structural_candidate_count"
                                        ]
                                    ),
                                    "slotpool_selected_structural_candidate_count": len(
                                        slotpool_result[
                                            "selected_structural_indices"
                                        ]
                                    ),
                                    "slotpool_selected_candidate_ids": list(
                                        slotpool_result["selected_candidate_ids"]
                                    ),
                                    "slotpool_selected_scores": list(
                                        slotpool_result["selected_scores"]
                                    ),
                                    "slotpool_dropped_candidate_ids": (
                                        dropped_candidate_ids
                                    ),
                                    "slotpool_v2_anchor_candidate_id": str(
                                        slotpool_result[
                                            "v2_anchor_candidate_id"
                                        ]
                                    ),
                                    "slotpool_v2_anchor_score": float(
                                        v2_base_scores[v2_anchor_base_index]
                                    ),
                                    "slotpool_v2_anchor_inference_seconds": (
                                        v2_anchor_seconds
                                    ),
                                    "slotpool_ranking_inference_seconds": (
                                        slotpool_rank_seconds
                                    ),
                                    "slotpool_total_inference_seconds": (
                                        slotpool_total_seconds
                                    ),
                                }
                            )
                        else:
                            proposal_metrics.update(
                                {
                                    "raw_candidate_count": len(candidates),
                                    "slotpool_reduction_applied": False,
                                    "slotpool_raw_structural_candidate_count": 0,
                                    "slotpool_selected_structural_candidate_count": 0,
                                    "slotpool_selected_candidate_ids": [],
                                    "slotpool_selected_scores": [],
                                    "slotpool_dropped_candidate_ids": [],
                                    "slotpool_v2_anchor_candidate_id": None,
                                    "slotpool_v2_anchor_score": None,
                                    "slotpool_v2_anchor_inference_seconds": 0.0,
                                    "slotpool_ranking_inference_seconds": 0.0,
                                    "slotpool_total_inference_seconds": 0.0,
                                }
                            )
                        if v3_s3_state is not None:
                            selected_local_index = 0
                            scores = [0.0] * len(candidate_rows)
                            margin = 0.0
                            inference_seconds = 0.0
                        elif learned_selector is not None:
                            inference_started = time.perf_counter()
                            selection = learned_selector.select(
                                SelectionRequest(
                                    candidates=candidates,
                                    candidate_rows=candidate_rows,
                                    before_fingerprint=before_repair_hash,
                                    agent_count=int(row["agent_count"]),
                                    profile=policy,
                                )
                            )
                            if selection.candidate_index is None:
                                raise ClosedLoopExecutionError(
                                    "controller_no_candidate",
                                    "learned controller did not select a candidate",
                                )
                            selected_local_index = int(selection.candidate_index)
                            scores = list(map(float, selection.diagnostics["scores"]))
                            margin = float(selection.diagnostics.get("margin", 0.0))
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )
                        else:
                            inference_started = time.perf_counter()
                            selected_local_index, scores, margin = (
                                score_online_candidates(
                                    candidate_rows, runtime_models[policy]
                                )
                            )
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )
                        inference_seconds += float(
                            proposal_metrics.get(
                                "slotpool_total_inference_seconds", 0.0
                            )
                        )
                        if stateful_controller is not None:
                            stateful_cache = {
                                "key": cache_key,
                                "candidates": candidates,
                                "candidate_rows": candidate_rows,
                                "scores": scores,
                                "margin": margin,
                                "base_selected_local_index": selected_local_index,
                                "proposal_metrics": dict(proposal_metrics),
                                "generation_decision_index": decision_index,
                            }
                        if (
                            exact_rollback_guard is not None
                            and not state_bounded_v2_fallback
                        ):
                            rollback_selection_cache = {
                                "key": cache_key,
                                "candidates": candidates,
                                "candidate_rows": candidate_rows,
                                "scores": scores,
                                "margin": margin,
                                "base_selected_local_index": selected_local_index,
                                "proposal_metrics": dict(proposal_metrics),
                                "generation_decision_index": decision_index,
                            }
                    controller["proposal"] = proposal_metrics
                    pruning_metrics = no_pruning_metrics(len(candidates))
                    base_selected_local_index = selected_local_index
                    diagnostic_shadow_seconds = 0.0
                    diagnostic_shadow_state_check_seconds = 0.0
                    diagnostic_shadow_total_seconds = 0.0
                    if diagnostic_shadow_selectors:
                        diagnostic_shadow_block_started = time.perf_counter()
                        shadow_check_started = time.perf_counter()
                        shadow_input_fingerprint = _fingerprint(
                            {
                                "state_fingerprint": before_repair_hash,
                                "candidate_rows": candidate_rows,
                            }
                        )
                        diagnostic_shadow_state_check_seconds += (
                            time.perf_counter() - shadow_check_started
                        )
                        diagnostic_records = {}
                        diagnostic_indices = {}
                        for shadow_id, shadow_selector in sorted(
                            diagnostic_shadow_selectors.items()
                        ):
                            shadow_started = time.perf_counter()
                            shadow_selection = shadow_selector.select(
                                SelectionRequest(
                                    candidates=candidates,
                                    candidate_rows=candidate_rows,
                                    before_fingerprint=before_repair_hash,
                                    agent_count=int(row["agent_count"]),
                                    profile=policy,
                                )
                            )
                            shadow_seconds = time.perf_counter() - shadow_started
                            diagnostic_shadow_seconds += shadow_seconds
                            shadow_index = shadow_selection.candidate_index
                            if (
                                shadow_index is None
                                or int(shadow_index) < 0
                                or int(shadow_index) >= len(candidates)
                            ):
                                raise ClosedLoopExecutionError(
                                    "diagnostic_shadow_invalid_action",
                                    f"{shadow_id} selected an invalid candidate",
                                )
                            shadow_index = int(shadow_index)
                            diagnostic_indices[shadow_id] = shadow_index
                            selected_shadow_row = candidate_rows[shadow_index]
                            range_diagnostic = feature_range_diagnostic(
                                selected_shadow_row,
                                policy,
                                diagnostic_shadow_ranges[shadow_id][policy],
                            )
                            fallback_threshold = (
                                diagnostic_shadow_fallback_thresholds[shadow_id]
                            )
                            diagnostic_records[shadow_id] = {
                                "selected_candidate_id": candidates[shadow_index][
                                    "candidate_id"
                                ],
                                "selected_candidate_matches_v2": (
                                    shadow_index == base_selected_local_index
                                ),
                                "action_overridden": False,
                                "candidate_count": len(candidates),
                                "inference_seconds": shadow_seconds,
                                "score_margin": float(
                                    shadow_selection.diagnostics.get("margin", 0.0)
                                ),
                                "feature_range": range_diagnostic,
                                "feature_range_fallback_threshold": (
                                    fallback_threshold
                                ),
                                "feature_range_fallback_suggested": (
                                    float(range_diagnostic["outside_fraction"])
                                    > fallback_threshold
                                ),
                            }
                            controller_totals[
                                f"diagnostic_shadow_decision_count:{shadow_id}"
                            ] += 1
                            controller_totals[
                                f"diagnostic_shadow_disagreement_count:{shadow_id}"
                            ] += int(shadow_index != base_selected_local_index)
                            controller_totals[
                                f"diagnostic_shadow_inference_seconds:{shadow_id}"
                            ] += shadow_seconds
                            controller_totals[
                                f"diagnostic_shadow_range_fallback_count:{shadow_id}"
                            ] += int(
                                float(range_diagnostic["outside_fraction"])
                                > fallback_threshold
                            )
                        shadow_check_started = time.perf_counter()
                        if _fingerprint(
                            {
                                "state_fingerprint": state_fingerprint(state),
                                "candidate_rows": candidate_rows,
                            }
                        ) != shadow_input_fingerprint:
                            raise ClosedLoopExecutionError(
                                "diagnostic_shadow_semantic_mismatch",
                                "diagnostic shadow changed state or candidate rows",
                            )
                        diagnostic_shadow_state_check_seconds += (
                            time.perf_counter() - shadow_check_started
                        )
                        controller_totals[
                            "diagnostic_shadow_state_check_seconds"
                        ] += diagnostic_shadow_state_check_seconds
                        controller_totals[
                            "diagnostic_shadow_semantic_mismatch_count"
                        ] += 0
                        controller_totals[
                            "diagnostic_shadow_action_override_count"
                        ] += 0
                        if len(diagnostic_indices) == 2:
                            controller_totals[
                                "diagnostic_shadow_pair_decision_count"
                            ] += 1
                            controller_totals[
                                "diagnostic_shadow_pair_disagreement_count"
                            ] += int(
                                len(set(diagnostic_indices.values())) > 1
                            )
                        controller["diagnostic_shadows"] = {
                            "passed": True,
                            "state_fingerprint_matches": True,
                            "candidate_rows_match": True,
                            "executed_controller": "v2-full",
                            "records": diagnostic_records,
                        }
                        diagnostic_shadow_total_seconds = (
                            time.perf_counter() - diagnostic_shadow_block_started
                        )
                        controller_totals[
                            "diagnostic_shadow_total_seconds"
                        ] += diagnostic_shadow_total_seconds
                    if bool(job.get("feature_shadow_validation", False)):
                        assert feature_engine is not None
                        shadow_rows = feature_engine.last_shadow_rows.get(policy)
                        if (
                            shadow_rows is not None
                            and slotpool_retained_raw_indices is not None
                        ):
                            if (
                                slotpool_raw_candidate_rows is None
                                or len(shadow_rows)
                                != len(slotpool_raw_candidate_rows)
                            ):
                                raise ClosedLoopExecutionError(
                                    "slotpool_shadow_mismatch",
                                    "SlotPool raw candidate shadow rows are incomplete",
                                )
                            shadow_rows = [
                                shadow_rows[index]
                                for index in slotpool_retained_raw_indices
                            ]
                        if shadow_rows is None or len(shadow_rows) != len(candidate_rows):
                            raise ClosedLoopExecutionError(
                                "controller_shadow_mismatch",
                                "feature shadow candidate rows are incomplete",
                                details={
                                    "candidate_count": len(candidate_rows),
                                    "shadow_candidate_count": (
                                        None if shadow_rows is None else len(shadow_rows)
                                    ),
                                },
                            )
                        feature_index, feature_scores, feature_margin = (
                            score_online_candidates(shadow_rows, runtime_models[policy])
                        )
                        feature_diagnostic = _score_equivalence_diagnostic(
                            left_rows=candidate_rows,
                            left_index=selected_local_index,
                            left_scores=scores,
                            left_margin=margin,
                            right_rows=shadow_rows,
                            right_index=feature_index,
                            right_scores=feature_scores,
                            right_margin=feature_margin,
                        )
                        if (
                            not feature_diagnostic["selected_candidate_matches"]
                            or not feature_diagnostic["ranking_matches"]
                            or feature_diagnostic["maximum_score_delta"] > 1e-12
                        ):
                            raise ClosedLoopExecutionError(
                                "controller_feature_shadow_mismatch",
                                "runtime scoring differs between native and reference feature rows",
                                details=feature_diagnostic,
                            )
                        feature_diagnostic["passed"] = True
                        controller["feature_shadow"] = feature_diagnostic
                        controller_totals["feature_shadow_validation_count"] += 1
                        controller_totals["feature_shadow_score_max_delta"] = max(
                            float(
                                controller_totals["feature_shadow_score_max_delta"]
                            ),
                            float(feature_diagnostic["maximum_score_delta"]),
                        )

                        if policy not in source_models:
                            raise ClosedLoopExecutionError(
                                "controller_source_model_missing",
                                "audit profile lacks a matching controller source model",
                                details={"profile": policy},
                            )
                        source_index, source_scores, source_margin = (
                            score_online_candidates(shadow_rows, source_models[policy])
                        )
                        source_diagnostic = _score_equivalence_diagnostic(
                            left_rows=shadow_rows,
                            left_index=feature_index,
                            left_scores=feature_scores,
                            left_margin=feature_margin,
                            right_rows=shadow_rows,
                            right_index=source_index,
                            right_scores=source_scores,
                            right_margin=source_margin,
                        )
                        source_diagnostic["source_model"] = dict(
                            source_model_provenance[policy]
                        )
                        source_diagnostic["score_tolerance"] = 1e-10
                        source_diagnostic["score_equivalent"] = (
                            source_diagnostic["maximum_score_delta"] <= 1e-10
                        )
                        source_diagnostic["ranking_equivalent"] = bool(
                            source_diagnostic["ranking_matches"]
                        )
                        source_diagnostic["action_equivalent"] = bool(
                            source_diagnostic["selected_candidate_matches"]
                        )
                        if not source_diagnostic["selected_candidate_matches"]:
                            raise ClosedLoopExecutionError(
                                "controller_source_model_mismatch",
                                "portable controller changes the source-model selected action",
                                details=source_diagnostic,
                            )
                        source_diagnostic["passed"] = True
                        controller["source_model_shadow"] = source_diagnostic
                        if controller_mode in {"v2-full", "mixed-full-v2"}:
                            controller["v1_v2_shadow"] = dict(source_diagnostic)
                        controller_totals["shadow_validation_count"] += 1
                        controller_totals["source_model_score_mismatch_count"] += int(
                            not source_diagnostic["score_equivalent"]
                        )
                        controller_totals[
                            "source_model_ranking_mismatch_count"
                        ] += int(not source_diagnostic["ranking_equivalent"])
                        controller_totals["shadow_score_max_delta"] = max(
                            float(controller_totals["shadow_score_max_delta"]),
                            float(source_diagnostic["maximum_score_delta"]),
                        )
                    v3_s3_seconds = 0.0
                    if v3_s3_state is not None:
                        assert v3_s3_selector is not None
                        v3_s3_select_started = time.perf_counter()
                        v3_s3_selection = v3_s3_selector.select(
                            SelectionRequest(
                                candidates=candidates,
                                candidate_rows=candidate_rows,
                                profile=policy,
                                before_fingerprint=before_repair_hash,
                                temporal_context=s3_temporal_context(
                                    v3_s3_history,
                                    int(row["agent_count"]),
                                    include_wall_time=bool(
                                        v3_s3_bundle.wall_time_history_required
                                    ),
                                ),
                                agent_count=int(row["agent_count"]),
                            )
                        )
                        v3_s3_selected_index = v3_s3_selection.candidate_index
                        v3_s3_diagnostic = dict(v3_s3_selection.diagnostics)
                        v3_s3_seconds = (
                            time.perf_counter() - v3_s3_select_started
                        )
                        selected_local_index = v3_s3_selected_index
                        controller["v3_s3"] = v3_s3_diagnostic
                        controller_totals["v3_s3_seconds"] += v3_s3_seconds
                    if exact_rollback_guard is not None:
                        rollback_guard_select_started = time.perf_counter()
                        rollback_guard_bannable_candidate_ids = {
                            str(candidate["candidate_id"])
                            for candidate in candidates
                            if "structshell_equal_four_size"
                            in set(
                                map(
                                    str,
                                    candidate.get("hybridstructpool_provenance") or (),
                                )
                            )
                            and "v2_base"
                            not in set(
                                map(
                                    str,
                                    candidate.get("hybridstructpool_provenance") or (),
                                )
                            )
                        }
                        v2_anchor_candidate_id = str(
                            proposal_metrics.get(
                                "hybridstructpool_v2_anchor_candidate_id",
                                candidates[base_selected_local_index]["candidate_id"],
                            )
                        )
                        if state_bounded_v2_fallback:
                            if rollback_guard_bannable_candidate_ids:
                                raise ClosedLoopExecutionError(
                                    "state_bounded_fallback_contains_structshell",
                                    "fresh V2 fallback unexpectedly contains a pure StructShell candidate",
                                )
                            selected_local_index = base_selected_local_index
                            rollback_guard_selection = exact_rollback_guard.snapshot(
                                before_repair_hash
                            )
                            rollback_guard_selection.update(
                                {
                                    "base_selected_candidate_id": str(
                                        candidates[base_selected_local_index][
                                            "candidate_id"
                                        ]
                                    ),
                                    "selected_candidate_id": str(
                                        candidates[selected_local_index]["candidate_id"]
                                    ),
                                    "selected_candidate_is_pure_structshell": False,
                                    "pure_structshell_candidate_count": 0,
                                    "v2_anchor_candidate_id": v2_anchor_candidate_id,
                                    "selection_phase": "fresh_v2_only",
                                    "selection_overridden": False,
                                    "newly_suppressed": False,
                                }
                            )
                        else:
                            (
                                selected_local_index,
                                rollback_guard_selection,
                            ) = exact_rollback_guard.select(
                                repair_fingerprint=before_repair_hash,
                                candidates=candidates,
                                candidate_rows=candidate_rows,
                                scores=scores,
                                v2_anchor_candidate_id=v2_anchor_candidate_id,
                                bannable_candidate_ids=(
                                    rollback_guard_bannable_candidate_ids
                                ),
                            )
                            if selected_local_index is None:
                                raise ClosedLoopExecutionError(
                                    "state_bounded_fallback_not_fresh",
                                    "state-bounded guard requested V2 fallback after Hybrid generation",
                                )
                        rollback_guard_selected_candidate_id = str(
                            candidates[selected_local_index]["candidate_id"]
                        )
                        controller[rollback_guard_trace_key] = {
                            "selection": rollback_guard_selection,
                            "observation": None,
                            "selection_seconds": (
                                time.perf_counter() - rollback_guard_select_started
                            ),
                            "observation_seconds": None,
                        }
                        controller_totals["exact_rollback_guard_seconds"] += float(
                            controller[rollback_guard_trace_key][
                                "selection_seconds"
                            ]
                        )
                        controller_totals[
                            "exact_rollback_guard_evaluated_count"
                        ] += 1
                        controller_totals[
                            "exact_rollback_guard_override_count"
                        ] += int(
                            bool(rollback_guard_selection["selection_overridden"])
                        )
                        controller_totals["repair_state_cache_hit_count"] += int(
                            bool(proposal_metrics.get("repair_state_cache_hit", False))
                        )
                    base_diagnostic = (
                        None
                        if v3_s3_bundle is not None
                        else feature_range_diagnostic(
                            candidate_rows[base_selected_local_index],
                            policy,
                            runtime_ranges[policy],
                        )
                    )
                    if selected_local_index is None:
                        if v3_s3_state is not None:
                            controller_stalled = True
                            break
                        raise ClosedLoopExecutionError(
                            "controller_no_candidate",
                            "controller did not select a candidate",
                        )
                    else:
                        selected = candidates[selected_local_index]
                        selected_row = candidate_rows[selected_local_index]
                        seed_draw_index = None
                        if episode_repair_seed_stream is not None:
                            seed_draw_index = episode_repair_seed_stream.draw_count
                            random_seed = episode_repair_seed_stream.next_seed(
                                selected["proposal_seeds"]
                            )
                        else:
                            random_seed = repair_random_seed(
                                str(row["task_id"]),
                                solver_seed,
                                before_hash,
                                decision_index,
                                str(selected["candidate_id"]),
                                selected["proposal_seeds"],
                            )
                        action = {
                            "mode": "explicit_neighborhood",
                            "agents": selected["agents"],
                            "random_seed": random_seed,
                        }
                        controller["repair_seed_policy"] = repair_seed_policy
                        controller["repair_seed_draw_index"] = seed_draw_index
                        diagnostic = feature_range_diagnostic(
                            selected_row, policy, runtime_ranges[policy]
                        )
                        proposal_metrics["guardpool_selected_structural"] = bool(
                            guardpool_runtime_enabled
                            and selected is not None
                            and selected.get("structpool_family_groups")
                        )
                        if hybridstructpool_runtime:
                            provenance = list(
                                selected.get("hybridstructpool_provenance") or ()
                            )
                            proposal_metrics.update(
                                {
                                    "hybridstructpool_selected_provenance": provenance,
                                    "hybridstructpool_selected_actual_size": len(
                                        selected["agents"]
                                    ),
                                    "hybridstructpool_selected_candidate_id": str(
                                        selected["candidate_id"]
                                    ),
                                }
                            )
                    retained_positions = {
                        str(candidate["candidate_id"]): local_index
                        for local_index, candidate in enumerate(candidates)
                    }
                    if len(retained_positions) != len(candidates):
                        raise ClosedLoopExecutionError(
                            "duplicate_candidate_id",
                            "runtime candidate IDs are not unique",
                        )
                    audit_candidates = (
                        slotpool_raw_candidates
                        if slotpool_raw_candidates is not None
                        else candidates
                    )
                    candidate_pool = []
                    for candidate in audit_candidates:
                        local_index = retained_positions.get(
                            str(candidate["candidate_id"])
                        )
                        candidate_pool.append(
                            {
                                **candidate,
                                "retained": local_index is not None,
                                "score": (
                                    scores[local_index]
                                    if (
                                        local_index is not None
                                        and v3_s3_bundle is None
                                    )
                                    else None
                                ),
                                "feature_out_of_range_fraction": (
                                    feature_range_diagnostic(
                                        candidate_rows[local_index],
                                        policy,
                                        runtime_ranges[policy],
                                    )["outside_fraction"]
                                    if local_index is not None
                                    else None
                                ),
                            }
                        )
                    # Keep controller test doubles and legacy proposal backends
                    # compatible with the timing-v1 schema.  Before native
                    # phased timing was added, proposal_seconds represented
                    # the complete candidate-generation stage.
                    candidate_generation_seconds = float(
                        proposal_metrics.get(
                            "candidate_generation_seconds",
                            proposal_metrics.get("proposal_seconds", 0.0),
                        )
                    )
                    state_check_seconds = float(
                        proposal_metrics.get("state_check_seconds", 0.0)
                    )
                    state_check_fingerprint_seconds = float(
                        proposal_metrics.get("state_check_fingerprint_seconds", 0.0)
                    )
                    controller_seconds_before_repair = max(
                        time.perf_counter() - proposal_started,
                        candidate_generation_seconds
                        + state_check_seconds
                        + feature_seconds
                        + float(pruning_metrics["pruner_seconds"])
                        + inference_seconds
                        + v3_s3_seconds
                        + diagnostic_shadow_total_seconds,
                    )
                    measured_selection_stages = (
                        candidate_generation_seconds
                        + state_check_seconds
                        + feature_seconds
                        + float(pruning_metrics["pruner_seconds"])
                        + inference_seconds
                        + v3_s3_seconds
                        + diagnostic_shadow_total_seconds
                    )
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "feature_backend": (
                                feature_engine.backend
                                if feature_engine is not None
                                else "reference-v1"
                            ),
                            "inference_backend": (
                                ",".join(v3_s3_bundle.inference_backends)
                                if v3_s3_bundle is not None
                                else getattr(
                                    runtime_models[policy],
                                    "inference_backend",
                                    (
                                        "native-portable-tree"
                                        if getattr(
                                            runtime_models[policy],
                                            "native_predictor",
                                            None,
                                        )
                                        is not None
                                        else "python-portable-tree"
                                    ),
                                )
                            ),
                            "v3_s3_inference_backends": (
                                list(v3_s3_bundle.inference_backends)
                                if v3_s3_bundle is not None
                                else None
                            ),
                            "diagnostic_shadow_seconds": (
                                diagnostic_shadow_seconds
                            ),
                            "diagnostic_shadow_state_check_seconds": (
                                diagnostic_shadow_state_check_seconds
                            ),
                            "diagnostic_shadow_total_seconds": (
                                diagnostic_shadow_total_seconds
                            ),
                            "candidate_pool": candidate_pool,
                            "pruning": pruning_metrics,
                            "feature_timings": {
                                **state_feature_metrics,
                                **proposal_feature_metrics,
                                **realized_feature_metrics,
                            },
                            "selected_candidate_id": (
                                selected["candidate_id"] if selected is not None else None
                            ),
                            "selected_score": (
                                scores[selected_local_index]
                                if (
                                    selected_local_index is not None
                                    and v3_s3_bundle is None
                                )
                                else None
                            ),
                            "base_selected_candidate_id": (
                                candidates[base_selected_local_index][
                                    "candidate_id"
                                ]
                                if v3_s3_bundle is None
                                else None
                            ),
                            "base_selected_score": (
                                scores[base_selected_local_index]
                                if v3_s3_bundle is None
                                else None
                            ),
                            "base_selected_feature_range": base_diagnostic,
                            "score_margin": (
                                margin if v3_s3_bundle is None else None
                            ),
                            "selected_feature_range": diagnostic,
                            "feature_seconds": feature_seconds,
                            "inference_seconds": inference_seconds,
                            "candidate_generation_seconds": candidate_generation_seconds,
                            "state_check_seconds": state_check_seconds,
                            "state_check_fingerprint_seconds": state_check_fingerprint_seconds,
                            "state_analysis_seconds": float(
                                state_analysis_seconds
                            ),
                            "proposal_feature_seconds": float(
                                proposal_feature_metrics.get(
                                    "proposal_feature_seconds", 0.0
                                )
                            ),
                            "realized_feature_seconds": float(
                                realized_feature_metrics.get(
                                    "realized_feature_seconds", 0.0
                                )
                            ),
                            "ranking_inference_seconds": inference_seconds,
                            "v3_s3_seconds": v3_s3_seconds,
                            "v3_s3_cache_hit": (
                                cache_hit and v3_s3_state is not None
                            ),
                            "selection_residual_seconds": max(
                                0.0,
                                controller_seconds_before_repair
                                - measured_selection_stages,
                            ),
                            "controller_seconds_before_repair": controller_seconds_before_repair,
                        }
                    )
                    if selected is not None:
                        selected_sizes[len(selected["agents"])] += 1
                        for family in selected["selection_families"]:
                            selected_families[str(family)] += 1
                    controller_totals["proposal_count"] += int(proposal_metrics["proposal_count"])
                    controller_totals["candidate_count"] += int(proposal_metrics["candidate_count"])
                    controller_totals["raw_candidate_count"] += int(
                        proposal_metrics.get(
                            "raw_candidate_count", proposal_metrics["candidate_count"]
                        )
                    )
                    controller_totals["base_candidate_count"] += int(
                        proposal_metrics.get(
                            "base_candidate_count", proposal_metrics["candidate_count"]
                        )
                    )
                    controller_totals["topology_boundary_generated_count"] += int(
                        proposal_metrics.get("topology_boundary_generated_count", 0)
                    )
                    controller_totals["topology_boundary_added_candidate_count"] += int(
                        proposal_metrics.get("topology_boundary_added_candidate_count", 0)
                    )
                    controller_totals["topology_boundary_analysis_seconds"] += float(
                        proposal_metrics.get("topology_boundary_analysis_seconds", 0.0)
                    )
                    for topology_metric in (
                        "topology_boundary_static_seconds",
                        "topology_boundary_dynamic_seconds",
                        "topology_boundary_candidate_seconds",
                        "topology_boundary_merge_seconds",
                        "topology_boundary_gate_seconds",
                    ):
                        controller_totals[topology_metric] += float(
                            proposal_metrics.get(topology_metric, 0.0)
                        )
                    controller_totals[
                        "topology_boundary_static_cache_hit_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "topology_boundary_static_cache_hit", False
                            )
                        )
                    )
                    controller_totals[
                        "topology_boundary_gate_evaluated_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "topology_boundary_gate_evaluated", False
                            )
                        )
                    )
                    controller_totals[
                        "topology_boundary_gate_passed_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "topology_boundary_gate_evaluated", False
                            )
                        )
                        and bool(
                            proposal_metrics.get(
                                "topology_boundary_gate_passed", False
                            )
                        )
                    )
                    gate_reason = str(
                        proposal_metrics.get(
                            "topology_boundary_gate_reason", "not_enabled"
                        )
                    )
                    controller_totals[
                        f"topology_boundary_gate_reason={gate_reason}"
                    ] += 1
                    controller_totals["structpool_generated_count"] += int(
                        proposal_metrics.get("structpool_generated_count", 0)
                    )
                    controller_totals["structpool_retained_candidate_count"] += int(
                        proposal_metrics.get("structpool_retained_candidate_count", 0)
                    )
                    controller_totals["structpool_filtered_candidate_count"] += int(
                        proposal_metrics.get("structpool_filtered_candidate_count", 0)
                    )
                    controller_totals["structpool_lean_filter_enabled_count"] += int(
                        bool(
                            proposal_metrics.get(
                                "structpool_lean_filter_enabled", False
                            )
                        )
                    )
                    controller_totals["structpool_added_candidate_count"] += int(
                        proposal_metrics.get("structpool_added_candidate_count", 0)
                    )
                    for structpool_metric in (
                        "structpool_analysis_seconds",
                        "structpool_static_seconds",
                        "structpool_dynamic_seconds",
                        "structpool_candidate_seconds",
                        "structpool_merge_seconds",
                        "structpool_gate_seconds",
                    ):
                        controller_totals[structpool_metric] += float(
                            proposal_metrics.get(structpool_metric, 0.0)
                        )
                    controller_totals["structpool_static_cache_hit_count"] += int(
                        bool(proposal_metrics.get("structpool_static_cache_hit", False))
                    )
                    controller_totals["structpool_gate_evaluated_count"] += int(
                        bool(proposal_metrics.get("structpool_gate_evaluated", False))
                    )
                    controller_totals["structpool_gate_passed_count"] += int(
                        bool(proposal_metrics.get("structpool_gate_evaluated", False))
                        and bool(proposal_metrics.get("structpool_gate_passed", False))
                    )
                    structpool_reason = str(
                        proposal_metrics.get("structpool_gate_reason", "not_enabled")
                    )
                    controller_totals[
                        f"structpool_gate_reason={structpool_reason}"
                    ] += 1
                    controller_totals["slotpool_reduction_applied_count"] += int(
                        bool(
                            proposal_metrics.get(
                                "slotpool_reduction_applied", False
                            )
                        )
                    )
                    controller_totals[
                        "slotpool_raw_structural_candidate_count"
                    ] += int(
                        proposal_metrics.get(
                            "slotpool_raw_structural_candidate_count", 0
                        )
                    )
                    controller_totals[
                        "slotpool_selected_structural_candidate_count"
                    ] += int(
                        proposal_metrics.get(
                            "slotpool_selected_structural_candidate_count", 0
                        )
                    )
                    for slotpool_metric in (
                        "slotpool_v2_anchor_inference_seconds",
                        "slotpool_ranking_inference_seconds",
                        "slotpool_total_inference_seconds",
                    ):
                        controller_totals[slotpool_metric] += float(
                            proposal_metrics.get(slotpool_metric, 0.0)
                        )
                    for hybrid_metric in (
                        "hybridstructpool_gate_seconds",
                        "hybridstructpool_seconds",
                        "hybridstructpool_generation_seconds",
                        "hybridstructpool_structural_generation_seconds",
                        "hybridstructpool_causal_generation_seconds",
                    ):
                        controller_totals[hybrid_metric] += float(
                            proposal_metrics.get(hybrid_metric, 0.0)
                        )
                    controller_totals[
                        "hybridstructpool_gate_evaluated_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "hybridstructpool_gate_evaluated", False
                            )
                        )
                    )
                    controller_totals[
                        "hybridstructpool_gate_passed_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "hybridstructpool_gate_evaluated", False
                            )
                        )
                        and bool(
                            proposal_metrics.get(
                                "hybridstructpool_gate_passed", False
                            )
                        )
                    )
                    controller_totals[
                        "hybridstructpool_structural_candidate_count"
                    ] += int(
                        proposal_metrics.get(
                            "hybridstructpool_structural_candidate_count", 0
                        )
                    )
                    controller_totals[
                        "hybridstructpool_causal_candidate_count"
                    ] += int(
                        proposal_metrics.get(
                            "hybridstructpool_causal_candidate_count", 0
                        )
                    )
                    selected_provenance = set(
                        map(
                            str,
                            proposal_metrics.get(
                                "hybridstructpool_selected_provenance", ()
                            ),
                        )
                    )
                    controller_totals[
                        "hybridstructpool_selected_structural_count"
                    ] += int("structshell_equal_four_size" in selected_provenance)
                    controller_totals[
                        "hybridstructpool_selected_causal_count"
                    ] += int("causalclosure_v2" in selected_provenance)
                    controller_totals[
                        "hybridstructpool_stall_guard_active_decision_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "hybridstructpool_stall_guard_active", False
                            )
                        )
                    )
                    controller_totals[
                        "hybridstructpool_stall_guard_trigger_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "hybridstructpool_stall_guard_triggered", False
                            )
                        )
                    )
                    controller_totals[
                        "hybridstructpool_stall_guard_release_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "hybridstructpool_stall_guard_released", False
                            )
                        )
                    )
                    controller_totals["guardpool_active_decision_count"] += int(
                        bool(proposal_metrics.get("guardpool_active", False))
                    )
                    controller_totals["guardpool_trigger_count"] += int(
                        bool(proposal_metrics.get("guardpool_triggered", False))
                    )
                    controller_totals["guardpool_release_count"] += int(
                        bool(proposal_metrics.get("guardpool_released", False))
                    )
                    controller_totals[
                        "guardpool_selected_structural_count"
                    ] += int(
                        bool(
                            proposal_metrics.get(
                                "guardpool_selected_structural", False
                            )
                        )
                    )
                    controller_totals["candidate_count_before_pruning"] += int(
                        pruning_metrics["candidate_count_before"]
                    )
                    controller_totals["candidate_count_after_pruning"] += int(
                        pruning_metrics["candidate_count_after"]
                    )
                    controller_totals["candidate_reduction_fraction_sum"] += float(
                        pruning_metrics["reduction_fraction"]
                    )
                    controller_totals["pruner_fallback_count"] += int(
                        bool(pruning_metrics["fallback"])
                    )
                    controller_totals["pruner_ood_fallback_count"] += int(
                        pruning_metrics.get("fallback_reason")
                        in {
                            "feature_out_of_range",
                            "non_finite_feature",
                            "unsupported_actual_size",
                        }
                    )
                    controller_totals["pruner_seconds"] += float(
                        pruning_metrics["pruner_seconds"]
                    )
                    controller_totals["proposal_seconds"] += float(proposal_metrics["proposal_seconds"])
                    controller_totals["candidate_generation_seconds"] += (
                        candidate_generation_seconds
                    )
                    controller_totals["state_check_seconds"] += state_check_seconds
                    controller_totals[f"proposal_backend={proposal_metrics['backend']}"] += 1
                    controller_totals[
                        f"state_check_backend={proposal_metrics['state_check_backend']}"
                    ] += 1
                    controller_totals["full_state_verification_count"] += int(
                        bool(proposal_metrics["full_state_verified"])
                    )
                    controller_totals["feature_seconds"] += feature_seconds
                    controller_totals["inference_seconds"] += inference_seconds
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_seconds_before_repair
                    )
                    for metrics in (
                        state_feature_metrics,
                        proposal_feature_metrics,
                        realized_feature_metrics,
                    ):
                        for key, value in metrics.items():
                            if key.endswith("_seconds"):
                                controller_totals[key] += float(value)
                    if diagnostic is not None:
                        controller_totals[
                            "selected_feature_outside_fraction_sum"
                        ] += float(diagnostic["outside_fraction"])
                        controller_totals["selected_feature_diagnostic_count"] += 1
                    controller_totals["learned_decisions"] += 1
                # Candidate generation and controller inference can consume the
                # remaining wall budget.  Do not enter the native solver after
                # its live deadline has expired: the environment correctly
                # rejects that call, but an expected timeout must not turn the
                # whole episode into an execution error.  Keep this before
                # route counters so only executed repairs are counted.
                if (
                    wall_budget is not None
                    and time.perf_counter() - ttf_started_wall >= wall_budget
                ):
                    external_timeout = True
                    break
                if (
                    bool(job.get("deterministic_pp_replay", False))
                    and not force_this_action
                    and rescue_override is None
                ):
                    # Pair the low-level PP stream across controller routes.
                    # Official neighborhood generation still consumes its
                    # upstream RNG stream before PP is reseeded; explicit
                    # learned actions differ only in the selected agent set.
                    replay_state_hash = (
                        before_hash
                        if pp_replay_seed_salt is None
                        else _fingerprint(
                            {
                                "state_hash": before_hash,
                                "seed_salt": pp_replay_seed_salt,
                            }
                        )
                    )
                    action["pp_random_seed"] = pp_replay_random_seed(
                        str(row["task_id"]),
                        solver_seed,
                        replay_state_hash,
                        decision_index,
                        route,
                    )
                    controller["pp_replay_seed_salt"] = pp_replay_seed_salt
                if native_retry_tracker is not None:
                    # Both arms collect the same native evidence.  Only the
                    # treatment arm is allowed to execute a second PP call.
                    action["collect_pp_diagnostics"] = True
                if (
                    failure_rescue_tracker is not None
                    and failure_rescue_tracker.requires_diagnostics(decision_index)
                ):
                    # All arms observe the same first native PP call.  A
                    # treatment, when eligible, is deferred to decision 1.
                    action["collect_pp_diagnostics"] = True
                if (
                    signature_rescue_tracker is not None
                    and signature_rescue_tracker.requires_diagnostics(decision_index)
                ):
                    action["collect_pp_diagnostics"] = True
                if (
                    wall_budget is not None
                    and time.perf_counter() - ttf_started_wall >= wall_budget
                ):
                    external_timeout = True
                    break
                repair_started = time.perf_counter()
                try:
                    timed_step = getattr(environment, "step_with_time_limit", None)
                    if wall_budget is not None and callable(timed_step):
                        live_pp_budget = max(
                            0.0,
                            wall_budget - (repair_started - ttf_started_wall),
                        )
                        result = _plain(timed_step(action, live_pp_budget))
                    else:
                        result = _plain(environment.step(action))
                except RuntimeError as error:
                    elapsed_after_error = time.perf_counter() - ttf_started_wall
                    if (
                        "repair episode" in str(error)
                        and "finished" in str(error)
                        and wall_budget is not None
                        and elapsed_after_error >= wall_budget
                    ):
                        external_timeout = True
                        break
                    raise
                bounded_retry_record = None
                if native_retry_tracker is not None:
                    first_result = result
                    first_state = dict(first_result["observation"])
                    first_metrics = dict(first_result["metrics"])
                    bounded_retry_record = native_retry_tracker.observe_first_attempt(
                        before=before,
                        after=first_state,
                        metrics=first_metrics,
                        decision_index=decision_index,
                    )
                    if bounded_retry_record["triggered"]:
                        retry_action = {
                            "mode": "explicit_neighborhood",
                            "agents": list(map(int, first_metrics["neighborhood"])),
                            "random_seed": int(bounded_retry_record["retry_seed"]),
                            "pp_random_seed": int(bounded_retry_record["retry_seed"]),
                            "collect_pp_diagnostics": True,
                        }
                        retry_started = time.perf_counter()
                        if wall_budget is not None and callable(timed_step):
                            live_retry_budget = max(
                                0.0,
                                wall_budget - (retry_started - ttf_started_wall),
                            )
                            if live_retry_budget <= 0.0:
                                bounded_retry_record = (
                                    native_retry_tracker.cancel_retry(
                                        bounded_retry_record,
                                        reason="episode_wall_budget_exhausted",
                                    )
                                )
                                retry_result = None
                            else:
                                retry_result = _plain(
                                    timed_step(retry_action, live_retry_budget)
                                )
                        else:
                            retry_result = _plain(environment.step(retry_action))
                        if retry_result is None:
                            result = {
                                **first_result,
                                "metrics": {
                                    **first_metrics,
                                    "bounded_native_retry": bounded_retry_record,
                                },
                            }
                        else:
                            bounded_retry_record = native_retry_tracker.observe_retry(
                                bounded_retry_record,
                                before=before,
                                after=dict(retry_result["observation"]),
                                metrics=dict(retry_result["metrics"]),
                            )
                            result = {
                                **retry_result,
                                "metrics": merged_retry_metrics(
                                    first_metrics,
                                    dict(retry_result["metrics"]),
                                    bounded_retry_record,
                                ),
                            }
                    else:
                        result = {
                            **first_result,
                            "metrics": {
                                **first_metrics,
                                "bounded_native_retry": bounded_retry_record,
                            },
                        }
                    native_retry_tracker.finalize_decision(bounded_retry_record)
                failure_rescue_record = None
                if failure_rescue_tracker is not None:
                    failure_rescue_record = (
                        failure_rescue_tracker.observe_decision(
                            decision_index=decision_index,
                            before=before,
                            after=dict(result["observation"]),
                            metrics=dict(result["metrics"]),
                        )
                    )
                    if failure_rescue_record is not None:
                        result = {
                            **result,
                            "metrics": {
                                **dict(result["metrics"]),
                                "failure_informed_rescue": (
                                    failure_rescue_record
                                ),
                            },
                        }
                signature_rescue_record = None
                if signature_rescue_tracker is not None:
                    signature_rescue_record = (
                        signature_rescue_tracker.observe_decision(
                            decision_index=decision_index,
                            before=before,
                            after=dict(result["observation"]),
                            metrics=dict(result["metrics"]),
                        )
                    )
                    result = {
                        **result,
                        "metrics": {
                            **dict(result["metrics"]),
                            "signature_scoped_rescue": signature_rescue_record,
                        },
                    }
                step_completed_wall = time.perf_counter()
                repair_wall_seconds = step_completed_wall - repair_started
                transition_ttf_elapsed_seconds = (
                    step_completed_wall - ttf_started_wall
                )
                post_step_started = time.perf_counter()
                total_repair_wall_seconds += repair_wall_seconds
                state = result["observation"]
                metrics = result["metrics"]
                if policy == "realized_dynamic":
                    route_prefix = (
                        "official" if route == "official_adaptive" else "model"
                    )
                    controller_totals[f"{route_prefix}_decision_count"] += 1
                    controller_totals[
                        f"{route_prefix}_controller_seconds"
                    ] += float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    )
                    if (
                        route == "official_adaptive"
                        and "candidate_pool" not in controller
                    ):
                        controller_totals[
                            "controller_seconds_before_repair"
                        ] += float(
                            controller.get(
                                "controller_seconds_before_repair", 0.0
                            )
                        )
                    if previous_route is not None and previous_route != route:
                        controller_totals["route_switch_count"] += 1
                    previous_route = route
                if "pp_random_seed" in action:
                    requested_pp_seed = int(action["pp_random_seed"])
                    retry_record = dict(metrics.get("bounded_native_retry") or {})
                    observed_first_seed = int(
                        dict(retry_record.get("first_attempt") or {}).get(
                            "requested_pp_random_seed",
                            metrics.get("requested_pp_random_seed", -1),
                        )
                    )
                    if observed_first_seed != requested_pp_seed:
                        raise ClosedLoopExecutionError(
                            "pp_seed_mismatch",
                            "native transition did not retain the requested PP seed",
                        )
                    first_attempt = dict(retry_record.get("first_attempt") or {})
                    first_repair_order = first_attempt.get(
                        "repair_order", metrics.get("repair_order")
                    )
                    first_applied_seed = int(
                        first_attempt.get(
                            "applied_pp_random_seed",
                            metrics.get("applied_pp_random_seed", -1),
                        )
                    )
                    if first_repair_order and first_applied_seed != requested_pp_seed:
                        raise ClosedLoopExecutionError(
                            "pp_seed_not_applied",
                            "native PP did not apply the deterministic replay seed",
                        )
                    if retry_record.get("triggered"):
                        retry_seed = int(retry_record["retry_seed"])
                        retry_attempt = dict(retry_record["retry_attempt"])
                        if (
                            int(retry_attempt["requested_pp_random_seed"])
                            != retry_seed
                            or retry_attempt.get("repair_order")
                            and int(retry_attempt["applied_pp_random_seed"])
                            != retry_seed
                        ):
                            raise ClosedLoopExecutionError(
                                "bounded_retry_seed_mismatch",
                                "native retry did not retain the registered seed",
                            )
                try:
                    native_timing_schema = _native_repair_timing_schema(metrics)
                except (TypeError, ValueError) as error:
                    raise ClosedLoopExecutionError(
                        "invalid_native_timing", str(error)
                    ) from error
                low_level_delta = _low_level_delta(before, state)
                actual = sorted(map(int, metrics.get("neighborhood", [])))
                if policy in LEARNED_POLICIES and route == "model":
                    if not bool(metrics.get("action_valid")) or actual != sorted(action["agents"]):
                        invalid_actions += 1
                        raise ClosedLoopExecutionError(
                            "invalid_action",
                            "explicit closed-loop action was rejected or changed",
                        )
                elif not bool(metrics.get("action_valid", True)):
                    invalid_actions += 1
                    raise ClosedLoopExecutionError(
                        "invalid_action", "official closed-loop action was rejected"
                    )
                pending_changed_agents.update(actual)
                topology_pending_changed_agents.update(actual)
                if policy == "realized_dynamic":
                    route_prefix = "official" if route == "official_adaptive" else "model"
                    route_controller_seconds = float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    )
                    controller["repair_wall_seconds"] = repair_wall_seconds
                    controller["total_decision_seconds"] = (
                        route_controller_seconds + repair_wall_seconds
                    )
                    controller_totals[f"{route_prefix}_repair_seconds"] += repair_wall_seconds
                    controller_totals[f"{route_prefix}_total_decision_seconds"] += (
                        route_controller_seconds + repair_wall_seconds
                    )
                if bounded_retry_record is not None:
                    controller["bounded_native_retry"] = bounded_retry_record
                    controller_totals["bounded_retry_trigger_count"] += int(
                        bool(bounded_retry_record["triggered"])
                    )
                    controller_totals["persistent_platform_decision_count"] += int(
                        bool(bounded_retry_record["persistent_after_transaction"])
                    )
                    controller_totals["bounded_retry_resolved_count"] += int(
                        bool(bounded_retry_record["resolved_by_retry"])
                    )
                if failure_rescue_record is not None:
                    controller["failure_informed_rescue"] = (
                        failure_rescue_record
                    )
                    if decision_index == 0:
                        controller_totals["failure_rescue_trigger_count"] += int(
                            bool(failure_rescue_record["triggered"])
                        )
                    controller_totals["failure_rescue_resolved_count"] += int(
                        bool(failure_rescue_record["resolved_by_rescue"])
                    )
                if signature_rescue_record is not None:
                    controller["signature_scoped_rescue"] = signature_rescue_record
                    controller_totals["signature_rescue_scheduled_count"] += int(
                        bool(signature_rescue_record["scheduled"])
                    )
                    controller_totals["signature_rescue_executed_count"] += int(
                        bool(signature_rescue_record["intervention_executed"])
                    )
                    controller_totals["signature_rescue_resolved_count"] += int(
                        bool(signature_rescue_record["resolved_by_rescue"])
                    )
                conflicts.append(int(state["num_of_colliding_pairs"]))
                if conflicts[-1] < conflicts[-2]:
                    no_progress_streak = 0
                else:
                    no_progress_streak += 1
                elapsed_wall = transition_ttf_elapsed_seconds
                transition_elapsed_seconds.append(elapsed_wall)
                within_wall_budget = (
                    wall_budget is None or elapsed_wall <= wall_budget
                )
                if within_wall_budget:
                    repair_iterations_within_budget += 1
                    budget_final_conflicts = conflicts[-1]
                    budget_final_sum_of_costs = int(state["sum_of_costs"])
                    budget_final_low_level = dict(state["low_level"])

                controller_before_repair_seconds = float(
                    controller.get("controller_seconds_before_repair", 0.0)
                )
                native_step_seconds = float(
                    metrics.get(
                        "native_step_seconds",
                        metrics.get("step_runtime", 0.0),
                    )
                )
                episode_runtime_delta_seconds = float(
                    metrics.get(
                        "episode_runtime_delta_seconds",
                        metrics.get("step_runtime", native_step_seconds),
                    )
                )
                native_neighborhood_seconds = float(
                    metrics.get("native_neighborhood_generation_seconds", 0.0)
                )
                pp_replan_seconds = float(metrics.get("pp_replan_seconds", 0.0))
                native_bookkeeping_seconds = float(
                    metrics.get("native_repair_bookkeeping_seconds", 0.0)
                )
                native_residual_seconds = float(
                    metrics.get("native_residual_seconds", 0.0)
                )
                native_state_snapshot_seconds = float(
                    metrics.get("native_state_snapshot_seconds", 0.0)
                )
                binding_state_snapshot_seconds = float(
                    metrics.get("binding_state_snapshot_seconds", 0.0)
                )
                state_to_python_seconds = float(
                    metrics.get("state_to_python_seconds", 0.0)
                )
                state_export_seconds = (
                    native_state_snapshot_seconds
                    + binding_state_snapshot_seconds
                    + state_to_python_seconds
                )
                neighborhood_selection_seconds = (
                    controller_before_repair_seconds + native_neighborhood_seconds
                )
                environment_step_residual_seconds = max(
                    0.0,
                    repair_wall_seconds
                    - native_neighborhood_seconds
                    - pp_replan_seconds
                    - native_bookkeeping_seconds
                    - state_export_seconds,
                )
                controller["neighborhood_selection_seconds"] = (
                    neighborhood_selection_seconds
                )
                controller_totals["neighborhood_selection_seconds"] += (
                    neighborhood_selection_seconds
                )
                controller_totals["pp_replan_seconds"] += pp_replan_seconds
                controller_totals["repair_bookkeeping_seconds"] += (
                    native_bookkeeping_seconds
                )
                controller_totals["state_export_seconds"] += state_export_seconds
                controller_totals["environment_step_residual_seconds"] += (
                    environment_step_residual_seconds
                )
                if route == "official_adaptive":
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_before_repair_seconds
                    )

                after_fingerprint_started = time.perf_counter()
                after_hash = state_fingerprint(state)
                current_state_fingerprint = after_hash
                if v3_s3_state is not None or exact_rollback_guard is not None:
                    after_repair_hash = repair_structure_fingerprint(state)
                else:
                    after_repair_hash = after_hash
                state_fingerprint_seconds = before_fingerprint_seconds + (
                    time.perf_counter() - after_fingerprint_started
                ) + float(controller.get("state_check_fingerprint_seconds", 0.0))
                if (
                    exact_rollback_guard is not None
                    and rollback_guard_selected_candidate_id is not None
                ):
                    rollback_guard_observe_started = time.perf_counter()
                    rollback_guard_observation = exact_rollback_guard.observe(
                        before_repair_fingerprint=before_repair_hash,
                        after_repair_fingerprint=after_repair_hash,
                        selected_candidate_id=rollback_guard_selected_candidate_id,
                        metrics=metrics,
                        bannable_candidate_ids=(
                            rollback_guard_bannable_candidate_ids
                        ),
                    )
                    guard_record = dict(
                        controller.get(rollback_guard_trace_key) or {}
                    )
                    guard_record["observation"] = rollback_guard_observation
                    guard_record["observation_seconds"] = (
                        time.perf_counter() - rollback_guard_observe_started
                    )
                    controller[rollback_guard_trace_key] = guard_record
                    if (
                        state_bounded_rollback_guard
                        and bool(
                            rollback_guard_observation.get(
                                "structshell_suppressed", False
                            )
                        )
                    ):
                        rollback_selection_cache = None
                    controller_totals["exact_rollback_guard_seconds"] += float(
                        guard_record["observation_seconds"]
                    )
                    controller_totals[
                        "exact_rollback_guard_exact_rollback_count"
                    ] += int(
                        bool(
                            rollback_guard_observation[
                                "exact_conflict_bound_rollback"
                            ]
                        )
                    )
                    controller_totals[
                        "exact_rollback_guard_new_ban_count"
                    ] += int(bool(rollback_guard_observation["newly_banned"]))
                    if state_bounded_rollback_guard:
                        controller_totals[
                            "exact_rollback_guard_new_state_suppression_count"
                        ] += int(
                            bool(
                                rollback_guard_observation.get(
                                    "newly_suppressed", False
                                )
                            )
                        )
                if v3_s3_state is not None:
                    v3_s3_observe_started = time.perf_counter()
                    repair_outcome = classify_repair_outcome(
                        before_fingerprint=before_repair_hash,
                        after_fingerprint=after_repair_hash,
                        replan_success=bool(metrics.get("replan_success")),
                        conflicts_before=int(before["num_of_colliding_pairs"]),
                        conflicts_after=int(state["num_of_colliding_pairs"]),
                        feasible=bool(state.get("feasible")),
                    )
                    conflict_reduction = max(
                        0,
                        int(before["num_of_colliding_pairs"])
                        - int(state["num_of_colliding_pairs"]),
                    )
                    continuation_expected = v3_s3_state.observe(
                        before_fingerprint=before_repair_hash,
                        after_fingerprint=after_repair_hash,
                        repair_outcome=repair_outcome,
                        conflict_reduction=float(conflict_reduction),
                        total_seconds=float(
                            controller_before_repair_seconds
                            + repair_wall_seconds
                        ),
                        feasible=bool(state.get("feasible")),
                    )
                    v3_s3_history.append(
                        {
                            "conflict_reduction": float(conflict_reduction),
                            "repair_seconds": float(repair_wall_seconds),
                            "state_changed": (
                                before_repair_hash != after_repair_hash
                            ),
                            "no_progress": repair_outcome
                            in {"hard_failure", "accepted_noop"},
                            "neighborhood_size": len(actual),
                        }
                    )
                    observe_seconds = (
                        time.perf_counter() - v3_s3_observe_started
                    )
                    controller["v3_s3"] = {
                        **dict(controller.get("v3_s3") or {}),
                        "repair_outcome": repair_outcome,
                        "continuation_expected": bool(
                            continuation_expected
                        ),
                        "state_unchanged": (
                            before_repair_hash == after_repair_hash
                        ),
                    }
                    controller["v3_s3_seconds"] = float(
                        controller.get("v3_s3_seconds", 0.0)
                    ) + observe_seconds
                    controller_totals["v3_s3_seconds"] += observe_seconds
                transition_timings = {
                    "native_step_seconds": native_step_seconds,
                    "episode_runtime_delta_seconds": (
                        episode_runtime_delta_seconds
                    ),
                    "pre_step_orchestration_seconds": pre_step_orchestration_seconds,
                    "controller_before_repair_seconds": controller_before_repair_seconds,
                    "candidate_generation_seconds": float(
                        controller.get("candidate_generation_seconds", 0.0)
                    ),
                    "state_check_seconds": float(
                        controller.get("state_check_seconds", 0.0)
                    ),
                    "state_check_fingerprint_seconds": float(
                        controller.get("state_check_fingerprint_seconds", 0.0)
                    ),
                    "state_analysis_seconds": float(
                        controller.get("state_analysis_seconds", 0.0)
                    ),
                    "proposal_feature_seconds": float(
                        controller.get("proposal_feature_seconds", 0.0)
                    ),
                    "realized_feature_seconds": float(
                        controller.get("realized_feature_seconds", 0.0)
                    ),
                    "ranking_inference_seconds": float(
                        controller.get("ranking_inference_seconds", 0.0)
                    ),
                    "v3_s3_seconds": float(
                        controller.get("v3_s3_seconds", 0.0)
                    ),
                    "selection_residual_seconds": float(
                        controller.get("selection_residual_seconds", 0.0)
                    ),
                    "native_neighborhood_generation_seconds": native_neighborhood_seconds,
                    "neighborhood_selection_seconds": neighborhood_selection_seconds,
                    "pp_replan_seconds": pp_replan_seconds,
                    "repair_bookkeeping_seconds": native_bookkeeping_seconds,
                    "native_residual_seconds": native_residual_seconds,
                    "state_export_seconds": state_export_seconds,
                    "environment_step_residual_seconds": environment_step_residual_seconds,
                    "environment_step_wall_seconds": repair_wall_seconds,
                    "state_fingerprint_seconds": state_fingerprint_seconds,
                    "post_step_orchestration_seconds": 0.0,
                    "iteration_wall_seconds": 0.0,
                }
                transition = {
                    "schema": EPISODE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "transition",
                    "episode_id": episode_id,
                    "decision_index": decision_index,
                    "action": action,
                    "before_fingerprint": before_hash,
                    "after_fingerprint": after_hash,
                    "metrics": metrics,
                    "low_level_delta": low_level_delta,
                    "repair_wall_seconds": repair_wall_seconds,
                    "elapsed_wall_seconds": elapsed_wall,
                    "within_wall_budget": within_wall_budget,
                    "native_timing_schema": native_timing_schema,
                    "timings": transition_timings,
                    "controller": controller,
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "after": state,
                }
                transition_timings["post_step_orchestration_seconds"] = (
                    time.perf_counter() - post_step_started
                )
                transition_timings["iteration_wall_seconds"] = (
                    time.perf_counter() - iteration_started
                )
                controller_totals["pre_step_orchestration_seconds"] += float(
                    transition_timings["pre_step_orchestration_seconds"]
                )
                controller_totals["post_step_orchestration_seconds"] += float(
                    transition_timings["post_step_orchestration_seconds"]
                )
                controller_totals["iteration_wall_seconds"] += float(
                    transition_timings["iteration_wall_seconds"]
                )
                controller_totals["state_fingerprint_seconds"] += float(
                    transition_timings["state_fingerprint_seconds"]
                )
                if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                    transition = encode_transition_event(transition, before, state)
                trace_write_seconds = _emit(stream, transition)
                transition_trace_write_seconds.append(trace_write_seconds)
                controller_totals["trace_write_seconds"] += trace_write_seconds
                if (
                    wall_budget is not None
                    and elapsed_wall >= wall_budget
                    and not bool(state["done"])
                ):
                    external_timeout = True
                    break
            elapsed_wall = time.perf_counter() - ttf_started_wall
            episode_observed_wall = time.perf_counter() - started_wall
            episode_finalize_started = time.perf_counter()
            algorithm_elapsed = (
                transition_elapsed_seconds[-1]
                if transition_elapsed_seconds
                else initial_state_elapsed_seconds
            )
            feasible_elapsed = algorithm_elapsed if bool(state["feasible"]) else None
            success = feasible_elapsed is not None and (
                wall_budget is None or feasible_elapsed <= wall_budget
            )
            if (
                not success
                and wall_budget is not None
                and algorithm_elapsed >= wall_budget
            ):
                external_timeout = True
            truncated = not success
            repair_limit_reached = (
                max_decisions > 0
                and len(conflicts) - 1 >= max_decisions
                and not success
            )
            metric_iteration_budget = job.get("metric_iteration_budget")
            fixed_auc = (
                fixed_budget_conflict_auc(
                    conflicts,
                    int(metric_iteration_budget),
                    success=success,
                )
                if metric_iteration_budget is not None
                else None
            )
            normalized_fixed_auc = (
                fixed_auc / (float(conflicts[0]) * int(metric_iteration_budget))
                if fixed_auc is not None
                and conflicts[0] > 0
                and metric_iteration_budget is not None
                else None
            )
            wall_metric_horizon = (
                wall_budget
                if wall_budget is not None
                else max(algorithm_elapsed, 1e-12)
            )
            wall_auc = wall_clock_conflict_auc(
                conflicts, transition_elapsed_seconds, wall_metric_horizon
            )
            normalized_wall_auc = (
                wall_auc / (float(conflicts[0]) * wall_metric_horizon)
                if conflicts[0] > 0
                else None
            )
            if success:
                stop_reason = "success"
            elif controller_stalled:
                stop_reason = "controller_stalled"
            elif repair_limit_reached:
                stop_reason = "repair_limit"
            elif external_timeout:
                stop_reason = "wall_timeout"
            elif bool(state["done"]):
                stop_reason = "native_terminal"
            else:
                stop_reason = "truncated"
            model_decisions = int(controller_totals["model_decision_count"])
            official_decisions = int(controller_totals["official_decision_count"])
            if (
                policy == "realized_dynamic"
                and model_decisions + official_decisions != len(conflicts) - 1
            ):
                raise ClosedLoopExecutionError(
                    "route_accounting_mismatch",
                    "controller route counts do not equal repair iterations",
                )
            summary = {
                "initial_fingerprint": initial_fingerprint,
                "initial_conflicts": conflicts[0],
                "final_conflicts": conflicts[-1],
                "budget_final_conflicts": budget_final_conflicts,
                "repairable": conflicts[0] > 0,
                "success": success,
                "truncated": truncated,
                "external_timeout": external_timeout,
                "stop_reason": stop_reason,
                "stopping_rule": stopping_rule,
                "wall_time_budget_seconds": wall_budget,
                "repair_iterations": len(conflicts) - 1,
                "repair_iterations_within_budget": repair_iterations_within_budget,
                "conflict_trajectory": conflicts,
                "transition_elapsed_seconds": transition_elapsed_seconds,
                "conflict_auc": sum(
                    (conflicts[index] + conflicts[index + 1]) / 2.0
                    for index in range(len(conflicts) - 1)
                ),
                "fixed_budget_conflict_auc": fixed_auc,
                "normalized_fixed_budget_conflict_auc": normalized_fixed_auc,
                "metric_iteration_budget": metric_iteration_budget,
                "wall_clock_conflict_auc": wall_auc,
                "normalized_wall_clock_conflict_auc": normalized_wall_auc,
                "wall_time_to_feasible": feasible_elapsed if success else None,
                "capped_wall_time_to_feasible": (
                    min(feasible_elapsed, wall_budget)
                    if success and wall_budget is not None
                    else wall_budget
                    if wall_budget is not None
                    else None
                ),
                "budget_overshoot_seconds": (
                    max(0.0, algorithm_elapsed - wall_budget)
                    if wall_budget is not None
                    else 0.0
                ),
                "native_time_to_feasible": float(state["runtime"]) if success else None,
                "repair_wall_seconds": total_repair_wall_seconds,
                "environment_construct_seconds": environment_construct_seconds,
                "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
                "ttf_observed_wall_seconds": elapsed_wall,
                "reset_wall_seconds": reset_wall_seconds,
                "reset_timings": reset_timings,
                "initial_state_elapsed_seconds": initial_state_elapsed_seconds,
                "initial_fingerprint_seconds": initial_fingerprint_seconds,
                "initial_trace_write_seconds": initial_trace_write_seconds,
                "transition_trace_write_seconds": transition_trace_write_seconds,
                "trace_write_seconds": initial_trace_write_seconds
                + sum(transition_trace_write_seconds),
                "episode_observed_wall_seconds": episode_observed_wall,
                "timing_unaccounted_seconds": max(
                    0.0,
                    episode_observed_wall
                    - environment_construct_seconds
                    - reset_wall_seconds
                    - initial_trace_write_seconds
                    - sum(transition_trace_write_seconds)
                    - float(controller_totals["iteration_wall_seconds"]),
                ),
                "controller_mode": controller_mode,
                "controller_runtime": controller_runtime,
                "requested_controller_runtime": requested_controller_runtime,
                "verification_profile": verification_profile,
                "feature_backend": (
                    feature_engine.backend
                    if feature_engine is not None
                    else "reference-v1"
                    if policy in LEARNED_POLICIES
                    and controller_mode == "official_adaptive"
                    else "not_used"
                    if policy in LEARNED_POLICIES
                    else None
                ),
                "controller_totals": dict(controller_totals),
                "model_decision_count": model_decisions,
                "official_decision_count": official_decisions,
                "model_route_fraction": (
                    model_decisions / (model_decisions + official_decisions)
                    if model_decisions + official_decisions
                    else 0.0
                ),
                "route_switch_count": int(controller_totals["route_switch_count"]),
                "balanced_controller": None,
                "v3_s3": (
                    {
                        **v3_s3_state.summary(),
                        "combined_runtime_feature_count": len(
                            required_model_features
                        ),
                        "history_length": len(v3_s3_history),
                    }
                    if v3_s3_state is not None
                    and policy == "realized_dynamic"
                    else None
                ),
                "candidate_reduction_fraction": (
                    1.0
                    - float(controller_totals["candidate_count_after_pruning"])
                    / float(controller_totals["candidate_count_before_pruning"])
                    if controller_totals["candidate_count_before_pruning"]
                    else 0.0
                ),
                "pruner_fallback_fraction": (
                    float(controller_totals["pruner_fallback_count"])
                    / float(controller_totals["learned_decisions"])
                    if controller_totals["learned_decisions"]
                    else 0.0
                ),
                "pruner_ood_fallback_fraction": (
                    float(controller_totals["pruner_ood_fallback_count"])
                    / float(controller_totals["learned_decisions"])
                    if controller_totals["learned_decisions"]
                    else 0.0
                ),
                "mean_selected_feature_outside_fraction": (
                    float(controller_totals["selected_feature_outside_fraction_sum"])
                    / float(controller_totals["selected_feature_diagnostic_count"])
                    if controller_totals["selected_feature_diagnostic_count"]
                    else 0.0
                ),
                "selected_size_counts": {
                    str(key): value for key, value in sorted(selected_sizes.items())
                },
                "selected_family_counts": dict(sorted(selected_families.items())),
                "invalid_action_count": invalid_actions,
                "fingerprint_mismatch_count": fingerprint_mismatches,
                "bounded_native_retry": (
                    native_retry_tracker.summary()
                    if native_retry_tracker is not None
                    else None
                ),
                "failure_informed_rescue": (
                    failure_rescue_tracker.summary()
                    if failure_rescue_tracker is not None
                    else None
                ),
                "signature_scoped_rescue": (
                    signature_rescue_tracker.summary()
                    if signature_rescue_tracker is not None
                    else None
                ),
                "final_sum_of_costs": int(state["sum_of_costs"]),
                "budget_final_sum_of_costs": budget_final_sum_of_costs,
                "final_low_level": state["low_level"],
                "budget_final_low_level": budget_final_low_level,
            }
            final_fingerprint_started = time.perf_counter()
            final_fingerprint = current_state_fingerprint
            final_fingerprint_seconds = time.perf_counter() - final_fingerprint_started
            summary["final_fingerprint_seconds"] = final_fingerprint_seconds
            finish_event = {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "finish",
                "episode_id": episode_id,
                "policy": policy,
                "success": success,
                "final_fingerprint": final_fingerprint,
                "summary": summary,
            }
            if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                finish_event = encode_finish_event(finish_event)
            finish_event_orchestration_seconds = (
                time.perf_counter() - episode_finalize_started
            )
            finish_trace_write_seconds = _emit(stream, finish_event)
            trace_close_started = time.perf_counter()
        trace_close_seconds = time.perf_counter() - trace_close_started
        validation_started = time.perf_counter()
        validated = validate_closed_loop_trace(
            partial_path,
            job["run_fingerprint"],
            expected_episode_id=episode_id,
            expected_policy=policy,
            expected_solver_seed=solver_seed,
            metric_iteration_budget=(
                int(job["metric_iteration_budget"])
                if job.get("metric_iteration_budget") is not None
                else None
            ),
            collection_root=output_root,
        )
        trace_validation_seconds = time.perf_counter() - validation_started
        if validated["summary"] != summary:
            raise ClosedLoopTraceError("new trace summary mismatch")
        rename_started = time.perf_counter()
        os.replace(partial_path, trace_path)
        atomic_rename_seconds = time.perf_counter() - rename_started
        metadata_started = time.perf_counter()
        metadata = trace_file_metadata(trace_path)
        trace_metadata_seconds = time.perf_counter() - metadata_started
        finalized_at = time.perf_counter()
        episode_process_wall_seconds = finalized_at - started_wall
        episode_finalization_timings = {
            "finish_event_orchestration_seconds": finish_event_orchestration_seconds,
            "finish_trace_write_seconds": finish_trace_write_seconds,
            "trace_close_seconds": trace_close_seconds,
            "trace_validation_seconds": trace_validation_seconds,
            "atomic_rename_seconds": atomic_rename_seconds,
            "trace_metadata_seconds": trace_metadata_seconds,
            "post_algorithm_finalize_seconds": finalized_at
            - episode_finalize_started,
            "episode_process_wall_seconds": episode_process_wall_seconds,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": relative_trace,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            **metadata,
            "trace_event_count": int(validated["event_count"]),
            "initial_state_ref": initial_state_ref,
            "status": "ok",
            "summary": summary,
            "episode_finalization_timings": episode_finalization_timings,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row.get("layout_mode"),
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            "trace_file": None,
            "partial_trace_file": partial_path.relative_to(output_root).as_posix()
            if partial_path.is_file()
            else None,
            "status": "error",
            "summary": None,
            "error_kind": getattr(error, "kind", type(error).__name__),
            "error": f"{type(error).__name__}: {error}",
            "error_details": dict(getattr(error, "details", {}) or {}),
        }


def _with_time_budget_overrides(
    config: dict[str, Any],
    wall_time_budget_seconds: float | None,
    episode_process_timeout_seconds: float | None,
    environment_time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    result = {**config, "environment": dict(config["environment"])}
    if wall_time_budget_seconds is None:
        if (
            episode_process_timeout_seconds is not None
            or environment_time_limit_seconds is not None
        ):
            raise ValueError(
                "time-limit overrides require a wall-time budget override"
            )
        return result
    wall_budget = float(wall_time_budget_seconds)
    if not math.isfinite(wall_budget) or wall_budget <= 0.0:
        raise ValueError("wall-time budget override must be finite and positive")
    environment_limit = float(
        environment_time_limit_seconds
        if environment_time_limit_seconds is not None
        else wall_budget
    )
    if not math.isfinite(environment_limit) or environment_limit <= 0.0:
        raise ValueError(
            "environment time-limit override must be finite and positive"
        )
    process_timeout = float(
        episode_process_timeout_seconds
        if episode_process_timeout_seconds is not None
        else max(wall_budget, environment_limit) + 60.0
    )
    if (
        not math.isfinite(process_timeout)
        or process_timeout <= max(wall_budget, environment_limit)
    ):
        raise ValueError(
            "episode process timeout must be finite and greater than both the "
            "wall-time budget and environment time limit"
        )
    result["wall_time_budget_seconds"] = wall_budget
    result["episode_process_timeout_seconds"] = process_timeout
    result["environment"]["time_limit"] = environment_limit
    return result


def _with_stopping_rule(
    config: dict[str, Any], stopping_rule: str
) -> dict[str, Any]:
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"unsupported stopping rule: {stopping_rule}")
    result = {**config, "environment": dict(config["environment"])}
    result["stopping_rule"] = stopping_rule
    if stopping_rule == "run-to-completion":
        result["max_decisions"] = 0
        result["metric_iteration_budget"] = None
        result["wall_time_budget_seconds"] = None
        result["episode_process_timeout_seconds"] = None
        result["environment"]["time_limit"] = 0.0
        result["environment"]["unlimited_time"] = True
        result["environment"]["max_repair_iterations"] = 0
        result["deterministic_pp_replay"] = True
        return result
    if stopping_rule in {"wall-clock", "wall-clock-fixed-metric"}:
        result["max_decisions"] = 0
        result["environment"]["max_repair_iterations"] = 0
    if stopping_rule == "wall-clock":
        result["metric_iteration_budget"] = None
    elif stopping_rule == "wall-clock-fixed-metric":
        metric_budget = result.get("metric_iteration_budget")
        if type(metric_budget) is not int or metric_budget <= 0:
            raise ValueError(
                "wall-clock-fixed-metric requires a positive integer "
                "metric_iteration_budget"
            )
        result["deterministic_pp_replay"] = True
    return result


def _collection_policy_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "episode_count": len(results),
        "success_count": sum(
            bool(dict(row.get("summary") or {}).get("success")) for row in results
        ),
        "error_count": sum(
            str(row.get("status")) not in {"ok", "resumed"} for row in results
        ),
        "timeout_count": sum(str(row.get("status")) == "timeout" for row in results),
    }


def run_closed_loop_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
    trace_format: str = TRACE_FORMAT_DELTA_GZIP_V2,
    controller: str | None = None,
    feature_backend: str = "auto",
    controller_bundle: str | Path | None = None,
    diagnostic_shadow_bundles: dict[str, str | Path] | None = None,
    feature_shadow_validation: bool = False,
    controller_runtime: str = "reference",
    verification_profile: str = "audit",
    v3_s3_bundle: str | Path | None = None,
    job_keys: set[tuple[str, int]] | None = None,
    cohort_job_keys: set[tuple[str, int]] | None = None,
    wall_time_budget_seconds: float | None = None,
    episode_process_timeout_seconds: float | None = None,
    environment_time_limit_seconds: float | None = None,
    qualification_process_timeout_seconds: float | None = None,
    stopping_rule: str = "historical",
    qualification_source: str | Path | None = None,
    use_global_collection_lock: bool = True,
    topology_boundary_augmentation: dict[str, Any] | None = None,
    structpool_augmentation: dict[str, Any] | None = None,
    hybridstructpool_augmentation: dict[str, Any] | None = None,
    repair_seed_policy: str | None = None,
    deterministic_pp_replay: bool | None = None,
    episode_overrides: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported closed-loop config")
    if not isinstance(config.get("deterministic_pp_replay", False), bool):
        raise ValueError("deterministic_pp_replay must be boolean")
    if deterministic_pp_replay is not None:
        if not isinstance(deterministic_pp_replay, bool):
            raise ValueError("deterministic_pp_replay override must be boolean")
        config = {**config, "deterministic_pp_replay": deterministic_pp_replay}
    topology_boundary_augmentation = validate_topology_boundary_augmentation(
        topology_boundary_augmentation
    )
    structpool_augmentation = validate_structpool_augmentation(
        structpool_augmentation
    )
    hybridstructpool_augmentation = validate_any_hybridstructpool_augmentation(
        hybridstructpool_augmentation
    )
    repair_seed_policy = validate_repair_seed_policy(
        repair_seed_policy
        if repair_seed_policy is not None
        else config.get("repair_seed_policy")
    )
    if sum(
        value is not None
        for value in (
            topology_boundary_augmentation,
            structpool_augmentation,
            hybridstructpool_augmentation,
        )
    ) > 1:
        raise ValueError("topology, StructPool, and HybridStructPool are exclusive")
    if topology_boundary_augmentation is not None:
        config = {
            **config,
            "proposal": {
                **dict(config["proposal"]),
                "topology_boundary": topology_boundary_augmentation,
            },
        }
    if structpool_augmentation is not None:
        config = {
            **config,
            "proposal": {
                **dict(config["proposal"]),
                "structpool": structpool_augmentation,
            },
        }
    if hybridstructpool_augmentation is not None:
        config = {
            **config,
            "proposal": {
                **dict(config["proposal"]),
                "hybridstructpool": hybridstructpool_augmentation,
            },
        }
    config = _with_time_budget_overrides(
        config,
        wall_time_budget_seconds,
        episode_process_timeout_seconds,
        environment_time_limit_seconds,
    )
    config = _with_stopping_rule(config, stopping_rule)
    if qualification_process_timeout_seconds is not None and (
        not math.isfinite(float(qualification_process_timeout_seconds))
        or float(qualification_process_timeout_seconds) <= 0.0
    ):
        raise ValueError("qualification process timeout must be finite and positive")
    if trace_format not in TRACE_FORMATS:
        raise ValueError(f"unsupported trace format: {trace_format}")
    if feature_backend not in FEATURE_BACKENDS:
        raise ValueError(f"unsupported feature backend: {feature_backend}")
    if controller_runtime not in CONTROLLER_RUNTIMES:
        raise ValueError(f"unsupported controller runtime: {controller_runtime}")
    if verification_profile not in VERIFICATION_PROFILES:
        raise ValueError(f"unsupported verification profile: {verification_profile}")
    controller_mode, controller_root, controller_manifest = resolve_controller_mode(
        project_root, controller, controller_bundle
    )
    if topology_boundary_augmentation is not None and controller_mode not in {
        "v2-full",
        "stride-augcontrol-v1",
        "stride-maprank-v1",
    }:
        raise ValueError(
            "topology-boundary augmentation requires v2-full, "
            "stride-augcontrol-v1, or stride-maprank-v1"
        )
    if structpool_augmentation is not None and controller_mode != "v2-full":
        raise ValueError("StructPool augmentation requires frozen v2-full")
    if hybridstructpool_augmentation is not None and controller_mode != "v2-full":
        raise ValueError("HybridStructPool augmentation requires frozen v2-full")
    diagnostic_shadow_roots: dict[str, Path] = {}
    diagnostic_shadow_manifests: dict[str, dict[str, Any]] = {}
    if diagnostic_shadow_bundles:
        if controller_mode != "v2-full":
            raise ValueError("diagnostic shadows require v2-full execution")
        shadow_ids = set(diagnostic_shadow_bundles)
        if not shadow_ids or not shadow_ids <= set(DIAGNOSTIC_CONTROLLER_IDS):
            raise ValueError("diagnostic shadow contains an unregistered STRIDE bundle")
        for shadow_id, raw_path in sorted(diagnostic_shadow_bundles.items()):
            shadow_root = Path(str(raw_path))
            if not shadow_root.is_absolute():
                shadow_root = project_root / shadow_root
            shadow_root = shadow_root.resolve()
            loaded_shadow = load_controller_bundle(shadow_root)
            if (
                str(loaded_shadow.manifest.get("controller_id")) != shadow_id
                or loaded_shadow.manifest.get("scientific_status")
                != "diagnostic_only"
                or loaded_shadow.manifest.get("default_replacement_allowed")
                is not False
            ):
                raise ValueError(f"invalid diagnostic shadow bundle: {shadow_id}")
            diagnostic_shadow_roots[shadow_id] = shadow_root
            diagnostic_shadow_manifests[shadow_id] = loaded_shadow.manifest
    v3_s3_root: Path | None = None
    v3_s3_manifest: dict[str, Any] | None = None
    if controller_mode == "v3-s3":
        v3_s3_root = Path(
            str(v3_s3_bundle or DEFAULT_V3_S3_BUNDLE)
        )
        if not v3_s3_root.is_absolute():
            v3_s3_root = project_root / v3_s3_root
        v3_s3_root = v3_s3_root.resolve()
        loaded_v3_s3 = load_v3_s3_bundle(v3_s3_root)
        v3_s3_manifest = loaded_v3_s3.manifest
        if not bool(v3_s3_manifest.get("native_audit_completed")):
            raise ValueError("v3-s3 requires a completed native audit")
        proposal_sizes = set(map(int, config["proposal"]["neighborhood_sizes"]))
        if proposal_sizes != {4, 8, 16}:
            raise ValueError(
                "v3-s3 requires the frozen 4/8/16 candidate space"
            )
    elif v3_s3_bundle is not None:
        raise ValueError("v3_s3_bundle is only valid with v3-s3")
    storage_fp = storage_fingerprint(trace_format)
    split = str(config["split"])
    solver_seeds = configured_solver_seeds(config)
    policies = configured_policies(config)
    phases = {"qualify", "all", *policies}
    if phase not in phases:
        raise ValueError(f"unsupported closed-loop phase: {phase}")
    all_rows = _load_dataset_rows(dataset_root, [split])
    rows = _selected_rows(all_rows, task_ids)
    available_job_keys = {
        (str(row["task_id"]), int(solver_seed))
        for row in rows
        for solver_seed in solver_seeds
    }
    normalized_job_keys = (
        {(str(task_id), int(seed)) for task_id, seed in job_keys}
        if job_keys is not None
        else None
    )
    normalized_cohort_job_keys = (
        {(str(task_id), int(seed)) for task_id, seed in cohort_job_keys}
        if cohort_job_keys is not None
        else None
    )
    normalized_episode_overrides = {
        (str(task_id), int(solver_seed)): dict(value)
        for (task_id, solver_seed), value in dict(episode_overrides or {}).items()
    }
    if (
        normalized_cohort_job_keys is not None
        and not normalized_cohort_job_keys <= available_job_keys
    ):
        unknown = sorted(normalized_cohort_job_keys - available_job_keys)
        raise ValueError(f"closed-loop cohort contains unknown task/seed pairs: {unknown}")
    if normalized_job_keys is not None and not normalized_job_keys <= available_job_keys:
        unknown = sorted(normalized_job_keys - available_job_keys)
        raise ValueError(f"closed-loop job filter contains unknown task/seed pairs: {unknown}")
    if not set(normalized_episode_overrides) <= available_job_keys:
        unknown = sorted(set(normalized_episode_overrides) - available_job_keys)
        raise ValueError(
            f"closed-loop episode overrides contain unknown task/seed pairs: {unknown}"
        )
    if normalized_job_keys is not None and not set(
        normalized_episode_overrides
    ) <= normalized_job_keys:
        raise ValueError("closed-loop episode overrides are outside the execution slice")
    if (
        normalized_job_keys is not None
        and normalized_cohort_job_keys is not None
        and not normalized_job_keys <= normalized_cohort_job_keys
    ):
        raise ValueError("closed-loop execution slice is outside the registered cohort")
    formal = task_ids is None and bool(config.get("formal", True))
    design = closed_loop_dataset_design(
        all_rows, split, dict(config.get("dataset_design", {}))
    )
    dataset_mode = str(config.get("dataset_design", {}).get("mode", "structured"))
    if dataset_mode in {"movingai_ood", "balanced_wall_clock"}:
        registered_ids = set(map(str, config["dataset_design"].get("historical_map_ids", [])))
        current_ids = {str(row["map_id"]) for row in all_rows}
        overlap = sorted(current_ids & registered_ids)
        isolation = {
            "passed": not overlap,
            "mode": f"{dataset_mode}_map_id",
            "current_map_ids": sorted(current_ids),
            "historical_overlap": overlap,
        }
    else:
        isolation = _seed_isolation(
            all_rows, list(config.get("reference_datasets", [])), project_root
        )
    frozen_root = Path(str(config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    bundle = load_frozen_policy_bundle(frozen_root, dict(config["model_registration"]))
    if controller_manifest is not None:
        expected_source_manifest = str(
            config["model_registration"].get("deployment_manifest_sha256", "")
        ).lower()
        actual_source_manifest = str(
            controller_manifest.get("source_bundle", {}).get("manifest_sha256", "")
        ).lower()
        registered_alternatives = dict(
            config["model_registration"].get("registered_controller_bundles", {})
        )
        controller_id = str(controller_manifest.get("controller_id", ""))
        alternative = dict(registered_alternatives.get(controller_id, {}))
        alternative_manifest_sha = str(
            alternative.get("controller_manifest_sha256", "")
        ).lower()
        loaded_manifest_sha = _sha256(controller_root / "controller_manifest.json")
        registered_alternative = bool(
            controller_id
            and alternative_manifest_sha
            and loaded_manifest_sha == alternative_manifest_sha
        )
        if actual_source_manifest != expected_source_manifest and not registered_alternative:
            raise ValueError("controller-v2 was built from a different v1 deployment bundle")
    effective_workers = int(workers or config["workers"])
    dataset_fp = _dataset_fingerprint(dataset_root)
    implementation = controller_implementation_fingerprint(project_root)
    effective = {
        **config,
        "task_ids_override": task_ids,
        "cohort_job_keys_override": (
            [list(value) for value in sorted(normalized_cohort_job_keys)]
            if normalized_cohort_job_keys is not None
            else None
        ),
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "diagnostic_shadow_controllers": sorted(diagnostic_shadow_roots),
        "deterministic_pp_replay": bool(
            config.get("deterministic_pp_replay", False)
        ),
        "repair_seed_policy": repair_seed_policy,
        "controller_bundle": str(controller_root),
        "diagnostic_shadow_bundles": {
            shadow_id: str(path)
            for shadow_id, path in diagnostic_shadow_roots.items()
        },
        "feature_shadow_validation": bool(
            feature_shadow_validation
            or verification_profile == "audit"
            and controller_runtime in {"optimized", "auto"}
        ),
        "v3_s3_bundle": (
            str(v3_s3_root) if v3_s3_root is not None else None
        ),
        "episode_override_fingerprints": {
            f"{task_id}::{solver_seed}": _fingerprint(value)
            for (task_id, solver_seed), value in sorted(
                normalized_episode_overrides.items()
            )
        },
        "qualification_process_timeout_seconds": (
            float(qualification_process_timeout_seconds)
            if qualification_process_timeout_seconds is not None
            else None
        ),
    }
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration_fingerprint": config_fp,
            "freeze_manifest": bundle.manifest,
            "controller_bundle_manifest": controller_manifest,
            "diagnostic_shadow_bundle_manifests": diagnostic_shadow_manifests,
            "v3_s3_bundle_manifest": v3_s3_manifest,
            "controller_implementation": implementation,
        }
    )
    registered_job_keys = normalized_cohort_job_keys or available_job_keys
    estimate = {
        "task_count": len(rows),
        "reset_count": len(registered_job_keys),
        "solver_seeds": list(solver_seeds),
        "policies": list(policies),
        "policy_episode_count": len(registered_job_keys) * len(policies),
        "maximum_decisions_per_episode": (
            int(config["max_decisions"])
            if int(config["max_decisions"]) > 0
            else None
        ),
        "stopping_rule": stopping_rule,
        "maximum_proposals_per_decision": int(config["proposal"]["max_seed_agents"])
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["trials"]),
        "maximum_candidates_per_decision": len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["candidates_per_family"])
        + int(
            dict(config["proposal"].get("topology_boundary") or {}).get(
                "maximum_added_candidates", 0
            )
        ),
        "workers": effective_workers,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "wall_time_budget_seconds": (
            float(config["wall_time_budget_seconds"])
            if config.get("wall_time_budget_seconds") is not None
            else None
        ),
        "episode_process_timeout_seconds": (
            float(config["episode_process_timeout_seconds"])
            if config.get("episode_process_timeout_seconds") is not None
            else None
        ),
        "environment_time_limit_seconds": float(config["environment"]["time_limit"]),
        "environment_max_repair_iterations": int(
            config["environment"]["max_repair_iterations"]
        ),
    }
    if dry_run:
        return {
            "schema": CLOSED_LOOP_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "formal": formal,
            "run_fingerprint": run_fp,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "frozen_models": bundle.manifest,
            "controller": controller_mode,
            "feature_backend": feature_backend,
            "controller_runtime": controller_runtime,
            "verification_profile": verification_profile,
            "feature_schema_id": (
                V3_S3_FEATURE_SCHEMA_ID
                if controller_mode == "v3-s3"
                else FEATURE_SCHEMA_ID
                if controller_mode != "official_adaptive"
                else None
            ),
            "feature_schema_sha256": (
                V3_S3_FEATURE_SCHEMA_SHA256
                if controller_mode == "v3-s3"
                else FEATURE_SCHEMA_SHA256
                if controller_mode != "official_adaptive"
                else None
            ),
            "controller_bundle": controller_manifest,
            "diagnostic_shadow_bundles": diagnostic_shadow_manifests,
            "v3_s3_bundle": v3_s3_manifest,
            "controller_implementation": implementation,
            "estimate": estimate,
        }
    run_config = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "trace_format": trace_format,
        "storage_fingerprint": storage_fp,
        "formal": formal,
        "dataset_design": design,
        "seed_isolation": isolation,
        "frozen_models": bundle.manifest,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "feature_schema_id": (
            V3_S3_FEATURE_SCHEMA_ID
            if controller_mode == "v3-s3"
            else FEATURE_SCHEMA_ID
            if controller_mode != "official_adaptive"
            else None
        ),
        "feature_schema_sha256": (
            V3_S3_FEATURE_SCHEMA_SHA256
            if controller_mode == "v3-s3"
            else FEATURE_SCHEMA_SHA256
            if controller_mode != "official_adaptive"
            else None
        ),
        "controller_bundle": controller_manifest,
        "diagnostic_shadow_bundles": diagnostic_shadow_manifests,
        "v3_s3_bundle": v3_s3_manifest,
        "controller_implementation": implementation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different closed-loop run")
        existing_format = str(existing.get("trace_format", TRACE_FORMAT_FULL_V1))
        existing_storage = str(
            existing.get("storage_fingerprint", storage_fingerprint(existing_format))
        )
        if existing_format != trace_format or existing_storage != storage_fp:
            raise ValueError("output contains a different closed-loop trace format")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    sequence = policies if phase == "all" else (phase,)
    if phase == "all":
        sequence = ("qualify",) + policies
    summary: dict[str, Any] = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fp,
        "trace_format": trace_format,
        "storage_fingerprint": storage_fp,
        "formal": formal,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "estimate": estimate,
    }
    for current in sequence:
        if current == "qualify":
            if qualification_source is not None:
                qualification_root = Path(qualification_source).resolve()
                source_run_path = qualification_root / "run_config.json"
                if not source_run_path.is_file():
                    raise ValueError("qualification source is missing run_config.json")
                source_run = _read_json(source_run_path)
                if _qualification_reuse_fingerprint(
                    source_run
                ) != _qualification_reuse_fingerprint(run_config):
                    raise ValueError(
                        "qualification source is incompatible with the current reset protocol"
                    )
                reused_results = _read_jsonl(
                    qualification_root / "qualification_manifest.jsonl"
                )
                expected_qualification_keys = (
                    normalized_job_keys
                    if normalized_job_keys is not None
                    else available_job_keys
                )
                results = [
                    value
                    for value in reused_results
                    if (
                        str(value["task_id"]),
                        int(value.get("solver_seed", solver_seeds[0])),
                    )
                    in expected_qualification_keys
                ]
            else:
                jobs = [
                    {
                        "row": row,
                        "solver_seed": solver_seed,
                        "dataset_root": str(dataset_root),
                        "environment": config["environment"],
                    }
                    for row in rows
                    for solver_seed in solver_seeds
                    if normalized_job_keys is None
                    or (str(row["task_id"]), int(solver_seed)) in normalized_job_keys
                ]
                with _CollectionRunLock(
                    output_root,
                    run_fp,
                    "closed-loop-qualification",
                    use_global_lock=use_global_collection_lock,
                ):
                    results = _run_jobs(
                        _qualification_worker,
                        jobs,
                        effective_workers,
                        phase="closed-loop-qualification",
                        output_root=output_root,
                        run_fingerprint=run_fp,
                        timeout_seconds=(
                            float(qualification_process_timeout_seconds)
                            if qualification_process_timeout_seconds is not None
                            else float(config["episode_process_timeout_seconds"])
                            if config.get("episode_process_timeout_seconds") is not None
                            else None
                        ),
                    )
            qualification_manifest = output_root / "qualification_manifest.jsonl"
            existing_results = (
                _read_jsonl(qualification_manifest)
                if qualification_manifest.is_file()
                else []
            )
            merged = {
                (str(value["task_id"]), int(value["solver_seed"])): value
                for value in existing_results
            }
            merged.update(
                {
                    (str(value["task_id"]), int(value["solver_seed"])): value
                    for value in results
                }
            )
            results = [merged[key] for key in sorted(merged)]
            _write_jsonl(qualification_manifest, results)
            report = closed_loop_qualification_report(
                rows,
                results,
                config,
                design,
                isolation,
                formal=formal,
                expected_job_keys=normalized_cohort_job_keys,
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
            continue
        qualification = closed_loop_qualification_report(
            rows,
            _read_jsonl(output_root / "qualification_manifest.jsonl"),
            config,
            design,
            isolation,
            formal=formal,
            expected_job_keys=normalized_cohort_job_keys,
        )
        if not qualification["passed"]:
            raise ValueError("closed-loop qualification failed; policy execution is forbidden")
        policy_manifest = output_root / f"{current}_manifest.jsonl"
        existing_results = (
            _read_jsonl(policy_manifest) if policy_manifest.is_file() else []
        )
        existing_by_key = {
            (str(value["task_id"]), int(value["solver_seed"])): value
            for value in existing_results
        }
        jobs = [
            {
                "row": row,
                "policy": current,
                "solver_seed": solver_seed,
                "dataset_root": str(dataset_root),
                "environment": config["environment"],
                "proposal": config["proposal"],
                "max_decisions": int(config["max_decisions"]),
                "metric_iteration_budget": (
                    int(config["metric_iteration_budget"])
                    if config.get("metric_iteration_budget") is not None
                    else None
                ),
                "wall_time_budget_seconds": (
                    float(config["wall_time_budget_seconds"])
                    if config.get("wall_time_budget_seconds") is not None
                    else None
                ),
                "stopping_rule": stopping_rule,
                "safety_max_decisions": (
                    None
                    if stopping_rule == "run-to-completion"
                    else WALL_CLOCK_SAFETY_MAX_DECISIONS
                ),
                "frozen_models": str(frozen_root.resolve()),
                "model_registration": config["model_registration"],
                "output_root": str(output_root),
                "run_fingerprint": run_fp,
                "trace_format": trace_format,
                "storage_fingerprint": storage_fp,
                "controller": controller_mode,
                "feature_backend": feature_backend,
                "controller_runtime": controller_runtime,
                "verification_profile": verification_profile,
                "controller_bundle": str(controller_root),
                "diagnostic_shadow_bundles": {
                    shadow_id: str(path)
                    for shadow_id, path in diagnostic_shadow_roots.items()
                },
                "feature_shadow_validation": bool(
                    feature_shadow_validation
                    or verification_profile == "audit"
                    and controller_runtime in {"optimized", "auto"}
                ),
                "proposal_shadow_validation": bool(
                    verification_profile == "audit"
                    and controller_runtime in {"optimized", "auto"}
                ),
                "v3_s3_bundle": (
                    str(v3_s3_root) if v3_s3_root is not None else None
                ),
                "proposal_state_verification": (
                    "always" if verification_profile == "audit" else "sampled"
                ),
                "resume": resume,
                "require_finalization_timings": True,
                "deterministic_pp_replay": bool(
                    config.get("deterministic_pp_replay", False)
                ),
                "repair_seed_policy": repair_seed_policy,
                "existing_manifest_row": existing_by_key.get(
                    (str(row["task_id"]), int(solver_seed))
                ),
                "episode_override": normalized_episode_overrides.get(
                    (str(row["task_id"]), int(solver_seed))
                ),
            }
            for row in rows
            for solver_seed in solver_seeds
            if normalized_job_keys is None
            or (str(row["task_id"]), int(solver_seed)) in normalized_job_keys
        ]
        with _CollectionRunLock(
            output_root,
            run_fp,
            f"closed-loop-{current}",
            use_global_lock=use_global_collection_lock,
        ):
            results = _run_jobs(
                _closed_loop_episode_worker,
                jobs,
                effective_workers,
                phase=f"closed-loop-{current}",
                output_root=output_root,
                run_fingerprint=run_fp,
                timeout_seconds=(
                    float(qualification_process_timeout_seconds)
                    if current == "qualify"
                    and qualification_process_timeout_seconds is not None
                    else float(config["episode_process_timeout_seconds"])
                    if config.get("episode_process_timeout_seconds") is not None
                    else None
                ),
            )
        merged = {
            (str(value["task_id"]), int(value["solver_seed"])): value
            for value in existing_results
        }
        merged.update(
            {
                (str(value["task_id"]), int(value["solver_seed"])): value
                for value in results
            }
        )
        results = [merged[key] for key in sorted(merged)]
        _write_jsonl(policy_manifest, results)
        summary[current] = _collection_policy_summary(results)
        if summary[current]["error_count"] and phase == "all":
            break
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "CLOSED_LOOP_SCHEMA",
    "CONTROLLER_MODES",
    "EXECUTABLE_CONTROLLER_MODES",
    "DEFAULT_CONTROLLER_BUNDLE",
    "STOPPING_RULES",
    "CollectionLockError",
    "LEARNED_POLICIES",
    "REPAIR_TIMING_SCHEMA",
    "REPAIR_TIMING_SCHEMA_V1",
    "REPAIR_TIMING_SCHEMA_V2",
    "REPAIR_TIMING_SCHEMAS",
    "FIXED_POLICIES",
    "POLICIES",
    "SUPPORTED_POLICIES",
    "closed_loop_dataset_design",
    "movingai_ood_dataset_design",
    "closed_loop_qualification_report",
    "configured_policies",
    "configured_solver_seeds",
    "feature_range_diagnostic",
    "export_portable_policy_bundle",
    "fixed_budget_conflict_auc",
    "wall_clock_conflict_auc",
    "generate_online_candidates",
    "load_frozen_policy_bundle",
    "online_candidate_rows",
    "PortablePairwiseModel",
    "proposal_random_seed",
    "proposal_random_seeds",
    "repair_random_seed",
    "resolve_controller_mode",
    "run_closed_loop_collection",
    "score_online_candidates",
    "verify_portable_policy_bundle",
]
